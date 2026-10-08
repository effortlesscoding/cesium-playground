// Shared plumbing for the GPU-instanced renderers (C: Shader, D: Ground).
//
// Both draw every field as one instance of a small static mesh, fed by the same 40 bytes of
// per-field data, merge overlapping fills with the stencil buffer, and live in the scene as a
// Primitive-like object. They differ only in the mesh and shaders (see ShaderRenderer.js and
// GroundRenderer.js).
import {
  Buffer,
  BufferUsage,
  BoundingSphere,
  Cartesian3,
  ComponentDatatype,
  DrawCommand,
  IndexDatatype,
  Pass,
  PrimitiveType,
  StencilFunction,
  StencilOperation,
  VertexArray,
} from "cesium";
import { TWO_PI, DEG, clockSpan, toEcef } from "../geo.js";

// Per-instance dynamic data, in floats: centerHigh(3) centerLow(3) shape(4).
export const FLOATS = 10;
export const STRIDE = FLOATS * 4;
// Same sphere radius the CPU path (geo.js destinations) uses for great-circle distances.
export const MEAN_EARTH_RADIUS = 6371008.8;
// Every field is drawn in the same aqua; the fill is see-through, the outline is solid.
export const AQUA_GLSL = "vec3(0.0, 0.9, 1.0)";
export const FILL_ALPHA = 0.3;
// Visible outline width in pixels. It is drawn just outside the shape's edge.
export const OUTLINE_PX = 1.5;

// Stencil bit (one of Cesium's classification bits, which nothing here uses) marking pixels some
// fill has covered. The scene clears it every frame.
//  - Fill: each pixel passes the test once, so overlapping fields merge into one flat color
//    instead of compositing on top of each other.
//  - Outline: drawn only where no fill covers the pixel, so edges that lie inside another
//    field vanish and what's left is the outline of the union.
export const COVERED_BIT = 0x01;
const KEEP = { fail: StencilOperation.KEEP, zFail: StencilOperation.KEEP, zPass: StencilOperation.KEEP };
const MARK = { fail: StencilOperation.KEEP, zFail: StencilOperation.KEEP, zPass: StencilOperation.REPLACE };
const stencilTest = (operation) => ({
  enabled: true,
  frontFunction: StencilFunction.NOT_EQUAL,
  backFunction: StencilFunction.NOT_EQUAL,
  reference: COVERED_BIT,
  mask: COVERED_BIT,
  frontOperation: operation,
  backOperation: operation,
});
/** Render-state pieces: the fill marks covered pixels, the outline only reads the mark. */
export const FILL_STENCIL = { stencilTest: stencilTest(MARK), stencilMask: COVERED_BIT };
export const OUTLINE_STENCIL = { stencilTest: stencilTest(KEEP), stencilMask: 0 };

/** Instanced attribute bindings for the 40-byte per-field record, at locations 1, 2, 3. */
function instanceAttributes(buffer) {
  const at = (index, size, offsetInBytes) => ({
    index,
    vertexBuffer: buffer,
    componentsPerAttribute: size,
    componentDatatype: ComponentDatatype.FLOAT,
    offsetInBytes,
    strideInBytes: STRIDE,
    instanceDivisor: 1,
  });
  return [at(1, 3, 0), at(2, 3, 12), at(3, 4, 24)];
}

/**
 * A Primitive-like scene object that draws a static mesh once per field. Subclasses implement
 * buildCommands(context) using makeCommand(); everything else (buffers, upload, culling,
 * lifetime) is handled here.
 */
export class InstancedFieldsPrimitive {
  constructor() {
    this.show = true;
    this.meshKey = "";
    this.meshOptions = null;
    this.count = 0;
    this.boundingSphere = new BoundingSphere();
    this.instances = null; // Float32Array, FLOATS per field; staged on the main thread
    this.meshDirty = true;
    this.instancesDirty = false;
    this.capacity = 0;
    this.instanceBuffer = null;
    this.commands = [];
    this.resources = [];
    this.lastUploadMs = 0;
  }

  /** Rebuilds the mesh and commands when the options change. */
  setMesh(options) {
    const key = JSON.stringify(options);
    if (key === this.meshKey) return;
    this.meshKey = key;
    this.meshOptions = options;
    this.meshDirty = true;
  }

  setInstances(instances, count) {
    this.instances = instances;
    this.instancesDirty = true;
    this.count = count;
  }

  releaseMesh() {
    for (const r of this.resources) if (!r.isDestroyed()) r.destroy();
    this.resources = [];
    this.commands = [];
  }

  releaseInstanceBuffer() {
    this.instanceBuffer?.destroy();
    this.instanceBuffer = null;
  }

  update(frameState) {
    if (!this.show || this.count === 0 || !this.instances) return;
    const context = frameState.context;

    if (this.instanceBuffer && this.capacity !== this.count) {
      this.releaseInstanceBuffer();
      this.releaseMesh();
      this.meshDirty = true;
    }
    if (!this.instanceBuffer) {
      this.capacity = this.count;
      this.instanceBuffer = Buffer.createVertexBuffer({
        context,
        sizeInBytes: this.count * STRIDE,
        usage: BufferUsage.DYNAMIC_DRAW,
      });
      // Shared by every vertex array below; we destroy it ourselves.
      this.instanceBuffer.vertexArrayDestroyable = false;
      this.meshDirty = true;
      this.instancesDirty = true;
    }
    if (this.meshDirty) {
      this.releaseMesh();
      this.meshDirty = false;
      this.buildCommands(context, this.meshOptions);
    }

    if (this.instancesDirty) {
      const t0 = performance.now();
      this.instanceBuffer.copyFromArrayView(this.instances.subarray(0, this.count * FLOATS));
      this.lastUploadMs = performance.now() - t0;
      this.instancesDirty = false;
    }

    if (!frameState.passes.render) return;
    for (const command of this.commands) {
      command.instanceCount = this.count;
      command.boundingVolume = this.boundingSphere;
      frameState.commandList.push(command);
    }
  }

  /**
   * Builds one instanced draw command.
   * `attributes` are the static per-vertex streams: { data: Float32Array, size, index } where
   * index is the shader attribute location (0 and, optionally, 4). The per-field record is
   * bound at locations 1-3.
   */
  makeCommand(context, { attributes, indices, shaderProgram, renderState, uniformMap }) {
    const vertexCount = attributes[0].data.length / attributes[0].size;
    const big = vertexCount > 65535;
    const indexBuffer = Buffer.createIndexBuffer({
      context,
      typedArray: big ? new Uint32Array(indices) : new Uint16Array(indices),
      usage: BufferUsage.STATIC_DRAW,
      indexDatatype: big ? IndexDatatype.UNSIGNED_INT : IndexDatatype.UNSIGNED_SHORT,
    });
    const meshAttributes = attributes.map(({ data, size, index }) => {
      const vertexBuffer = Buffer.createVertexBuffer({ context, typedArray: data, usage: BufferUsage.STATIC_DRAW });
      this.resources.push(vertexBuffer);
      return { index, vertexBuffer, componentsPerAttribute: size, componentDatatype: ComponentDatatype.FLOAT };
    });
    const vertexArray = new VertexArray({
      context,
      indexBuffer,
      attributes: [...meshAttributes, ...instanceAttributes(this.instanceBuffer)],
    });
    this.resources.push(vertexArray, indexBuffer);
    return new DrawCommand({
      primitiveType: PrimitiveType.TRIANGLES,
      vertexArray,
      shaderProgram,
      renderState,
      uniformMap,
      // Opaque pass, not translucent: translucent commands may be rendered into OIT's own
      // framebuffer (no stencil). We blend by hand and never write depth, so nothing else
      // is affected, and OIT can stay on for the rest of the scene. The scene's depth texture
      // (globe + 3D tiles) is also already complete when this pass runs.
      pass: Pass.OPAQUE,
      owner: this,
      modelMatrix: undefined,
    });
  }

  isDestroyed() {
    return false;
  }

  destroy() {
    this.releaseMesh();
    this.releaseInstanceBuffer();
    return undefined;
  }
}

// Reused across ticks.
const scratchEcef = new Float64Array(3);
const scratchSum = new Cartesian3();

/** Splits a double into float32-exact high/low parts, like Cesium's EncodedCartesian3. */
function splitInto(out, offset, value) {
  const high = Math.floor(value / 65536) * 65536;
  out[offset] = high;
  out[offset + 3] = value - high;
}

/**
 * Packs the fields into `data` (FLOATS per field) and fits `sphere` around them, padded by
 * `extra` meters, for culling.
 */
function packFields(fields, height, data, sphere, extra) {
  const n = fields.length;
  let cx = 0, cy = 0, cz = 0, maxRadius = 0;
  for (let i = 0; i < n; i++) {
    const f = fields[i];
    toEcef(f.lat * DEG, f.lon * DEG, height, scratchEcef, 0);
    const o = i * FLOATS;
    splitInto(data, o, scratchEcef[0]);
    splitInto(data, o + 1, scratchEcef[1]);
    splitInto(data, o + 2, scratchEcef[2]);
    const span = clockSpan(f);
    const full = span < 1e-9;
    data[o + 6] = f.radius;
    data[o + 7] = (f.startClock / 12) * TWO_PI;
    data[o + 8] = full ? TWO_PI : (span / 12) * TWO_PI;
    data[o + 9] = full ? 1 : 0;
    cx += scratchEcef[0];
    cy += scratchEcef[1];
    cz += scratchEcef[2];
    if (f.radius > maxRadius) maxRadius = f.radius;
  }

  // Centroid + farthest field + its radius.
  Cartesian3.fromElements(cx / n, cy / n, cz / n, scratchSum);
  let far = 0;
  for (let i = 0; i < n; i++) {
    const o = i * FLOATS;
    const dx = data[o] + data[o + 3] - scratchSum.x;
    const dy = data[o + 1] + data[o + 4] - scratchSum.y;
    const dz = data[o + 2] + data[o + 5] - scratchSum.z;
    far = Math.max(far, Math.hypot(dx, dy, dz));
  }
  Cartesian3.clone(scratchSum, sphere.center);
  sphere.radius = far + maxRadius * 1.1 + extra;
}

/**
 * Base for the renderers the engine drives. Subclasses say which primitive to create, which
 * mesh options to use and how many vertices/triangles a field costs.
 */
export class InstancedRenderer {
  constructor(viewer, stats) {
    this.viewer = viewer;
    this.stats = stats;
    this.primitive = null;
    this.instances = new Float32Array(0);
    this.awaitFirstFrame = false;
  }

  update(fields, params) {
    const t0 = performance.now();
    this.primitive ??= this.viewer.scene.primitives.add(this.createPrimitive());
    const prim = this.primitive;
    const n = fields.length;

    if (this.instances.length < n * FLOATS) this.instances = new Float32Array(n * FLOATS);
    packFields(fields, params.heightMeters, this.instances, prim.boundingSphere, this.boundingPadding(params));
    prim.setMesh(this.meshOptions(params));
    prim.setInstances(this.instances, n);
    this.awaitFirstFrame = true;

    const s = this.stats;
    s.updates++;
    s.mainBuildMs = performance.now() - t0;
    s.transferBytes = n * STRIDE;
    const { vertices, triangles } = this.meshCost(params);
    s.vertices = n * vertices;
    s.triangles = n * triangles;
    s.lastUpdateLatencyMs = s.mainBuildMs;
    return true;
  }

  /** Called after each rendered frame to record the GPU buffer upload cost. */
  onPostRender() {
    if (this.primitive && this.awaitFirstFrame) {
      this.stats.uploadMs = this.primitive.lastUploadMs;
      this.awaitFirstFrame = false;
    }
  }

  clear() {
    if (this.primitive) {
      this.viewer.scene.primitives.remove(this.primitive);
      this.primitive = null;
    }
  }

  destroy() {
    this.clear();
  }
}
