// Approach C: GPU-instanced fields. One static unit mesh (a disk/pie parameterized by radial
// fraction + angular fraction) is drawn once per field; the vertex shader places every vertex
// on the globe from per-instance data (center, radius, start bearing, sweep, color).
//
// Per tick the only work is packing ~40 bytes per field and one bufferSubData. There are no
// geometry workers, no vertex/index rebuilds and no Primitive batching/combining.
//
// It is a Primitive-like object added to scene.primitives, so culling, log-depth, the
// translucent pass and frame-state plumbing are Cesium's; only the geometry math is ours.
import {
  BlendingState,
  BoundingSphere,
  Buffer,
  BufferUsage,
  Cartesian3,
  ComponentDatatype,
  DrawCommand,
  IndexDatatype,
  Pass,
  PrimitiveType,
  RenderState,
  ShaderProgram,
  ShaderSource,
  VertexArray,
} from "cesium";
import { TWO_PI, DEG, clockSpan, fieldRgb, toEcef } from "../geo.js";

// Per-instance dynamic data, in floats: centerHigh(3) centerLow(3) shape(4).
const FLOATS = 10;
const STRIDE = FLOATS * 4;
// Same sphere radius the CPU path (geo.js destinations) uses for great-circle distances.
const MEAN_EARTH_RADIUS = 6371008.8;
const FILL_ALPHA = 0.45;

const ATTRIBUTE_LOCATIONS = { unit: 0, centerHigh: 1, centerLow: 2, shape: 3, color: 4 };

const VS = `
in vec3 unit;        // x: radial fraction 0..1, y: angular fraction 0..1, z: 1 on the pie's radial edges
in vec3 centerHigh;  // field center in ECEF (meters, incl. height), split into high/low parts
in vec3 centerLow;
in vec4 shape;       // x: radius m, y: start bearing rad (clockwise from north), z: sweep rad, w: 1 if full circle
in vec4 color;       // normalized ubyte rgba
uniform float u_alpha;
out vec4 v_color;

const float R = ${MEAN_EARTH_RADIUS.toFixed(1)};
const float A2_OVER_B2 = 1.006739496742; // WGS84 a^2 / b^2: ellipsoid normal ~ (x, y, z * a^2/b^2)

void main()
{
    // Full circles have no radial edges: collapse them onto the center (zero-length lines).
    float s = unit.x * (1.0 - unit.z * shape.w);
    float bearing = shape.y + unit.y * shape.z;
    float theta = s * shape.x / R;

    vec3 c = centerHigh + centerLow;
    vec3 up = normalize(vec3(c.xy, c.z * A2_OVER_B2));
    vec3 east = normalize(vec3(-up.y, up.x, 0.0));
    vec3 north = cross(up, east);
    vec3 tangent = east * sin(bearing) + north * cos(bearing);

    // Point at angular distance theta along the great circle, as an offset from the center:
    // tangent * R sin(theta) - up * R (1 - cos(theta)). Written with sin^2(theta/2) so the
    // sag keeps full float precision at small theta.
    float h = sin(0.5 * theta);
    vec4 p = czm_translateRelativeToEye(centerHigh, centerLow);
    p.xyz += tangent * (R * sin(theta)) - up * (2.0 * R * h * h);

    gl_Position = czm_modelViewProjectionRelativeToEye * p;
    v_color = vec4(color.rgb, u_alpha);
}
`;

const FS = `
in vec4 v_color;
void main()
{
    out_FragColor = czm_gammaCorrect(v_color);
}
`;

/** Static unit mesh shared by every field. */
function buildMesh(segments, rings, style) {
  const m = segments + 1; // samples per ring (last == first for full circles; harmless)
  const out = {};

  if (style !== "outline") {
    const pos = new Float32Array((1 + rings * m) * 3); // vertex 0 = center (zeros)
    let p = 3;
    for (let k = 1; k <= rings; k++) {
      for (let j = 0; j < m; j++) {
        pos[p++] = k / rings;
        pos[p++] = j / segments;
        pos[p++] = 0;
      }
    }
    const v = (k, j) => 1 + (k - 1) * m + j;
    const idx = [];
    for (let j = 0; j < segments; j++) idx.push(0, v(1, j), v(1, j + 1));
    for (let k = 2; k <= rings; k++) {
      for (let j = 0; j < segments; j++) {
        const a = v(k - 1, j), b = v(k - 1, j + 1), c = v(k, j), d = v(k, j + 1);
        idx.push(a, c, d, a, d, b);
      }
    }
    out.fill = { positions: pos, indices: idx, primitiveType: PrimitiveType.TRIANGLES };
  }

  if (style !== "fill") {
    // Outer arc, then two radial edges (center -> arc start, arc end -> center).
    const pos = new Float32Array((m + 4) * 3);
    let p = 0;
    for (let j = 0; j < m; j++) {
      pos[p++] = 1;
      pos[p++] = j / segments;
      pos[p++] = 0;
    }
    pos.set([0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1], p);
    const idx = [];
    for (let j = 0; j < segments; j++) idx.push(j, j + 1);
    idx.push(m, m + 1, m + 2, m + 3);
    out.line = { positions: pos, indices: idx, primitiveType: PrimitiveType.LINES };
  }
  return out;
}

class FieldsPrimitive {
  constructor() {
    this.show = true;
    this.style = "fill+outline";
    this.segments = 64;
    this.rings = 6;
    this.count = 0;
    this.boundingSphere = new BoundingSphere();
    // Staged by the renderer on the main thread, consumed in update().
    this.instances = null; // Float32Array, FLOATS per field
    this.colors = null; // Uint8Array, 4 per field
    this.meshDirty = true;
    this.instancesDirty = false;
    this.colorsDirty = false;
    this.capacity = 0;
    this.instanceBuffer = null;
    this.colorBuffer = null;
    this.commands = [];
    this.resources = [];
    this.shader = null;
    this.lastUploadMs = 0;
  }

  setMesh(style, segments, rings) {
    if (this.style === style && this.segments === segments && this.rings === rings) return;
    this.style = style;
    this.segments = segments;
    this.rings = rings;
    this.meshDirty = true;
  }

  setInstances(instances, colors, count, colorsChanged) {
    this.instances = instances;
    this.instancesDirty = true;
    if (colorsChanged) {
      this.colors = colors;
      this.colorsDirty = true;
    }
    this.count = count;
  }

  releaseMesh() {
    for (const r of this.resources) if (!r.isDestroyed()) r.destroy();
    this.resources = [];
    this.commands = [];
  }

  releaseInstanceBuffers() {
    this.instanceBuffer?.destroy();
    this.colorBuffer?.destroy();
    this.instanceBuffer = this.colorBuffer = null;
  }

  update(frameState) {
    if (!this.show || this.count === 0 || !this.instances) return;
    const context = frameState.context;

    if (this.instanceBuffer && this.capacity !== this.count) {
      this.releaseInstanceBuffers();
      this.releaseMesh();
      this.meshDirty = true;
      this.colorsDirty = true;
    }
    if (!this.instanceBuffer) {
      this.capacity = this.count;
      this.instanceBuffer = Buffer.createVertexBuffer({
        context,
        sizeInBytes: this.count * STRIDE,
        usage: BufferUsage.DYNAMIC_DRAW,
      });
      this.colorBuffer = Buffer.createVertexBuffer({
        context,
        sizeInBytes: this.count * 4,
        usage: BufferUsage.STATIC_DRAW,
      });
      // Shared by the fill and outline vertex arrays; we destroy them ourselves.
      this.instanceBuffer.vertexArrayDestroyable = false;
      this.colorBuffer.vertexArrayDestroyable = false;
      this.meshDirty = true;
      this.instancesDirty = true;
    }
    if (this.meshDirty) this.buildCommands(context);

    if (this.colorsDirty && this.colors) {
      this.colorBuffer.copyFromArrayView(this.colors);
      this.colorsDirty = false;
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

  buildCommands(context) {
    this.releaseMesh();
    this.meshDirty = false;
    const mesh = buildMesh(this.segments, this.rings, this.style);

    this.shader ??= ShaderProgram.fromCache({
      context,
      vertexShaderSource: new ShaderSource({ sources: [VS] }),
      fragmentShaderSource: new ShaderSource({ sources: [FS] }),
      attributeLocations: ATTRIBUTE_LOCATIONS,
    });

    const make = ({ positions, indices, primitiveType }, alpha) => {
      const big = positions.length / 3 > 65535;
      const indexBuffer = Buffer.createIndexBuffer({
        context,
        typedArray: big ? new Uint32Array(indices) : new Uint16Array(indices),
        usage: BufferUsage.STATIC_DRAW,
        indexDatatype: big ? IndexDatatype.UNSIGNED_INT : IndexDatatype.UNSIGNED_SHORT,
      });
      const unitBuffer = Buffer.createVertexBuffer({ context, typedArray: positions, usage: BufferUsage.STATIC_DRAW });
      const vertexArray = new VertexArray({
        context,
        indexBuffer,
        attributes: [
          { index: 0, vertexBuffer: unitBuffer, componentsPerAttribute: 3, componentDatatype: ComponentDatatype.FLOAT },
          { index: 1, vertexBuffer: this.instanceBuffer, componentsPerAttribute: 3, componentDatatype: ComponentDatatype.FLOAT, offsetInBytes: 0, strideInBytes: STRIDE, instanceDivisor: 1 },
          { index: 2, vertexBuffer: this.instanceBuffer, componentsPerAttribute: 3, componentDatatype: ComponentDatatype.FLOAT, offsetInBytes: 12, strideInBytes: STRIDE, instanceDivisor: 1 },
          { index: 3, vertexBuffer: this.instanceBuffer, componentsPerAttribute: 4, componentDatatype: ComponentDatatype.FLOAT, offsetInBytes: 24, strideInBytes: STRIDE, instanceDivisor: 1 },
          { index: 4, vertexBuffer: this.colorBuffer, componentsPerAttribute: 4, componentDatatype: ComponentDatatype.UNSIGNED_BYTE, normalize: true, instanceDivisor: 1 },
        ],
      });
      this.resources.push(vertexArray, unitBuffer, indexBuffer);
      return new DrawCommand({
        primitiveType,
        vertexArray,
        shaderProgram: this.shader,
        renderState: RenderState.fromCache({
          depthTest: { enabled: true },
          depthMask: false,
          blending: BlendingState.ALPHA_BLEND,
        }),
        uniformMap: { u_alpha: () => alpha },
        pass: Pass.TRANSLUCENT,
        owner: this,
        modelMatrix: undefined,
      });
    };

    // Fill first so the (opaque) outline blends on top of it.
    if (mesh.fill) this.commands.push(make(mesh.fill, FILL_ALPHA));
    if (mesh.line) this.commands.push(make(mesh.line, 1));
  }

  isDestroyed() {
    return false;
  }

  destroy() {
    this.releaseMesh();
    this.releaseInstanceBuffers();
    this.shader?.destroy();
    this.shader = null;
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

export class ShaderRenderer {
  constructor(viewer, stats) {
    this.viewer = viewer;
    this.stats = stats;
    this.primitive = null;
    this.instances = new Float32Array(0);
    this.colors = new Uint8Array(0);
    this.colorCount = 0;
    this.awaitFirstFrame = false;
  }

  ensurePrimitive() {
    if (!this.primitive) this.primitive = this.viewer.scene.primitives.add(new FieldsPrimitive());
    return this.primitive;
  }

  update(fields, params) {
    const t0 = performance.now();
    const prim = this.ensurePrimitive();
    const n = fields.length;
    const height = params.heightMeters;

    if (this.instances.length < n * FLOATS) this.instances = new Float32Array(n * FLOATS);
    const data = this.instances;

    // Colors depend only on field ids; rebuild when the field set changes size.
    const colorsChanged = this.colorCount !== n;
    if (colorsChanged) {
      this.colors = new Uint8Array(n * 4);
      this.colorCount = n;
    }

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
      if (colorsChanged) {
        const [r, g, b] = fieldRgb(f.id);
        const c = i * 4;
        this.colors[c] = Math.round(r * 255);
        this.colors[c + 1] = Math.round(g * 255);
        this.colors[c + 2] = Math.round(b * 255);
        this.colors[c + 3] = 255;
      }
    }

    // Bounding sphere for culling/sorting: centroid + farthest field + its radius.
    Cartesian3.fromElements(cx / n, cy / n, cz / n, scratchSum);
    let far = 0;
    for (let i = 0; i < n; i++) {
      const o = i * FLOATS;
      const dx = data[o] + data[o + 3] - scratchSum.x;
      const dy = data[o + 1] + data[o + 4] - scratchSum.y;
      const dz = data[o + 2] + data[o + 5] - scratchSum.z;
      far = Math.max(far, Math.hypot(dx, dy, dz));
    }
    Cartesian3.clone(scratchSum, prim.boundingSphere.center);
    prim.boundingSphere.radius = far + maxRadius * 1.1 + Math.abs(height) + 10;

    prim.setMesh(params.primitiveStyle === "points" ? "fill+outline" : params.primitiveStyle, params.segments, params.rings);
    prim.setInstances(data, this.colors, n, colorsChanged);
    this.awaitFirstFrame = true;

    const s = this.stats;
    s.updates++;
    s.mainBuildMs = performance.now() - t0;
    s.transferBytes = n * (STRIDE + (colorsChanged ? 4 : 0));
    const m = params.segments + 1;
    const fillVerts = 1 + params.rings * m;
    s.vertices = n * fillVerts;
    s.triangles = n * (params.segments * (1 + 2 * (params.rings - 1)));
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
