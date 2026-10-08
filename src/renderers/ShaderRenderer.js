// Approach C: GPU-instanced fields. One static unit mesh (a disk/pie parameterized by radial
// fraction + angular fraction) is drawn once per field; the vertex shader places every vertex
// on the globe from per-instance data (center, radius, start bearing, sweep).
//
// Per tick the only work is packing ~40 bytes per field and one bufferSubData. There are no
// geometry workers, no vertex/index rebuilds and no Primitive batching/combining.
//
// It is a Primitive-like object added to scene.primitives, so culling, log-depth, the
// opaque pass and frame-state plumbing are Cesium's; only the geometry math is ours.
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
  StencilFunction,
  StencilOperation,
  VertexArray,
} from "cesium";
import { TWO_PI, DEG, clockSpan, toEcef } from "../geo.js";

// Per-instance dynamic data, in floats: centerHigh(3) centerLow(3) shape(4).
const FLOATS = 10;
const STRIDE = FLOATS * 4;
// Same sphere radius the CPU path (geo.js destinations) uses for great-circle distances.
const MEAN_EARTH_RADIUS = 6371008.8;
// Every field is drawn in the same aqua; the fill is see-through, the outline is solid.
const FILL_ALPHA = 0.3;
// Visible outline width in pixels. The line is drawn twice this wide, centered on the field's
// edge, and only the half that falls outside every fill is visible (see COVERED_BIT).
const OUTLINE_PX = 1.5;
// Stencil bit (one of Cesium's classification bits, which nothing here uses) marking pixels some
// fill has covered. The scene clears it every frame.
//  - Fill: each pixel passes the test once, so overlapping fields merge into one flat color
//    instead of compositing on top of each other.
//  - Outline: drawn only where no fill covers the pixel, so edges that lie inside another
//    field vanish and what's left is the outline of the union.
const COVERED_BIT = 0x01;
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

const FILL_LOCATIONS = { unit: 0, centerHigh: 1, centerLow: 2, shape: 3 };
const LINE_LOCATIONS = { unit: 0, centerHigh: 1, centerLow: 2, shape: 3, other: 4 };

// Per-instance inputs plus the function that places a unit-mesh vertex on the globe.
const COMMON = `
in vec3 centerHigh;  // field center in ECEF (meters, incl. height), split into high/low parts
in vec3 centerLow;
in vec4 shape;       // x: radius m, y: start bearing rad (clockwise from north), z: sweep rad, w: 1 if full circle

const float R = ${MEAN_EARTH_RADIUS.toFixed(1)};
const vec3 AQUA = vec3(0.0, 0.9, 1.0);
const float A2_OVER_B2 = 1.006739496742; // WGS84 a^2 / b^2: ellipsoid normal ~ (x, y, z * a^2/b^2)

// u.x: radial fraction 0..1, u.y: angular fraction 0..1, u.z: 1 on the pie's radial edges.
vec4 fieldClip(vec3 u)
{
    // Full circles have no radial edges: collapse them onto the center (zero-length lines).
    float s = u.x * (1.0 - u.z * shape.w);
    float bearing = shape.y + u.y * shape.z;
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
    return czm_modelViewProjectionRelativeToEye * p;
}
`;

const FILL_VS = `
in vec3 unit;
${COMMON}
uniform float u_alpha;
out vec4 v_color;
void main()
{
    gl_Position = fieldClip(unit);
    v_color = vec4(AQUA, u_alpha);
}
`;

// Each outline segment is a screen-space quad: 4 vertices, each knowing its own end (unit.xyz),
// the opposite end (other) and which side of the line it sits on (unit.w = -1 or +1).
const LINE_VS = `
in vec4 unit;
in vec3 other;
${COMMON}
uniform float u_widthPx;
out vec4 v_color;
void main()
{
    vec4 a = fieldClip(unit.xyz);
    vec4 b = fieldClip(other);
    vec2 viewport = czm_viewport.zw;
    vec2 pa = a.xy / a.w * 0.5 * viewport;
    vec2 pb = b.xy / b.w * 0.5 * viewport;
    vec2 d = pb - pa;
    float len = length(d);
    vec2 dir = len > 1e-4 ? d / len : vec2(1.0, 0.0);
    vec2 normal = vec2(-dir.y, dir.x);
    float halfW = 0.5 * u_widthPx;
    // Extend each end by half the width (square caps) so consecutive segments leave no gaps.
    vec2 offsetPx = -dir * halfW + normal * unit.w * halfW;
    gl_Position = a;
    gl_Position.xy += offsetPx * 2.0 / viewport * a.w;
    v_color = vec4(AQUA, 1.0);
}
`;

const FS = `
in vec4 v_color;
void main()
{
    out_FragColor = czm_gammaCorrect(v_color);
}
`;

/** Static unit meshes shared by every field. */
function buildMesh(segments, rings, style) {
  const m = segments + 1; // samples per ring (last == first for full circles; harmless)

  // The fill is always built: with style "outline" it is drawn color-masked, purely to mark
  // the stencil so the outline can hide edges inside other fields.
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
  const out = { fill: { attributes: [{ data: pos, size: 3 }], indices: idx } };

  if (style !== "fill") {
    // Segments of the outline: the outer arc, then two radial edges (center -> arc start,
    // arc end -> center). Each becomes a quad of 4 vertices.
    const segs = [];
    for (let j = 0; j < segments; j++) segs.push([[1, j / segments, 0], [1, (j + 1) / segments, 0]]);
    segs.push([[0, 0, 1], [1, 0, 1]], [[1, 1, 1], [0, 1, 1]]);

    const self = new Float32Array(segs.length * 4 * 4);
    const other = new Float32Array(segs.length * 4 * 3);
    const lineIdx = [];
    segs.forEach(([A, B], i) => {
      const ends = [[A, B, -1], [A, B, 1], [B, A, -1], [B, A, 1]];
      ends.forEach(([s, o, side], k) => {
        const vi = i * 4 + k;
        self.set([s[0], s[1], s[2], side], vi * 4);
        other.set(o, vi * 3);
      });
      const b = i * 4;
      lineIdx.push(b, b + 1, b + 2, b + 2, b + 1, b + 3);
    });
    out.line = { attributes: [{ data: self, size: 4 }, { data: other, size: 3 }], indices: lineIdx };
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
    this.meshDirty = true;
    this.instancesDirty = false;
    this.capacity = 0;
    this.instanceBuffer = null;
    this.commands = [];
    this.resources = [];
    this.fillShader = null;
    this.lineShader = null;
    this.lastUploadMs = 0;
  }

  setMesh(style, segments, rings) {
    if (this.style === style && this.segments === segments && this.rings === rings) return;
    this.style = style;
    this.segments = segments;
    this.rings = rings;
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

  releaseInstanceBuffers() {
    this.instanceBuffer?.destroy();
    this.instanceBuffer = null;
  }

  update(frameState) {
    if (!this.show || this.count === 0 || !this.instances) return;
    const context = frameState.context;

    if (this.instanceBuffer && this.capacity !== this.count) {
      this.releaseInstanceBuffers();
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
      // Shared by the fill and outline vertex arrays; we destroy it ourselves.
      this.instanceBuffer.vertexArrayDestroyable = false;
      this.meshDirty = true;
      this.instancesDirty = true;
    }
    if (this.meshDirty) this.buildCommands(context);

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

    const program = (vs, locations) =>
      ShaderProgram.fromCache({
        context,
        vertexShaderSource: new ShaderSource({ sources: [vs] }),
        fragmentShaderSource: new ShaderSource({ sources: [FS] }),
        attributeLocations: locations,
      });
    this.fillShader ??= program(FILL_VS, FILL_LOCATIONS);
    this.lineShader ??= program(LINE_VS, LINE_LOCATIONS);

    const instanced = (index, size, offsetInBytes) => ({
      index,
      vertexBuffer: this.instanceBuffer,
      componentsPerAttribute: size,
      componentDatatype: ComponentDatatype.FLOAT,
      offsetInBytes,
      strideInBytes: STRIDE,
      instanceDivisor: 1,
    });

    const make = ({ attributes, indices }, shaderProgram, renderState, uniformMap) => {
      const big = attributes[0].data.length / attributes[0].size > 65535;
      const indexBuffer = Buffer.createIndexBuffer({
        context,
        typedArray: big ? new Uint32Array(indices) : new Uint16Array(indices),
        usage: BufferUsage.STATIC_DRAW,
        indexDatatype: big ? IndexDatatype.UNSIGNED_INT : IndexDatatype.UNSIGNED_SHORT,
      });
      const meshAttributes = attributes.map(({ data, size }, i) => {
        const vertexBuffer = Buffer.createVertexBuffer({ context, typedArray: data, usage: BufferUsage.STATIC_DRAW });
        this.resources.push(vertexBuffer);
        // Mesh attributes sit at locations 0 and (for the outline's far end) 4.
        return { index: i === 0 ? 0 : 4, vertexBuffer, componentsPerAttribute: size, componentDatatype: ComponentDatatype.FLOAT };
      });
      const vertexArray = new VertexArray({
        context,
        indexBuffer,
        attributes: [...meshAttributes, instanced(1, 3, 0), instanced(2, 3, 12), instanced(3, 4, 24)],
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
        // is affected, and OIT can stay on for the rest of the scene.
        pass: Pass.OPAQUE,
        owner: this,
        modelMatrix: undefined,
      });
    };

    // Order matters: the fill must run first so the stencil is complete when the outline tests it.
    // Within one pass Cesium keeps push order for commands sharing a bounding volume.
    const outlineOnly = this.style === "outline";
    this.commands.push(
      make(
        mesh.fill,
        this.fillShader,
        RenderState.fromCache({
          depthTest: { enabled: true },
          depthMask: false,
          blending: BlendingState.ALPHA_BLEND,
          colorMask: { red: !outlineOnly, green: !outlineOnly, blue: !outlineOnly, alpha: !outlineOnly },
          stencilTest: stencilTest(MARK),
          stencilMask: COVERED_BIT,
        }),
        { u_alpha: () => FILL_ALPHA },
      ),
    );
    if (mesh.line) {
      this.commands.push(
        make(
          mesh.line,
          this.lineShader,
          RenderState.fromCache({
            depthTest: { enabled: true },
            depthMask: false,
            blending: BlendingState.ALPHA_BLEND,
            stencilTest: stencilTest(KEEP),
            stencilMask: 0,
          }),
          { u_widthPx: () => OUTLINE_PX * 2 },
        ),
      );
    }
  }

  isDestroyed() {
    return false;
  }

  destroy() {
    this.releaseMesh();
    this.releaseInstanceBuffers();
    this.fillShader?.destroy();
    this.lineShader?.destroy();
    this.fillShader = this.lineShader = null;
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
    prim.setInstances(data, n);
    this.awaitFirstFrame = true;

    const s = this.stats;
    s.updates++;
    s.mainBuildMs = performance.now() - t0;
    s.transferBytes = n * STRIDE;
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
