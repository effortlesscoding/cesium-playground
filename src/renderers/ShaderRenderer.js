// Approach C: GPU-instanced fields. One static unit mesh (a disk/pie parameterized by radial
// fraction + angular fraction) is drawn once per field; the vertex shader places every vertex
// on the globe from per-instance data (center, radius, start bearing, sweep).
//
// Per tick the only work is packing ~40 bytes per field and one bufferSubData. There are no
// geometry workers, no vertex/index rebuilds and no Primitive batching/combining.
//
// Shapes sit at a fixed height above the ellipsoid (they are not clamped to terrain; see
// GroundRenderer.js for that). The fill is merged with the stencil buffer, and the outline is
// drawn as thick screen-space strips outside every fill so only the union's edge shows.
import {
  BlendingState,
  RenderState,
  ShaderProgram,
  ShaderSource,
} from "cesium";
import {
  AQUA_GLSL,
  FILL_ALPHA,
  FILL_STENCIL,
  InstancedFieldsPrimitive,
  InstancedRenderer,
  MEAN_EARTH_RADIUS,
  OUTLINE_PX,
  OUTLINE_STENCIL,
} from "./instancing.js";

const FILL_LOCATIONS = { unit: 0, centerHigh: 1, centerLow: 2, shape: 3 };
const LINE_LOCATIONS = { unit: 0, centerHigh: 1, centerLow: 2, shape: 3, other: 4 };

// Per-instance inputs plus the function that places a unit-mesh vertex on the globe.
const COMMON = `
in vec3 centerHigh;  // field center in ECEF (meters, incl. height), split into high/low parts
in vec3 centerLow;
in vec4 shape;       // x: radius m, y: start bearing rad (clockwise from north), z: sweep rad, w: 1 if full circle

const float R = ${MEAN_EARTH_RADIUS.toFixed(1)};
const vec3 AQUA = ${AQUA_GLSL};
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
  const out = { fill: { attributes: [{ data: pos, size: 3, index: 0 }], indices: idx } };

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
    out.line = {
      attributes: [{ data: self, size: 4, index: 0 }, { data: other, size: 3, index: 4 }],
      indices: lineIdx,
    };
  }
  return out;
}

class FieldsPrimitive extends InstancedFieldsPrimitive {
  constructor() {
    super();
    this.fillShader = null;
    this.lineShader = null;
  }

  buildCommands(context, { style, segments, rings }) {
    const mesh = buildMesh(segments, rings, style);

    const program = (vs, attributeLocations) =>
      ShaderProgram.fromCache({
        context,
        vertexShaderSource: new ShaderSource({ sources: [vs] }),
        fragmentShaderSource: new ShaderSource({ sources: [FS] }),
        attributeLocations,
      });
    this.fillShader ??= program(FILL_VS, FILL_LOCATIONS);
    this.lineShader ??= program(LINE_VS, LINE_LOCATIONS);

    // Order matters: the fill must run first so the stencil is complete when the outline tests
    // it. Within one pass Cesium keeps push order for commands sharing a bounding volume.
    const outlineOnly = style === "outline";
    const visible = !outlineOnly;
    this.commands.push(
      this.makeCommand(context, {
        ...mesh.fill,
        shaderProgram: this.fillShader,
        renderState: RenderState.fromCache({
          depthTest: { enabled: true },
          depthMask: false,
          blending: BlendingState.ALPHA_BLEND,
          colorMask: { red: visible, green: visible, blue: visible, alpha: visible },
          ...FILL_STENCIL,
        }),
        uniformMap: { u_alpha: () => FILL_ALPHA },
      }),
    );
    if (mesh.line) {
      this.commands.push(
        this.makeCommand(context, {
          ...mesh.line,
          shaderProgram: this.lineShader,
          renderState: RenderState.fromCache({
            depthTest: { enabled: true },
            depthMask: false,
            blending: BlendingState.ALPHA_BLEND,
            ...OUTLINE_STENCIL,
          }),
          // Twice the visible width: only the half outside every fill survives the stencil test.
          uniformMap: { u_widthPx: () => OUTLINE_PX * 2 },
        }),
      );
    }
  }

  destroy() {
    super.destroy();
    this.fillShader?.destroy();
    this.lineShader?.destroy();
    this.fillShader = this.lineShader = null;
  }
}

export class ShaderRenderer extends InstancedRenderer {
  createPrimitive() {
    return new FieldsPrimitive();
  }

  meshOptions(params) {
    const style = params.primitiveStyle === "points" ? "fill+outline" : params.primitiveStyle;
    return { style, segments: params.segments, rings: params.rings };
  }

  boundingPadding(params) {
    return Math.abs(params.heightMeters) + 10;
  }

  meshCost(params) {
    const { segments, rings } = params;
    return {
      vertices: 1 + rings * (segments + 1),
      triangles: segments * (1 + 2 * (rings - 1)),
    };
  }
}
