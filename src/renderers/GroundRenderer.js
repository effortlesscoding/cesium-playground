// Approach D: ground-clamped fields as depth-projected decals.
//
// Each field is one instance of a unit box that encloses its circle. Nothing is placed on the
// globe geometrically. Instead the fragment shader reads the scene's depth texture (globe +
// terrain + 3D tiles, the same texture Cesium's own ground primitives use), rebuilds the world
// position of whatever surface is behind each box pixel, drops it onto the field's tangent
// plane and tests whether it lies inside the circle/pie. So the shapes follow terrain and
// buildings exactly, and edges are analytic (anti-aliased distance fields) at any zoom.
//
// Limits: only surfaces in the depth texture are painted (globe, terrain, 3D tiles; not
// ordinary opaque primitives), and anything within +/- groundRange meters of the field's center
// height is projected onto, including walls. Merging and the outline-of-the-union work exactly
// as in ShaderRenderer.js: fills mark the stencil once, the outline is drawn outside every fill.
import {
  BlendingState,
  CullFace,
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

const LOCATIONS = { corner: 0, centerHigh: 1, centerLow: 2, shape: 3 };

// The box is padded this many pixels beyond the circle so the anti-aliased edge and outline fit.
const MARGIN_PX = OUTLINE_PX + 2;

const VS = `
in vec3 corner;      // box corner, each component -1 or +1 (east, north, up)
in vec3 centerHigh;
in vec3 centerLow;
in vec4 shape;       // x: radius m, y: start bearing rad, z: sweep rad, w: 1 if full circle
uniform float u_vRange;
uniform float u_marginPx;
out vec4 v_shape;
out vec3 v_centerEye;
out vec3 v_eastEye;
out vec3 v_upEye;

const float R = ${MEAN_EARTH_RADIUS.toFixed(1)};
const float A2_OVER_B2 = 1.006739496742;

void main()
{
    vec3 c = centerHigh + centerLow;
    vec3 up = normalize(vec3(c.xy, c.z * A2_OVER_B2));
    vec3 east = normalize(vec3(-up.y, up.x, 0.0));
    vec3 north = cross(up, east);

    vec4 pc = czm_translateRelativeToEye(centerHigh, centerLow);
    vec4 centerEye = czm_modelViewRelativeToEye * pc;
    float ext = shape.x + u_marginPx * czm_metersPerPixel(centerEye);
    // The ground sags below the tangent plane by about d^2 / 2R; make room for it at the corners.
    float vExt = u_vRange + ext * ext / R;

    vec3 offset = east * (corner.x * ext) + north * (corner.y * ext) + up * (corner.z * vExt);
    gl_Position = czm_modelViewProjectionRelativeToEye * vec4(pc.xyz + offset, 1.0);

    v_shape = shape;
    v_centerEye = centerEye.xyz;
    v_eastEye = czm_viewRotation * east;
    v_upEye = czm_viewRotation * up;
}
`;

const FS = `
in vec4 v_shape;
in vec3 v_centerEye;
in vec3 v_eastEye;
in vec3 v_upEye;
uniform float u_alpha;
uniform float u_vRange;
uniform float u_outlinePx;

const float R = ${MEAN_EARTH_RADIUS.toFixed(1)};
const float PI = 3.14159265359;
const float TWO_PI = 6.28318530718;
const vec3 AQUA = ${AQUA_GLSL};

// Signed distance (meters, negative inside) to a pie of radius r whose axis is +y and whose
// half-aperture is given as c = (sin, cos). Inigo Quilez's sdPie.
float sdPie(vec2 p, vec2 c, float r)
{
    p.x = abs(p.x);
    float l = length(p) - r;
    float m = length(p - c * clamp(dot(p, c), 0.0, r));
    return max(l, m * sign(c.y * p.x - c.x * p.y));
}

void main()
{
    vec2 uv = gl_FragCoord.xy / czm_viewport.zw;
    float depth = czm_unpackDepth(texture(czm_globeDepthTexture, uv));
    if (depth >= 1.0) discard; // nothing in the depth texture here (sky)

    // World position of the surface behind this pixel, relative to the field's center (eye frame).
    vec4 e = czm_windowToEyeCoordinates(gl_FragCoord.xy, depth);
    vec3 pe = e.xyz / e.w;
    vec3 d = pe - v_centerEye;
    vec3 up = normalize(v_upEye);
    vec3 east = normalize(v_eastEye);
    vec3 north = cross(up, east);

    vec2 p = vec2(dot(d, east), dot(d, north));
    float r2 = dot(p, p);
    // Height above the sphere through the center (the tangent plane drops away by r^2 / 2R).
    float h = dot(d, up) + r2 / (2.0 * R);
    if (abs(h) > u_vRange) discard;

    float radius = v_shape.x;
    float len = sqrt(r2);
    float sdf;
    if (v_shape.w > 0.5) {
        sdf = len - radius;
    } else if (len < 1e-3) {
        sdf = -radius;
    } else {
        float halfSweep = 0.5 * v_shape.z;
        // Bearing (clockwise from north) relative to the pie's axis, wrapped to [-pi, pi).
        float a = atan(p.x, p.y) - (v_shape.y + halfSweep);
        a -= TWO_PI * floor((a + PI) / TWO_PI);
        sdf = sdPie(vec2(len * sin(a), len * cos(a)), vec2(sin(halfSweep), cos(halfSweep)), radius);
    }

    float px = sdf / czm_metersPerPixel(vec4(pe, 1.0)); // signed distance in pixels
#ifdef OUTLINE
    // A band just outside the fill (which owns everything up to px = 0), anti-aliased on its
    // outer edge. The stencil test removes the parts that lie inside any other field.
    float alpha = clamp(px + 0.5, 0.0, 1.0) * clamp(u_outlinePx + 0.5 - px, 0.0, 1.0);
    if (px <= 0.0 || alpha <= 0.0) discard;
    out_FragColor = czm_gammaCorrect(vec4(AQUA, alpha));
#else
    // Hard edge: a partly transparent edge pixel would still mark the stencil, which would then
    // block another field's full fill there and leave a faint seam inside merged areas.
    if (px > 0.0) discard; // discarded pixels do not mark the stencil
    out_FragColor = czm_gammaCorrect(vec4(AQUA, u_alpha));
#endif
}
`;

/** Unit box with outward-facing triangles, so culling the front faces leaves the back faces. */
function buildBox() {
  const corner = [];
  for (let i = 0; i < 8; i++) corner.push(i & 1 ? 1 : -1, i & 2 ? 1 : -1, i & 4 ? 1 : -1);
  const at = (i) => corner.slice(i * 3, i * 3 + 3);
  const idx = [];
  // For each face: the two in-plane axes and the face normal axis/sign.
  for (const axis of [0, 1, 2]) {
    for (const sign of [-1, 1]) {
      const face = [];
      for (let i = 0; i < 8; i++) if (at(i)[axis] === sign) face.push(i);
      // Order the four vertices around the face, then wind them counter-clockwise from outside.
      const [a, b, c, d] = [face[0], face[1], face[3], face[2]];
      const u = at(b).map((v, k) => v - at(a)[k]);
      const w = at(c).map((v, k) => v - at(a)[k]);
      const normal = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
      const outward = normal[axis] * sign > 0;
      idx.push(...(outward ? [a, b, c, a, c, d] : [a, c, b, a, d, c]));
    }
  }
  return { attributes: [{ data: new Float32Array(corner), size: 3, index: 0 }], indices: idx };
}

class GroundPrimitive extends InstancedFieldsPrimitive {
  constructor() {
    super();
    this.fillShader = null;
    this.lineShader = null;
  }

  buildCommands(context, { style, groundRange }) {
    const program = (defines) =>
      ShaderProgram.fromCache({
        context,
        vertexShaderSource: new ShaderSource({ sources: [VS] }),
        fragmentShaderSource: new ShaderSource({ sources: [FS], defines }),
        attributeLocations: LOCATIONS,
      });
    this.fillShader ??= program([]);
    this.lineShader ??= program(["OUTLINE"]);

    const box = buildBox();
    const uniformMap = {
      u_alpha: () => FILL_ALPHA,
      u_vRange: () => groundRange,
      u_marginPx: () => MARGIN_PX,
      u_outlinePx: () => OUTLINE_PX,
    };
    // Back faces only, no depth test: the box is just a screen-space cover for the field; the
    // real surface depth is read from the depth texture in the fragment shader.
    const state = (extra) =>
      RenderState.fromCache({
        depthTest: { enabled: false },
        depthMask: false,
        cull: { enabled: true, face: CullFace.FRONT },
        blending: BlendingState.ALPHA_BLEND,
        ...extra,
      });

    // Fill first (it marks the stencil), outline second (it reads it).
    const visible = style !== "outline";
    this.commands.push(
      this.makeCommand(context, {
        ...box,
        shaderProgram: this.fillShader,
        renderState: state({
          colorMask: { red: visible, green: visible, blue: visible, alpha: visible },
          ...FILL_STENCIL,
        }),
        uniformMap,
      }),
    );
    if (style !== "fill") {
      this.commands.push(
        this.makeCommand(context, {
          ...box,
          shaderProgram: this.lineShader,
          renderState: state(OUTLINE_STENCIL),
          uniformMap,
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

export class GroundRenderer extends InstancedRenderer {
  createPrimitive() {
    return new GroundPrimitive();
  }

  meshOptions(params) {
    const style = params.primitiveStyle === "points" ? "fill+outline" : params.primitiveStyle;
    return { style, groundRange: params.groundRange };
  }

  boundingPadding(params) {
    return params.groundRange + Math.abs(params.heightMeters) + 50;
  }

  meshCost() {
    return { vertices: 8, triangles: 12 };
  }
}
