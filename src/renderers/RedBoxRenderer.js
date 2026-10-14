// A minimal custom Cesium primitive: one box, drawn with our own shader.
//
// Written as a tutorial. It does not use Cesium.Entity or Cesium.Primitive, and the box is never
// given a Cesium position object. We hand the GPU raw numbers and let our own GLSL work out where
// they land on the globe.
//
// The whole pipeline, in the order you should read this file:
//
//   1. VERTEX DATA   A unit cube as plain typed arrays (corner positions + face normals).
//   2. SHADERS       GLSL vertex shader (places each corner on the globe) and fragment shader
//                    (colors each pixel).
//   3. UNIFORMS      Three knobs: box size in meters, location (lon/lat/height), color.
//   4. DRAW COMMAND  Bundles buffers + shader + render state into one thing Cesium can draw.
//   5. SCENE HOOK    Cesium calls update(frameState) on every object in scene.primitives each
//                    frame; we push our command into frameState.commandList.
//
// Coordinate frames used below:
//   - ECEF: Cesium's world frame. Origin at Earth's center, units are meters. A point on the
//     surface near Abu Dhabi is about (3.6e6, 5.2e6, 2.6e6) meters.
//   - ENU:  "East, North, Up" at the box's location. We model the box in this local frame
//     (x = east, y = north, z = up) and rotate it into ECEF in the vertex shader.
//   - Eye:  camera space. Cesium gives us built-in uniforms/functions (czm_*) to get there.
import {
  BoundingSphere,
  Buffer,
  BufferUsage,
  Cartesian3,
  Color,
  ComponentDatatype,
  CullFace,
  DrawCommand,
  IndexDatatype,
  Pass,
  PrimitiveType,
  RenderState,
  ShaderProgram,
  ShaderSource,
  VertexArray,
} from "cesium";
import { DEG, toEcef } from "../geo.js";

// ---------------------------------------------------------------------------------------------
// 1. VERTEX DATA
// ---------------------------------------------------------------------------------------------
// A cube needs 8 corners, but each face needs its own normal (the direction it faces, used for
// lighting), and a corner shared by three faces has three different normals. So we store 24
// vertices: 4 per face. Positions are in a unit cube: x (east) and y (north) run from -0.5 to
// 0.5, and z (up) from 0 to 1, so the box sits ON the location instead of being centered on it.
function buildCube() {
  const positions = [];
  const normals = [];
  const indices = [];
  // Each face: its outward normal n and two in-plane axes u, v with u x v = n, which makes the
  // vertices below counter-clockwise when viewed from outside (so back-face culling works).
  const faces = [
    { n: [1, 0, 0], u: [0, 1, 0], v: [0, 0, 1] },
    { n: [-1, 0, 0], u: [0, 0, 1], v: [0, 1, 0] },
    { n: [0, 1, 0], u: [0, 0, 1], v: [1, 0, 0] },
    { n: [0, -1, 0], u: [1, 0, 0], v: [0, 0, 1] },
    { n: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] },
    { n: [0, 0, -1], u: [0, 1, 0], v: [1, 0, 0] },
  ];
  for (const { n, u, v } of faces) {
    const base = positions.length / 3;
    for (const [su, sv] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      for (let k = 0; k < 3; k++) {
        const centered = 0.5 * n[k] + 0.5 * (su * u[k] + sv * v[k]); // in [-0.5, 0.5]
        positions.push(k === 2 ? centered + 0.5 : centered); // lift z so the base is at z = 0
        normals.push(n[k]);
      }
    }
    // Two triangles per face.
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: new Uint16Array(indices),
  };
}

// ---------------------------------------------------------------------------------------------
// 2. SHADERS
// ---------------------------------------------------------------------------------------------
// Cesium compiles these for you and prepends a library of built-ins named czm_*. Write them as
// GLSL ES 3.00 ("in"/"out", no "attribute"/"varying"); Cesium translates if needed.
//
// The vertex shader runs once per vertex (24 times) and must output gl_Position: where on the
// screen that vertex lands. The fragment shader then runs once per covered pixel.
const VERTEX_SHADER = `
// Per-vertex inputs, fed from the buffers in step 1.
in vec3 position;   // unit-cube corner (east, north, up)
in vec3 normal;     // outward face normal, same axes

// Uniforms: values that are the same for every vertex. These are our three knobs.
uniform float u_sizeMeters;  // edge length of the box
uniform vec3 u_centerHigh;   // box location in ECEF meters, split in two (see "Precision" below)
uniform vec3 u_centerLow;

out vec3 v_normalEye;       // handed to the fragment shader (interpolated across each face)

// Earth is slightly flattened. For a point on the surface, the direction "up" (perpendicular to
// the ground) is proportional to (x, y, z * a^2/b^2) with WGS84's a and b.
const float A2_OVER_B2 = 1.006739496742;

void main()
{
    // Build the local East-North-Up axes at the box's location, expressed in ECEF.
    vec3 center = u_centerHigh + u_centerLow;
    vec3 up = normalize(vec3(center.xy, center.z * A2_OVER_B2));
    vec3 east = normalize(vec3(-up.y, up.x, 0.0)); // perpendicular to up and to Earth's axis
    vec3 north = cross(up, east);

    // Scale the unit cube to meters and rotate it from ENU into ECEF.
    vec3 offset = (east * position.x + north * position.y + up * position.z) * u_sizeMeters;

    // Precision: ECEF coordinates are ~6,000,000 m, but a float32 has only ~7 significant digits,
    // so a single float can only place things to about half a meter, and a small box would
    // visibly jitter as the camera moves. Cesium's trick ("relative to eye"): we pass the center
    // as two floats (high + low) and czm_translateRelativeToEye subtracts the camera position
    // from both parts. The result is small, so float32 is plenty accurate.
    vec4 positionRelativeToEye = czm_translateRelativeToEye(u_centerHigh, u_centerLow);
    positionRelativeToEye.xyz += offset;

    // Project to the screen. czm_modelViewProjectionRelativeToEye is Cesium's built-in matrix for
    // positions that are already relative to the camera.
    gl_Position = czm_modelViewProjectionRelativeToEye * positionRelativeToEye;

    // For lighting, rotate the normal into eye space (czm_viewRotation: ECEF -> eye).
    v_normalEye = czm_viewRotation * (east * normal.x + north * normal.y + up * normal.z);
}
`;

const FRAGMENT_SHADER = `
in vec3 v_normalEye;
uniform vec4 u_color;  // the box color (rgba)

void main()
{
    // Cheap fixed "headlight" shading so the box reads as 3D: faces turned toward a light
    // direction (in eye space, up and to the right of the camera) are brighter.
    vec3 n = normalize(v_normalEye);
    float light = 0.5 + 0.5 * max(dot(n, normalize(vec3(0.4, 0.5, 0.75))), 0.0);

    // out_FragColor is this pixel's final color. czm_gammaCorrect keeps colors consistent with
    // the rest of Cesium's rendering.
    out_FragColor = czm_gammaCorrect(vec4(u_color.rgb * light, u_color.a));
}
`;

// Attribute locations: the number linking a buffer to an "in" variable in the vertex shader.
const ATTRIBUTE_LOCATIONS = { position: 0, normal: 1 };

// ---------------------------------------------------------------------------------------------
// Precision helper: split a double into two float32-exact halves (see the vertex shader).
// ---------------------------------------------------------------------------------------------
function splitDouble(value) {
  const high = Math.floor(value / 65536) * 65536;
  return [high, value - high];
}

// ---------------------------------------------------------------------------------------------
// 3, 4, 5. UNIFORMS, DRAW COMMAND AND SCENE HOOK
// ---------------------------------------------------------------------------------------------
// Anything added to scene.primitives just needs: update(frameState), isDestroyed(), destroy().
class BoxPrimitive {
  constructor() {
    this.show = true;

    // The three configurable parameters. Change them any time; they are read every frame.
    this.sizeMeters = 1000;
    this.location = { lon: 54.3773, lat: 24.4539, height: 0 }; // degrees, degrees, meters
    this.color = Color.RED;

    // Values the shader reads. These objects are what Cesium uploads as uniforms; we keep them
    // as long-lived fields so no objects are created per frame. (Cartesian3 is used here purely
    // as a container for three numbers; it is not a position handed to Cesium.)
    this.centerHigh = new Cartesian3();
    this.centerLow = new Cartesian3();
    this.boundingSphere = new BoundingSphere();

    this.command = undefined;
    this.resources = [];
  }

  /** Recomputes the uniforms that depend on location and size. Cheap; called every frame. */
  syncUniforms() {
    const { lon, lat, height } = this.location;
    const ecef = new Float64Array(3);
    toEcef(lat * DEG, lon * DEG, height, ecef, 0); // plain math: degrees -> meters from Earth's center
    const [hx, lx] = splitDouble(ecef[0]);
    const [hy, ly] = splitDouble(ecef[1]);
    const [hz, lz] = splitDouble(ecef[2]);
    Cartesian3.fromElements(hx, hy, hz, this.centerHigh);
    Cartesian3.fromElements(lx, ly, lz, this.centerLow);

    // Cesium skips drawing a command if its bounding sphere is off screen. It needs the sphere
    // in ECEF; the box's farthest point is its diagonal, about 1.8x the edge length from the
    // base center.
    Cartesian3.fromElements(ecef[0], ecef[1], ecef[2], this.boundingSphere.center);
    this.boundingSphere.radius = this.sizeMeters * 1.8;
  }

  createCommand(context) {
    const cube = buildCube();

    // Buffers: GPU memory holding the vertex data. STATIC_DRAW = uploaded once, never changed.
    const positionBuffer = Buffer.createVertexBuffer({ context, typedArray: cube.positions, usage: BufferUsage.STATIC_DRAW });
    const normalBuffer = Buffer.createVertexBuffer({ context, typedArray: cube.normals, usage: BufferUsage.STATIC_DRAW });
    const indexBuffer = Buffer.createIndexBuffer({
      context,
      typedArray: cube.indices,
      usage: BufferUsage.STATIC_DRAW,
      indexDatatype: IndexDatatype.UNSIGNED_SHORT,
    });

    // VertexArray: says which buffer feeds which shader input, and how to read it.
    const vertexArray = new VertexArray({
      context,
      indexBuffer,
      attributes: [
        { index: ATTRIBUTE_LOCATIONS.position, vertexBuffer: positionBuffer, componentsPerAttribute: 3, componentDatatype: ComponentDatatype.FLOAT },
        { index: ATTRIBUTE_LOCATIONS.normal, vertexBuffer: normalBuffer, componentsPerAttribute: 3, componentDatatype: ComponentDatatype.FLOAT },
      ],
    });

    // ShaderProgram: compiles + links the two shaders above.
    const shaderProgram = ShaderProgram.fromCache({
      context,
      vertexShaderSource: new ShaderSource({ sources: [VERTEX_SHADER] }),
      fragmentShaderSource: new ShaderSource({ sources: [FRAGMENT_SHADER] }),
      attributeLocations: ATTRIBUTE_LOCATIONS,
    });

    // RenderState: the fixed-function GPU settings for this draw.
    const renderState = RenderState.fromCache({
      depthTest: { enabled: true }, // hide the box behind terrain and nearer objects
      depthMask: true, // write the box's depth so things behind it are hidden too
      cull: { enabled: true, face: CullFace.BACK }, // skip faces pointing away from the camera
    });

    this.resources.push(positionBuffer, normalBuffer, indexBuffer, vertexArray, shaderProgram);

    // DrawCommand: one "draw this" instruction. The uniformMap maps shader uniform names to
    // functions that return their current value; Cesium calls them right before drawing.
    this.command = new DrawCommand({
      primitiveType: PrimitiveType.TRIANGLES,
      vertexArray,
      shaderProgram,
      renderState,
      uniformMap: {
        u_sizeMeters: () => this.sizeMeters,
        u_centerHigh: () => this.centerHigh,
        u_centerLow: () => this.centerLow,
        u_color: () => this.color,
      },
      pass: Pass.OPAQUE, // Cesium draws opaque things first, then translucent ones
      owner: this,
      modelMatrix: undefined, // identity: our vertices are already in world (ECEF) terms
    });
  }

  // Called by Cesium once per frame (and again for picking/shadow passes, hence the check).
  update(frameState) {
    if (!this.show) return;
    if (!this.command) this.createCommand(frameState.context);
    this.syncUniforms();
    this.command.boundingVolume = this.boundingSphere;
    if (frameState.passes.render) frameState.commandList.push(this.command);
  }

  isDestroyed() {
    return false;
  }

  destroy() {
    for (const r of this.resources) if (!r.isDestroyed()) r.destroy();
    this.resources = [];
    this.command = undefined;
    return undefined;
  }
}

/** Wires the box into the playground: owns the primitive and pushes the UI parameters into it. */
export class RedBoxRenderer {
  constructor(viewer) {
    this.viewer = viewer;
    this.primitive = null;
  }

  update(params, location) {
    this.primitive ??= this.viewer.scene.primitives.add(new BoxPrimitive());
    this.primitive.sizeMeters = params.boxSizeMeters;
    this.primitive.location = location;
    this.primitive.color = Color.fromCssColorString(params.boxColor);
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
