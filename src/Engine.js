// Owns the field data, the update loop, both renderers and the perf counters.
// Lives outside React so re-renders never touch Cesium.
import { generateFields, jitterFields } from "./fields.js";
import { REGION } from "./config.js";
import { PrimitiveRenderer } from "./renderers/PrimitiveRenderer.js";
import { ImageryRenderer } from "./renderers/ImageryRenderer.js";
import { ShaderRenderer } from "./renderers/ShaderRenderer.js";
import { GroundRenderer } from "./renderers/GroundRenderer.js";

const newRendererStats = () => ({
  updates: 0,
  droppedTicks: 0,
  lastUpdateLatencyMs: 0,
  // A
  workerComputeMs: 0,
  workerWallMs: 0,
  mainBuildMs: 0,
  firstFrameMs: 0,
  uploadMs: 0,
  vertices: 0,
  triangles: 0,
  transferBytes: 0,
  // B
  tilesRequested: 0,
  tilesDone: 0,
  tilesEmpty: 0,
  tileWorkerMsTotal: 0,
  tileWallMsTotal: 0,
  tilesInLastVersion: 0,
  activeLayers: 0,
});

export class Engine {
  constructor(viewer, params) {
    this.viewer = viewer;
    this.params = params;
    this.stats = {
      primitive: newRendererStats(),
      imagery: newRendererStats(),
      shader: newRendererStats(),
      ground: newRendererStats(),
      frame: { fps: 0, avgFrameMs: 0, maxFrameMs: 0, avgRenderCpuMs: 0, maxRenderCpuMs: 0 },
    };
    this.fields = generateFields(params.fieldCount, REGION);
    this.primitive = new PrimitiveRenderer(viewer, this.stats.primitive);
    this.imagery = new ImageryRenderer(viewer, this.stats.imagery);
    this.shader = new ShaderRenderer(viewer, this.stats.shader);
    this.ground = new GroundRenderer(viewer, this.stats.ground);
    this.primitive.onApplied = () => (this.awaitFirstFrame = true);
    this.installFrameMonitor();
    this.applyMode();
    this.tick();
    this.restartTimer();
  }

  installFrameMonitor() {
    const scene = this.viewer.scene;
    let renderStart = 0;
    let last = performance.now();
    let win = { frames: 0, sum: 0, max: 0, cpuSum: 0, cpuMax: 0, start: last };
    this.awaitFirstFrame = false;

    this.removePreUpdate = scene.preUpdate.addEventListener(() => {
      renderStart = performance.now();
    });
    this.removePostRender = scene.postRender.addEventListener(() => {
      const now = performance.now();
      const dt = now - last;
      const cpu = now - renderStart;
      last = now;
      win.frames++;
      win.sum += dt;
      win.max = Math.max(win.max, dt);
      win.cpuSum += cpu;
      win.cpuMax = Math.max(win.cpuMax, cpu);
      // The synchronous Primitive batch (combine + GPU upload) happens inside the first
      // render after apply(), so that frame's CPU time is the real main-thread cost of A.
      this.shader.onPostRender();
      this.ground.onPostRender();
      if (this.awaitFirstFrame) {
        this.stats.primitive.firstFrameMs = cpu;
        this.awaitFirstFrame = false;
      }
      if (now - win.start >= 1000) {
        const f = this.stats.frame;
        f.fps = (win.frames * 1000) / (now - win.start);
        f.avgFrameMs = win.sum / win.frames;
        f.maxFrameMs = win.max;
        f.avgRenderCpuMs = win.cpuSum / win.frames;
        f.maxRenderCpuMs = win.cpuMax;
        win = { frames: 0, sum: 0, max: 0, cpuSum: 0, cpuMax: 0, start: now };
      }
    });
  }

  get usePrimitive() {
    return this.params.mode === "primitive" || this.params.mode === "both";
  }

  get useShader() {
    return this.params.mode === "shader";
  }

  get useGround() {
    return this.params.mode === "ground";
  }

  get useImagery() {
    return this.params.mode === "imagery" || this.params.mode === "both";
  }

  setParams(next) {
    const prev = this.params;
    this.params = next;
    if (next.fieldCount !== prev.fieldCount) this.fields = generateFields(next.fieldCount, REGION);
    if (next.updateIntervalMs !== prev.updateIntervalMs) this.restartTimer();
    if (next.mode !== prev.mode) this.applyMode();
    const geometryChanged = ["fieldCount", "segments", "rings", "primitiveStyle", "heightMeters", "groundRange", "tileSize", "maximumLevel", "mode"]
      .some((k) => next[k] !== prev[k]);
    if (geometryChanged) this.render();
  }

  applyMode() {
    if (!this.usePrimitive) this.primitive.clear();
    if (!this.useImagery) this.imagery.clear();
    if (!this.useShader) this.shader.clear();
    if (!this.useGround) this.ground.clear();
  }

  restartTimer() {
    clearInterval(this.timer);
    if (this.params.updateIntervalMs > 0) this.timer = setInterval(() => this.tick(), this.params.updateIntervalMs);
  }

  tick() {
    this.fields = jitterFields(this.fields, REGION, this.params.jitterScale);
    this.render();
  }

  render() {
    if (this.usePrimitive) this.primitive.update(this.fields, this.params);
    if (this.useImagery) this.imagery.update(this.fields, this.params);
    if (this.useShader) this.shader.update(this.fields, this.params);
    if (this.useGround) this.ground.update(this.fields, this.params);
  }

  resetCounters() {
    this.stats.primitive = Object.assign(this.stats.primitive, newRendererStats());
    this.stats.imagery = Object.assign(this.stats.imagery, newRendererStats());
    this.stats.shader = Object.assign(this.stats.shader, newRendererStats());
    this.stats.ground = Object.assign(this.stats.ground, newRendererStats());
  }

  destroy() {
    clearInterval(this.timer);
    this.removePreUpdate();
    this.removePostRender();
    this.primitive.destroy();
    this.imagery.destroy();
    this.shader.destroy();
    this.ground.destroy();
  }
}
