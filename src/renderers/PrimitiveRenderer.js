// Approach A: worker-generated Cartesian3 vertices -> batched Cesium Primitives.
// One GeometryInstance per field, all batched into a single fill Primitive and a single
// outline Primitive (or a PointPrimitiveCollection in "points" style).
import {
  BoundingSphere,
  Cartesian3,
  Color,
  ColorGeometryInstanceAttribute,
  ComponentDatatype,
  Geometry,
  GeometryAttribute,
  GeometryInstance,
  PerInstanceColorAppearance,
  PointPrimitiveCollection,
  Primitive,
  PrimitiveType,
} from "cesium";
import { fieldRgb } from "../geo.js";

const META = 6;

function makeWorker() {
  return new Worker(new URL("../workers/geometry.worker.js", import.meta.url), { type: "module" });
}

export class PrimitiveRenderer {
  constructor(viewer, stats) {
    this.viewer = viewer;
    this.stats = stats;
    this.workers = [];
    this.fillPrimitive = null;
    this.linePrimitive = null;
    this.points = null;
    this.busy = false;
    this.jobId = 0;
    this.colorCache = new Map();
  }

  ensureWorkers(count) {
    while (this.workers.length > count) this.workers.pop().terminate();
    while (this.workers.length < count) this.workers.push(makeWorker());
  }

  color(id, alpha) {
    const key = id * 2 + (alpha < 1 ? 1 : 0);
    let c = this.colorCache.get(key);
    if (!c) {
      const [r, g, b] = fieldRgb(id);
      c = ColorGeometryInstanceAttribute.fromColor(new Color(r, g, b, alpha));
      this.colorCache.set(key, c);
    }
    return c;
  }

  /** Returns false if the previous update is still in flight (tick dropped). */
  update(fields, params) {
    if (this.busy) {
      this.stats.droppedTicks++;
      return false;
    }
    this.busy = true;
    this.ensureWorkers(params.geometryWorkers);
    const jobId = ++this.jobId;
    const t0 = performance.now();

    // Split fields into one contiguous chunk per worker.
    const n = this.workers.length;
    const chunk = Math.ceil(fields.length / n);
    const jobs = [];
    for (let w = 0; w < n; w++) {
      const slice = fields.slice(w * chunk, (w + 1) * chunk);
      if (slice.length === 0) continue;
      const worker = this.workers[w];
      jobs.push(
        new Promise((resolve) => {
          worker.onmessage = (e) => resolve(e.data);
          worker.postMessage({
            jobId,
            fields: slice,
            segments: params.segments,
            rings: params.rings,
            height: params.heightMeters,
          });
        }),
      );
    }

    Promise.all(jobs).then((results) => {
      this.busy = false;
      if (this.destroyed || jobId !== this.jobId) return;
      const workerWallMs = performance.now() - t0;
      const b0 = performance.now();
      this.apply(results, params.primitiveStyle);
      const buildMs = performance.now() - b0;
      this.onApplied?.();

      let vertices = 0, triangles = 0, bytes = 0;
      for (const r of results) {
        vertices += r.positions.length / 3;
        triangles += r.fill.length / 3;
        bytes += r.positions.byteLength + r.fill.byteLength + r.lines.byteLength;
      }
      const s = this.stats;
      s.updates++;
      s.workerComputeMs = Math.max(...results.map((r) => r.computeMs));
      s.workerWallMs = workerWallMs;
      s.mainBuildMs = buildMs;
      s.vertices = vertices;
      s.triangles = params.primitiveStyle === "outline" || params.primitiveStyle === "points" ? 0 : triangles;
      s.transferBytes = bytes;
      s.lastUpdateLatencyMs = performance.now() - t0;
    });
    return true;
  }

  apply(results, style) {
    const scene = this.viewer.scene;
    const wantFill = style === "fill+outline" || style === "fill";
    const wantLines = style === "fill+outline" || style === "outline";
    const wantPoints = style === "points";

    const fillInstances = [];
    const lineInstances = [];
    for (const r of results) {
      for (let i = 0; i < r.ids.length; i++) {
        const id = r.ids[i];
        const m = i * META;
        const vOff = r.meta[m], vCount = r.meta[m + 1];
        const position = new GeometryAttribute({
          componentDatatype: ComponentDatatype.DOUBLE,
          componentsPerAttribute: 3,
          values: r.positions.subarray(vOff * 3, (vOff + vCount) * 3),
        });
        const sphere = new BoundingSphere(
          new Cartesian3(r.spheres[i * 4], r.spheres[i * 4 + 1], r.spheres[i * 4 + 2]),
          r.spheres[i * 4 + 3],
        );
        if (wantFill) {
          fillInstances.push(
            new GeometryInstance({
              id,
              geometry: new Geometry({
                attributes: { position },
                indices: r.fill.subarray(r.meta[m + 2], r.meta[m + 2] + r.meta[m + 3]),
                primitiveType: PrimitiveType.TRIANGLES,
                boundingSphere: sphere,
              }),
              attributes: { color: this.color(id, 0.45) },
            }),
          );
        }
        if (wantLines) {
          lineInstances.push(
            new GeometryInstance({
              id,
              geometry: new Geometry({
                attributes: { position },
                indices: r.lines.subarray(r.meta[m + 4], r.meta[m + 4] + r.meta[m + 5]),
                primitiveType: PrimitiveType.LINES,
                boundingSphere: sphere,
              }),
              attributes: { color: this.color(id, 1) },
            }),
          );
        }
      }
    }

    // Raw Geometry (not a geometry description) must be built synchronously; Cesium batches
    // all instances into one draw call per primitive during the next scene.render().
    const primitiveOptions = {
      asynchronous: false,
      allowPicking: false,
      compressVertices: false,
      vertexCacheOptimize: false,
      interleave: false,
      releaseGeometryInstances: true,
    };

    const oldFill = this.fillPrimitive;
    const oldLines = this.linePrimitive;
    this.fillPrimitive = fillInstances.length
      ? scene.primitives.add(
          new Primitive({
            ...primitiveOptions,
            geometryInstances: fillInstances,
            appearance: new PerInstanceColorAppearance({ flat: true, translucent: true, closed: false }),
          }),
        )
      : null;
    this.linePrimitive = lineInstances.length
      ? scene.primitives.add(
          new Primitive({
            ...primitiveOptions,
            geometryInstances: lineInstances,
            appearance: new PerInstanceColorAppearance({ flat: true, translucent: false }),
          }),
        )
      : null;
    if (oldFill) scene.primitives.remove(oldFill);
    if (oldLines) scene.primitives.remove(oldLines);

    if (wantPoints) this.updatePoints(results);
    else if (this.points) {
      scene.primitives.remove(this.points);
      this.points = null;
    }
  }

  updatePoints(results) {
    const scene = this.viewer.scene;
    let total = 0;
    for (const r of results) total += r.positions.length / 3;
    if (this.points && this.points.length !== total) {
      scene.primitives.remove(this.points);
      this.points = null;
    }
    const fresh = !this.points;
    if (fresh) this.points = scene.primitives.add(new PointPrimitiveCollection());
    const pc = this.points;
    const scratch = new Cartesian3();
    let p = 0;
    for (const r of results) {
      for (let i = 0; i < r.ids.length; i++) {
        const [cr, cg, cb] = fieldRgb(r.ids[i]);
        const color = new Color(cr, cg, cb, 1);
        const vOff = r.meta[i * META], vCount = r.meta[i * META + 1];
        for (let v = vOff; v < vOff + vCount; v++) {
          scratch.x = r.positions[v * 3];
          scratch.y = r.positions[v * 3 + 1];
          scratch.z = r.positions[v * 3 + 2];
          if (fresh) pc.add({ position: scratch, color, pixelSize: 3 });
          else {
            const pt = pc.get(p);
            pt.position = scratch;
            pt.color = color;
          }
          p++;
        }
      }
    }
  }

  clear() {
    this.jobId++; // an update still in flight must not re-add primitives after we clear
    const prims = this.viewer.scene.primitives;
    for (const k of ["fillPrimitive", "linePrimitive", "points"]) {
      if (this[k]) prims.remove(this[k]);
      this[k] = null;
    }
  }

  destroy() {
    this.destroyed = true;
    this.clear();
    this.workers.forEach((w) => w.terminate());
    this.workers = [];
  }
}
