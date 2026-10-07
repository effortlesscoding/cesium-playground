// Approach B: fields rasterized into imagery tiles by a pool of OffscreenCanvas workers.
// Each update creates a new provider "version" in a hidden layer; once the globe has loaded
// its tiles, it is shown and the previous layer is removed (double buffering, no flicker).
import {
  Credit,
  DefaultProxy,
  Event,
  GeographicTilingScheme,
  Math as CesiumMath,
  Rectangle,
} from "cesium";

function makeWorker() {
  return new Worker(new URL("../workers/tile.worker.js", import.meta.url), { type: "module" });
}

class TilePool {
  constructor(stats) {
    this.stats = stats;
    this.workers = [];
    this.pending = new Map(); // id -> resolve
    this.nextId = 0;
    this.rr = 0;
  }

  resize(count) {
    while (this.workers.length > count) this.workers.pop().terminate();
    while (this.workers.length < count) {
      const w = makeWorker();
      w.onmessage = (e) => this.onMessage(e.data);
      this.workers.push(w);
    }
  }

  onMessage(msg) {
    if (msg.type !== "tile") return;
    const resolve = this.pending.get(msg.id);
    this.pending.delete(msg.id);
    if (resolve) resolve(msg);
  }

  broadcastFields(version, fields, segments) {
    for (const w of this.workers) w.postMessage({ type: "fields", version, fields, segments });
  }

  renderTile(req) {
    const id = this.nextId++;
    const worker = this.workers[this.rr++ % this.workers.length];
    return new Promise((resolve) => {
      this.pending.set(id, resolve);
      worker.postMessage({ type: "tile", id, ...req });
    });
  }

  destroy() {
    this.workers.forEach((w) => w.terminate());
    this.workers = [];
    this.pending.clear();
  }
}

/** Minimal custom ImageryProvider backed by the tile worker pool. */
class FieldsImageryProvider {
  constructor({ pool, version, tileSize, maximumLevel, rectangle, stats, emptyTile }) {
    this._pool = pool;
    this._version = version;
    this._stats = stats;
    this._emptyTile = emptyTile;
    this.tilingScheme = new GeographicTilingScheme();
    this.tileWidth = tileSize;
    this.tileHeight = tileSize;
    this.minimumLevel = 0;
    this.maximumLevel = maximumLevel;
    this.rectangle = rectangle;
    this.tileDiscardPolicy = undefined;
    this.errorEvent = new Event();
    this.credit = new Credit("Fields (worker raster)");
    this.proxy = new DefaultProxy("");
    this.hasAlphaChannel = true;
    this.outstanding = 0;
    this.requested = 0;
  }

  getTileCredits() {
    return undefined;
  }

  pickFeatures() {
    return undefined;
  }

  requestImage(x, y, level) {
    const r = this.tilingScheme.tileXYToRectangle(x, y, level);
    const t0 = performance.now();
    this.outstanding++;
    this.requested++;
    const stats = this._stats;
    stats.tilesRequested++;
    return this._pool
      .renderTile({
        version: this._version,
        west: CesiumMath.toDegrees(r.west),
        south: CesiumMath.toDegrees(r.south),
        east: CesiumMath.toDegrees(r.east),
        north: CesiumMath.toDegrees(r.north),
        size: this.tileWidth,
      })
      .then((msg) => {
        this.outstanding--;
        stats.tileWorkerMsTotal += msg.ms;
        stats.tileWallMsTotal += performance.now() - t0;
        stats.tilesDone++;
        if (msg.empty) {
          stats.tilesEmpty++;
          return this._emptyTile;
        }
        return msg.bitmap;
      });
  }
}

/** Bounding rectangle of all fields (degrees -> radians), padded by the largest radius. */
function fieldsRectangle(fields) {
  let w = 180, e = -180, s = 90, n = -90, maxR = 0;
  for (const f of fields) {
    w = Math.min(w, f.lon);
    e = Math.max(e, f.lon);
    s = Math.min(s, f.lat);
    n = Math.max(n, f.lat);
    maxR = Math.max(maxR, f.radius);
  }
  const padLat = (maxR / 111_000) * 1.1;
  const padLon = padLat / Math.max(0.1, Math.cos(CesiumMath.toRadians(Math.max(Math.abs(s), Math.abs(n)))));
  return Rectangle.fromDegrees(
    Math.max(-180, w - padLon),
    Math.max(-90, s - padLat),
    Math.min(180, e + padLon),
    Math.min(90, n + padLat),
  );
}

export class ImageryRenderer {
  constructor(viewer, stats) {
    this.viewer = viewer;
    this.stats = stats;
    this.pool = new TilePool(stats);
    this.version = 0;
    this.activeLayer = null;
    this.pendingLayer = null;
    this.pendingStart = 0;
    this.emptyTiles = new Map();
    this.removePostRender = viewer.scene.postRender.addEventListener(() => this.checkSwap());
  }

  emptyTile(size) {
    let c = this.emptyTiles.get(size);
    if (!c) {
      c = document.createElement("canvas");
      c.width = c.height = size;
      this.emptyTiles.set(size, c);
    }
    return c;
  }

  update(fields, params) {
    if (this.pendingLayer) {
      // Previous version still loading: drop this tick rather than piling up layers.
      this.stats.droppedTicks++;
      return false;
    }
    this.pool.resize(params.tileWorkers);
    const version = ++this.version;
    this.pool.broadcastFields(version, fields, params.segments);

    const provider = new FieldsImageryProvider({
      pool: this.pool,
      version,
      tileSize: params.tileSize,
      maximumLevel: params.maximumLevel,
      rectangle: fieldsRectangle(fields),
      stats: this.stats,
      emptyTile: this.emptyTile(params.tileSize),
    });
    const layers = this.viewer.imageryLayers;
    const layer = layers.addImageryProvider(provider);
    // Hidden via alpha (not `show`) so the globe still loads its tiles.
    layer.alpha = 0;
    this.pendingLayer = layer;
    this.pendingStart = performance.now();
    this.stats.updates++;
    return true;
  }

  checkSwap() {
    const layer = this.pendingLayer;
    if (!layer) return;
    const provider = layer.imageryProvider;
    const elapsed = performance.now() - this.pendingStart;
    const loaded = this.viewer.scene.globe.tilesLoaded && provider.outstanding === 0;
    // Fallback: if the camera keeps moving tilesLoaded may never settle.
    if (!loaded && elapsed < 5000) return;

    layer.alpha = 1;
    const layers = this.viewer.imageryLayers;
    if (this.activeLayer) layers.remove(this.activeLayer, true);
    this.activeLayer = layer;
    this.pendingLayer = null;
    this.stats.lastUpdateLatencyMs = elapsed;
    this.stats.tilesInLastVersion = provider.requested;
    this.stats.activeLayers = layers.length - 1; // minus base layer
  }

  clear() {
    const layers = this.viewer.imageryLayers;
    if (this.activeLayer) layers.remove(this.activeLayer, true);
    if (this.pendingLayer) layers.remove(this.pendingLayer, true);
    this.activeLayer = this.pendingLayer = null;
  }

  destroy() {
    this.clear();
    this.removePostRender();
    this.pool.destroy();
  }
}
