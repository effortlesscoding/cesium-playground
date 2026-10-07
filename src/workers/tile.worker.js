// Approach B worker: rasterizes fields into imagery tiles with OffscreenCanvas.
// The main thread broadcasts each new field "version"; tile requests reference a version.
import { DEG, shapeSpec, bearingTables, destinations, fieldRgb } from "../geo.js";

const versions = new Map(); // version -> { shapes }
const canvases = new Map(); // tileSize -> OffscreenCanvas (reused; transferToImageBitmap clears it)

function buildShapes(fields, segments) {
  return fields.map((f) => {
    const spec = shapeSpec(f, segments);
    const { cosB, sinB } = bearingTables(spec);
    const lat = new Float64Array(spec.samples);
    const lon = new Float64Array(spec.samples);
    destinations(f.lat * DEG, f.lon * DEG, cosB, sinB, f.radius, lat, lon);

    // Outline as lon/lat degrees; pies start and end at the center.
    const n = spec.samples + (spec.full ? 0 : 1);
    const pts = new Float64Array(n * 2);
    let p = 0;
    if (!spec.full) {
      pts[p++] = f.lon;
      pts[p++] = f.lat;
    }
    let west = Infinity, east = -Infinity, south = Infinity, north = -Infinity;
    for (let j = 0; j < spec.samples; j++) {
      pts[p++] = lon[j] / DEG;
      pts[p++] = lat[j] / DEG;
    }
    for (let j = 0; j < pts.length; j += 2) {
      west = Math.min(west, pts[j]);
      east = Math.max(east, pts[j]);
      south = Math.min(south, pts[j + 1]);
      north = Math.max(north, pts[j + 1]);
    }
    const [r, g, b] = fieldRgb(f.id).map((c) => Math.round(c * 255));
    return {
      pts,
      west, east, south, north,
      fill: `rgba(${r},${g},${b},0.45)`,
      stroke: `rgb(${r},${g},${b})`,
    };
  });
}

function getCanvas(size) {
  let c = canvases.get(size);
  if (!c) {
    c = new OffscreenCanvas(size, size);
    canvases.set(size, c);
  }
  return c;
}

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "fields") {
    const t0 = performance.now();
    versions.set(msg.version, { shapes: buildShapes(msg.fields, msg.segments) });
    for (const v of versions.keys()) if (v < msg.version - 3) versions.delete(v);
    self.postMessage({ type: "fieldsReady", version: msg.version, ms: performance.now() - t0 });
    return;
  }

  if (msg.type === "tile") {
    const t0 = performance.now();
    const { id, version, west, south, east, north, size } = msg;
    const set = versions.get(version);
    const hits = set ? set.shapes.filter((s) => s.east >= west && s.west <= east && s.north >= south && s.south <= north) : [];
    if (hits.length === 0) {
      self.postMessage({ type: "tile", id, empty: true, ms: performance.now() - t0 });
      return;
    }

    const canvas = getCanvas(size);
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, size, size);
    const sx = size / (east - west);
    // Cesium uploads imagery ImageBitmaps pre-flipped (WebGL ignores UNPACK_FLIP_Y for bitmaps),
    // so draw with south at the top of the canvas.
    const sy = size / (north - south);
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    for (const s of hits) {
      const pts = s.pts;
      ctx.beginPath();
      ctx.moveTo((pts[0] - west) * sx, (pts[1] - south) * sy);
      for (let j = 2; j < pts.length; j += 2) ctx.lineTo((pts[j] - west) * sx, (pts[j + 1] - south) * sy);
      ctx.closePath();
      ctx.fillStyle = s.fill;
      ctx.fill();
      ctx.strokeStyle = s.stroke;
      ctx.stroke();
    }
    const bitmap = canvas.transferToImageBitmap();
    self.postMessage({ type: "tile", id, bitmap, shapes: hits.length, ms: performance.now() - t0 }, [bitmap]);
  }
};
