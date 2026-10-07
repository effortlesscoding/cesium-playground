// Approach A worker: turns fields into ECEF (Cartesian3) vertex buffers + index buffers.
// Everything is packed into a few typed arrays and transferred (zero-copy) back.
import { DEG, shapeSpec, bearingTables, destinations, toEcef } from "../geo.js";

// Per-field metadata layout in `meta` (Int32Array, 6 ints per field).
// [vertexOffset, vertexCount, fillIndexOffset, fillIndexCount, lineIndexOffset, lineIndexCount]
const META = 6;

self.onmessage = (e) => {
  const { jobId, fields, segments, rings, height } = e.data;
  const t0 = performance.now();

  // Pass 1: sizes.
  const specs = new Array(fields.length);
  let totalV = 0;
  let totalFill = 0;
  let totalLine = 0;
  for (let i = 0; i < fields.length; i++) {
    const spec = shapeSpec(fields[i], segments);
    const m = spec.samples;
    const arcSegs = spec.full ? m : m - 1;
    spec.vCount = 1 + rings * m;
    spec.fillCount = 3 * (arcSegs + (rings - 1) * arcSegs * 2);
    spec.lineCount = 2 * (arcSegs + (spec.full ? 0 : 2));
    specs[i] = spec;
    totalV += spec.vCount;
    totalFill += spec.fillCount;
    totalLine += spec.lineCount;
  }

  const positions = new Float64Array(totalV * 3);
  const fill = new Uint32Array(totalFill);
  const lines = new Uint32Array(totalLine);
  const meta = new Int32Array(fields.length * META);
  const spheres = new Float64Array(fields.length * 4);

  // Pass 2: generate. Indices are local to each field so the main thread can slice them.
  let vOff = 0;
  let fOff = 0;
  let lOff = 0;
  for (let i = 0; i < fields.length; i++) {
    const f = fields[i];
    const spec = specs[i];
    const m = spec.samples;
    const arcSegs = spec.full ? m : m - 1;
    const lat = f.lat * DEG;
    const lon = f.lon * DEG;
    const { cosB, sinB } = bearingTables(spec);
    const ringLat = new Float64Array(m);
    const ringLon = new Float64Array(m);

    meta.set([vOff, spec.vCount, fOff, spec.fillCount, lOff, spec.lineCount], i * META);

    // Vertex 0: center. Ring k (1..rings), sample j: 1 + (k - 1) * m + j.
    toEcef(lat, lon, height, positions, vOff * 3);
    spheres[i * 4] = positions[vOff * 3];
    spheres[i * 4 + 1] = positions[vOff * 3 + 1];
    spheres[i * 4 + 2] = positions[vOff * 3 + 2];
    spheres[i * 4 + 3] = f.radius * 1.01 + Math.abs(height) + 1;

    for (let k = 1; k <= rings; k++) {
      destinations(lat, lon, cosB, sinB, (f.radius * k) / rings, ringLat, ringLon);
      const base = (vOff + 1 + (k - 1) * m) * 3;
      for (let j = 0; j < m; j++) toEcef(ringLat[j], ringLon[j], height, positions, base + j * 3);
    }

    const v = (k, j) => 1 + (k - 1) * m + (j % m);
    let fi = fOff;
    for (let j = 0; j < arcSegs; j++) {
      fill[fi++] = 0;
      fill[fi++] = v(1, j);
      fill[fi++] = v(1, j + 1);
    }
    for (let k = 2; k <= rings; k++) {
      for (let j = 0; j < arcSegs; j++) {
        const a = v(k - 1, j);
        const b = v(k - 1, j + 1);
        const c = v(k, j);
        const d = v(k, j + 1);
        fill[fi++] = a;
        fill[fi++] = c;
        fill[fi++] = d;
        fill[fi++] = a;
        fill[fi++] = d;
        fill[fi++] = b;
      }
    }

    let li = lOff;
    for (let j = 0; j < arcSegs; j++) {
      lines[li++] = v(rings, j);
      lines[li++] = v(rings, j + 1);
    }
    if (!spec.full) {
      lines[li++] = 0;
      lines[li++] = v(rings, 0);
      lines[li++] = v(rings, m - 1);
      lines[li++] = 0;
    }

    vOff += spec.vCount;
    fOff += spec.fillCount;
    lOff += spec.lineCount;
  }

  const computeMs = performance.now() - t0;
  self.postMessage(
    { jobId, ids: fields.map((f) => f.id), positions, fill, lines, meta, spheres, computeMs },
    [positions.buffer, fill.buffer, lines.buffer, meta.buffer, spheres.buffer],
  );
};
