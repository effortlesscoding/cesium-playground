// Pure math shared by the main thread and the workers. No Cesium imports here so
// workers stay small and never touch DOM-dependent code.

export const TWO_PI = Math.PI * 2;
export const DEG = Math.PI / 180;

const WGS84_A = 6378137.0;
const WGS84_E2 = 6.69437999014e-3;
const MEAN_EARTH_RADIUS = 6371008.8;

/**
 * Clock positions are hours on a clock face: 12 (or 0) = north, 3 = east, 6 = south.
 * The shape sweeps clockwise from startClock to endClock. start === end means a full circle.
 */
export function clockSpan(field) {
  return (((field.endClock - field.startClock) % 12) + 12) % 12;
}

/**
 * Describes how the outer arc of a field is sampled.
 * - full circle: `segments` samples, closed ring
 * - pie/pacman: samples proportional to the swept angle, open arc (center closes it)
 */
export function shapeSpec(field, segments) {
  const span = clockSpan(field);
  const full = span < 1e-9;
  const sweep = full ? TWO_PI : (span / 12) * TWO_PI;
  const startBearing = (field.startClock / 12) * TWO_PI;
  if (full) {
    return { full, startBearing, step: sweep / segments, samples: segments };
  }
  const arcSegments = Math.max(2, Math.ceil((segments * sweep) / TWO_PI));
  return { full, startBearing, step: sweep / arcSegments, samples: arcSegments + 1 };
}

/**
 * Fills `outLat`/`outLon` (radians) with the great-circle destination from the
 * center for each bearing (given as precomputed cos/sin) at distance `distM`.
 */
export function destinations(latRad, lonRad, cosB, sinB, distM, outLat, outLon) {
  const d = distM / MEAN_EARTH_RADIUS;
  const sinLat = Math.sin(latRad);
  const cosLat = Math.cos(latRad);
  const sinD = Math.sin(d);
  const cosD = Math.cos(d);
  for (let j = 0; j < cosB.length; j++) {
    const sinLat2 = sinLat * cosD + cosLat * sinD * cosB[j];
    const lat2 = Math.asin(sinLat2);
    outLat[j] = lat2;
    outLon[j] = lonRad + Math.atan2(sinB[j] * sinD * cosLat, cosD - sinLat * sinLat2);
  }
}

/** Geodetic (radians, meters) -> ECEF on WGS84, written into `out` at `offset`. */
export function toEcef(latRad, lonRad, height, out, offset) {
  const sinLat = Math.sin(latRad);
  const cosLat = Math.cos(latRad);
  const n = WGS84_A / Math.sqrt(1 - WGS84_E2 * sinLat * sinLat);
  out[offset] = (n + height) * cosLat * Math.cos(lonRad);
  out[offset + 1] = (n + height) * cosLat * Math.sin(lonRad);
  out[offset + 2] = (n * (1 - WGS84_E2) + height) * sinLat;
}

/** Bearing cos/sin tables for one field's arc. */
export function bearingTables(spec) {
  const cosB = new Float64Array(spec.samples);
  const sinB = new Float64Array(spec.samples);
  for (let j = 0; j < spec.samples; j++) {
    const b = spec.startBearing + j * spec.step;
    cosB[j] = Math.cos(b);
    sinB[j] = Math.sin(b);
  }
  return { cosB, sinB };
}

/** Deterministic per-field color, [r, g, b] in 0..1. */
export function fieldRgb(id) {
  const h = (id * 0.61803398875) % 1;
  const s = 0.75;
  const l = 0.55;
  const k = (n) => (n + h * 12) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return [f(0), f(8), f(4)];
}
