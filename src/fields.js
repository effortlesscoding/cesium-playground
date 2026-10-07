import { clockSpan } from "./geo.js";

const rand = (min, max) => min + Math.random() * (max - min);
const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
const wrapClock = (c) => ((c % 12) + 12) % 12;

/**
 * Generates `count` circular fields inside the region.
 * Roughly a third are full circles (startClock === endClock), the rest pies/pacmans.
 */
export function generateFields(count, region) {
  const fields = new Array(count);
  for (let i = 0; i < count; i++) {
    const start = rand(0, 12);
    const full = Math.random() < 0.35;
    const span = full ? 0 : rand(0.5, 11.5);
    fields[i] = {
      id: i,
      lat: rand(region.lat - region.spread / 2, region.lat + region.spread / 2),
      lon: rand(region.lon - region.spread, region.lon + region.spread),
      radius: rand(region.minRadius, region.maxRadius),
      startClock: start,
      endClock: wrapClock(start + span),
    };
  }
  return fields;
}

/**
 * Returns a new array with every field nudged slightly. `scale` multiplies the jitter.
 * Full circles rotate as a unit so they stay full circles.
 */
export function jitterFields(fields, region, scale) {
  // Positional jitter is proportional to the region so fields stay put at any scale.
  const drift = (region.spread / 360) * scale;
  return fields.map((f) => {
    const full = clockSpan(f) === 0;
    const startDelta = rand(-0.15, 0.15) * scale;
    let startClock = wrapClock(f.startClock + startDelta);
    let endClock;
    if (full) {
      endClock = startClock;
    } else {
      const span = clamp(clockSpan(f) + rand(-0.15, 0.15) * scale, 0.5, 11.5);
      endClock = wrapClock(startClock + span);
    }
    return {
      id: f.id,
      lat: clamp(f.lat + rand(-drift, drift), -80, 80),
      lon: f.lon + rand(-drift, drift),
      radius: clamp(f.radius * (1 + rand(-0.03, 0.03) * scale), region.minRadius / 2, region.maxRadius * 2),
      startClock,
      endClock,
    };
  });
}
