const cores = typeof navigator !== "undefined" ? navigator.hardwareConcurrency || 4 : 4;

export const DEFAULT_PARAMS = {
  // Which renderer(s) are active: "primitive" (A), "imagery" (B), "both", "none" (baseline).
  mode: "primitive",
  fieldCount: 50,
  updateIntervalMs: 1000,
  jitterScale: 1,
  // Granularity: samples on a full circle's outer ring (pies get a proportional share).
  segments: 64,
  // Concentric rings used to tessellate the fill (A only). More rings = triangles hug the
  // ellipsoid better on big radii, and more vertices to push.
  rings: 6,
  // A: what to draw from the generated Cartesian3 points.
  primitiveStyle: "fill+outline", // "fill+outline" | "fill" | "outline" | "points"
  heightMeters: 10,
  // D: fields are projected onto any surface within +/- this many meters of the field's height.
  groundRange: 1000,
  // "flat" = ellipsoid; "hills" = synthetic terrain to show which modes follow the ground.
  terrain: "flat",
  // Worker pool sizes.
  geometryWorkers: Math.max(1, Math.min(4, cores - 1)),
  tileWorkers: Math.max(1, Math.min(4, cores - 1)),
  // B: imagery tile settings.
  tileSize: 256,
  maximumLevel: 14,
};

export const REGION = {
  // Abu Dhabi city
  lat: 24.4539,
  lon: 54.3773,
  spread: 0.12, // degrees of latitude; longitude spread is 2x this
  minRadius: 120,
  maxRadius: 1_000,
  cameraHeight: 40_000,
};
