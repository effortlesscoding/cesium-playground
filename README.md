# Cesium circular fields: Primitives vs Imagery

A React + Cesium testbed that draws N "circular fields" (center lat/lon, radius, start clock,
end clock) two different ways and measures the performance difference between them.

```
npm install
npm run dev        # http://localhost:5173
```

## Data model

```js
{ id, lat, lon, radius /* m */, startClock, endClock }
```

Clock values are hours on a clock face (12/0 = north, 3 = east). The shape sweeps clockwise
from `startClock` to `endClock`; `startClock === endClock` is a full circle, anything else is
a pie or pacman. `src/fields.js` generates `fieldCount` fields (default 50) and jitters
position, radius and clocks every `updateIntervalMs` (default 1000).

## A: worker-generated Cartesian3 vertices into a batched `Primitive`

- `src/workers/geometry.worker.js` (pool of `geometryWorkers`) computes ECEF positions for every
  field. It samples `segments` points per full circle and `rings` concentric rings so the fill
  follows the ellipsoid. It also builds triangle and line index buffers. Everything goes back
  as transferable typed arrays (zero-copy).
- `src/renderers/PrimitiveRenderer.js` wraps each field in a `GeometryInstance` (raw `Geometry`
  with DOUBLE positions) and batches them all into one fill `Primitive` and one outline
  `Primitive`. Style `points` instead feeds the same vertices to a `PointPrimitiveCollection`.

## B: worker-rasterized imagery tiles

- `src/workers/tile.worker.js` (pool of `tileWorkers`) draws the fields onto an `OffscreenCanvas`
  for each requested tile and returns an `ImageBitmap` (transferable).
- `src/renderers/ImageryRenderer.js` is a custom `ImageryProvider` limited to the bounding
  rectangle of the fields. Each update adds a new provider version as a hidden (alpha 0) layer.
  Once the globe has loaded all its tiles, that layer is shown and the old one is removed.
  This double buffering avoids flicker.

## Measuring

The side panel shows live counters. **Run benchmark** cycles None → A → B with the current
parameters (2.5 s warm-up, 8 s sample each). Keep the camera still while it runs.

| Metric | Meaning |
|---|---|
| Frame ms avg / worst | Wall time between rendered frames (includes GPU stalls) |
| Render CPU ms | `scene.preUpdate` → `postRender`, i.e. main-thread time inside Cesium's render |
| A: worker compute / round-trip | Vertex generation time in the worker / including postMessage |
| A: build instances | JS time to wrap results in `GeometryInstance`s |
| A: batch+upload frame | CPU time of the first frame after an update. Cesium's synchronous batching (combine, high/low position encoding, VBO upload) runs **on the main thread** here |
| B: tile worker / round-trip | Rasterization time per tile / including queueing |
| B: update → visible | Time until the new version's tiles are all loaded and swapped in |
| dropped ticks | Updates skipped because the previous one hadn't finished (backpressure) |

## Performance differences (what to expect)

| | A · Primitives | B · Imagery |
|---|---|---|
| Cost scales with | Total vertices (`fields × segments × rings`) | Visible tiles × fields per tile; independent of total vertex count |
| Main-thread work per update | **High**: Cesium batching is synchronous for custom `Geometry`, O(vertices) | **Low**: texture upload per tile only |
| Steady-state frame cost | One draw call per primitive. GPU cost grows with vertices | Free. Shapes are part of the globe's existing tile draws |
| Update latency | One worker round-trip + one frame | Worker round-trip **per tile** + globe tile-load cycle (tens to hundreds of ms) |
| Zooming / panning | No extra work. Geometry is resolution-independent | Every new tile/level triggers rasterization. Edges blur past `maximumLevel` |
| Visual quality | Exact vector edges. Big fills need enough `rings` or they cut under the ellipsoid | Always drapes on the surface/terrain. Pixel edges, line width in pixels per tile |
| Memory | Vertex buffers (24 B/vertex double before encoding) | One RGBA texture per tile (256 KB at 256², 1 MB at 512²) × 2 during swaps |

Headless smoke test (software GPU, so absolute numbers are not meaningful), 2,000 fields,
128 segments, 8 rings (~2M vertices), 1 s updates:

| mode | fps | frame avg ms | frame worst ms | render CPU worst ms | update latency |
|---|---|---|---|---|---|
| none | 58 | 17 | 39 | 4 | – |
| A | 14 | 246 | 1053 | 834 | ~40 ms + batching frame |
| B | 36 | 28 | 399 | 4 | ~110 ms |

At 50 fields both approaches run at full frame rate. The gap appears as vertex counts grow:
A pays for every vertex on the main thread each update, while B pays per visible tile in the
workers. Run the benchmark on your own hardware with your real field counts before deciding.
