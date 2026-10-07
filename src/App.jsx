import { useEffect, useRef, useState } from "react";
import {
  Cartesian3,
  ImageryLayer,
  TileMapServiceImageryProvider,
  Viewer,
  buildModuleUrl,
} from "cesium";
import { DEFAULT_PARAMS, REGION } from "./config.js";
import { Engine } from "./Engine.js";

const MODES = [
  { value: "primitive", label: "A · Primitives" },
  { value: "imagery", label: "B · Imagery" },
  { value: "both", label: "Both" },
  { value: "none", label: "None" },
];

const NUMBER_PARAMS = [
  { key: "fieldCount", label: "Fields", min: 1, max: 20000, step: 1 },
  { key: "updateIntervalMs", label: "Update interval (ms, 0 = off)", min: 0, max: 10000, step: 50 },
  { key: "jitterScale", label: "Jitter scale", min: 0, max: 20, step: 0.1 },
  { key: "segments", label: "Segments per circle", min: 4, max: 4096, step: 1 },
  { key: "rings", label: "Fill rings (A)", min: 1, max: 128, step: 1 },
  { key: "heightMeters", label: "Height m (A)", min: 0, max: 50000, step: 10 },
  { key: "geometryWorkers", label: "Geometry workers (A)", min: 1, max: 16, step: 1 },
  { key: "tileWorkers", label: "Tile workers (B)", min: 1, max: 16, step: 1 },
  { key: "maximumLevel", label: "Max tile level (B)", min: 0, max: 20, step: 1 },
];

const BENCH_MODES = ["none", "primitive", "imagery"];
const BENCH_WARMUP_MS = 2500;
const BENCH_SAMPLE_MS = 8000;

const fmt = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : "–");
const fmtInt = (v) => (Number.isFinite(v) ? Math.round(v).toLocaleString() : "–");
const fmtBytes = (b) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(0)} KB`);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function heapMb() {
  const m = performance.memory;
  return m ? m.usedJSHeapSize / 1e6 : NaN;
}

export default function App() {
  const containerRef = useRef(null);
  const engineRef = useRef(null);
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [snap, setSnap] = useState(null);
  const [bench, setBench] = useState({ running: false, step: "", results: [] });

  // Cesium viewer + engine: created once.
  useEffect(() => {
    const viewer = new Viewer(containerRef.current, {
      baseLayer: ImageryLayer.fromProviderAsync(
        TileMapServiceImageryProvider.fromUrl(buildModuleUrl("Assets/Textures/NaturalEarthII")),
      ),
      baseLayerPicker: false,
      geocoder: false,
      timeline: false,
      animation: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      fullscreenButton: false,
      infoBox: false,
      selectionIndicator: false,
      scene3DOnly: true, // skips 2D/CV projection work in the Primitive pipeline
    });
    viewer.scene.debugShowFramesPerSecond = false;
    viewer.camera.setView({ destination: Cartesian3.fromDegrees(REGION.lon, REGION.lat, REGION.cameraHeight) });
    engineRef.current = new Engine(viewer, DEFAULT_PARAMS);
    return () => {
      engineRef.current.destroy();
      engineRef.current = null;
      viewer.destroy();
    };
  }, []);

  useEffect(() => {
    engineRef.current?.setParams(params);
  }, [params]);

  // Poll stats twice a second; derive rates from deltas.
  useEffect(() => {
    let prev = null;
    const id = setInterval(() => {
      const e = engineRef.current;
      if (!e) return;
      const now = performance.now();
      const s = structuredClone(e.stats);
      const im = s.imagery;
      const dt = prev ? (now - prev.t) / 1000 : 1;
      const doneDelta = prev ? im.tilesDone - prev.tilesDone : 0;
      s.derived = {
        tilesPerSec: doneDelta / dt,
        avgTileWorkerMs: im.tilesDone ? im.tileWorkerMsTotal / im.tilesDone : NaN,
        avgTileWallMs: im.tilesDone ? im.tileWallMsTotal / im.tilesDone : NaN,
        emptyPct: im.tilesDone ? (100 * im.tilesEmpty) / im.tilesDone : NaN,
        heapMb: heapMb(),
      };
      prev = { t: now, tilesDone: im.tilesDone };
      setSnap(s);
    }, 500);
    return () => clearInterval(id);
  }, []);

  const set = (key, value) => setParams((p) => ({ ...p, [key]: value }));

  async function runBenchmark() {
    const e = engineRef.current;
    const original = params.mode;
    const results = [];
    setBench({ running: true, step: "", results });
    for (const mode of BENCH_MODES) {
      setBench((b) => ({ ...b, step: `${mode}: warming up` }));
      setParams((p) => ({ ...p, mode }));
      await sleep(BENCH_WARMUP_MS);
      e.resetCounters();
      setBench((b) => ({ ...b, step: `${mode}: sampling` }));
      const samples = [];
      const end = performance.now() + BENCH_SAMPLE_MS;
      while (performance.now() < end) {
        await sleep(1000);
        samples.push({ ...e.stats.frame });
      }
      const avg = (k) => samples.reduce((a, s) => a + s[k], 0) / samples.length;
      const max = (k) => Math.max(...samples.map((s) => s[k]));
      const rs = mode === "primitive" ? e.stats.primitive : mode === "imagery" ? e.stats.imagery : null;
      results.push({
        mode,
        fps: avg("fps"),
        avgFrameMs: avg("avgFrameMs"),
        worstFrameMs: max("maxFrameMs"),
        avgCpuMs: avg("avgRenderCpuMs"),
        worstCpuMs: max("maxRenderCpuMs"),
        updates: rs?.updates ?? NaN,
        dropped: rs?.droppedTicks ?? NaN,
        latencyMs: rs?.lastUpdateLatencyMs ?? NaN,
        heapMb: heapMb(),
      });
      setBench((b) => ({ ...b, results: [...results] }));
    }
    setParams((p) => ({ ...p, mode: original }));
    setBench((b) => ({ ...b, running: false, step: "done" }));
  }

  const a = snap?.primitive;
  const b = snap?.imagery;
  const f = snap?.frame;
  const d = snap?.derived;

  return (
    <div className="app">
      <div ref={containerRef} className="viewer" />
      <aside className="panel">
        <h1>Circular fields</h1>

        <section>
          <div className="modes">
            {MODES.map((m) => (
              <button
                key={m.value}
                className={params.mode === m.value ? "active" : ""}
                onClick={() => set("mode", m.value)}
                disabled={bench.running}
              >
                {m.label}
              </button>
            ))}
          </div>
        </section>

        <section className="params">
          {NUMBER_PARAMS.map((p) => (
            <label key={p.key}>
              <span>{p.label}</span>
              <input
                type="number"
                min={p.min}
                max={p.max}
                step={p.step}
                value={params[p.key]}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  if (Number.isFinite(v)) set(p.key, Math.min(p.max, Math.max(p.min, v)));
                }}
              />
            </label>
          ))}
          <label>
            <span>Draw style (A)</span>
            <select value={params.primitiveStyle} onChange={(e) => set("primitiveStyle", e.target.value)}>
              <option value="fill+outline">fill + outline</option>
              <option value="fill">fill</option>
              <option value="outline">outline</option>
              <option value="points">points</option>
            </select>
          </label>
          <label>
            <span>Tile size (B)</span>
            <select value={params.tileSize} onChange={(e) => set("tileSize", Number(e.target.value))}>
              <option value={256}>256</option>
              <option value={512}>512</option>
            </select>
          </label>
        </section>

        <section>
          <h2>Frame</h2>
          <table>
            <tbody>
              <tr><td>FPS</td><td>{fmt(f?.fps)}</td></tr>
              <tr><td>Frame ms avg / worst</td><td>{fmt(f?.avgFrameMs)} / {fmt(f?.maxFrameMs)}</td></tr>
              <tr><td>Render CPU ms avg / worst</td><td>{fmt(f?.avgRenderCpuMs, 2)} / {fmt(f?.maxRenderCpuMs)}</td></tr>
              <tr><td>JS heap</td><td>{Number.isFinite(d?.heapMb) ? `${fmt(d.heapMb, 0)} MB` : "n/a"}</td></tr>
            </tbody>
          </table>
        </section>

        <section>
          <h2>A · Primitives</h2>
          <table>
            <tbody>
              <tr><td>Vertices / triangles</td><td>{fmtInt(a?.vertices)} / {fmtInt(a?.triangles)}</td></tr>
              <tr><td>Worker compute ms</td><td>{fmt(a?.workerComputeMs, 2)}</td></tr>
              <tr><td>Worker round-trip ms</td><td>{fmt(a?.workerWallMs, 2)}</td></tr>
              <tr><td>Main: build instances ms</td><td>{fmt(a?.mainBuildMs, 2)}</td></tr>
              <tr><td>Main: batch+upload frame ms</td><td>{fmt(a?.firstFrameMs, 2)}</td></tr>
              <tr><td>Transferred / update</td><td>{a ? fmtBytes(a.transferBytes) : "–"}</td></tr>
              <tr><td>Updates / dropped ticks</td><td>{fmtInt(a?.updates)} / {fmtInt(a?.droppedTicks)}</td></tr>
            </tbody>
          </table>
        </section>

        <section>
          <h2>B · Imagery</h2>
          <table>
            <tbody>
              <tr><td>Tiles / sec</td><td>{fmt(d?.tilesPerSec)}</td></tr>
              <tr><td>Tiles in last version</td><td>{fmtInt(b?.tilesInLastVersion)}</td></tr>
              <tr><td>Avg tile worker ms</td><td>{fmt(d?.avgTileWorkerMs, 2)}</td></tr>
              <tr><td>Avg tile round-trip ms</td><td>{fmt(d?.avgTileWallMs, 2)}</td></tr>
              <tr><td>Empty tiles</td><td>{fmt(d?.emptyPct, 0)}%</td></tr>
              <tr><td>Update → visible ms</td><td>{fmt(b?.lastUpdateLatencyMs, 0)}</td></tr>
              <tr><td>Updates / dropped ticks</td><td>{fmtInt(b?.updates)} / {fmtInt(b?.droppedTicks)}</td></tr>
            </tbody>
          </table>
        </section>

        <section>
          <h2>Benchmark</h2>
          <p className="hint">
            Runs None → A → B for {BENCH_SAMPLE_MS / 1000}s each with the current parameters. Keep the camera still.
          </p>
          <button onClick={runBenchmark} disabled={bench.running}>
            {bench.running ? `Running… ${bench.step}` : "Run benchmark"}
          </button>
          {bench.results.length > 0 && (
            <table className="bench">
              <thead>
                <tr>
                  <th>mode</th><th>fps</th><th>frame avg</th><th>frame worst</th><th>cpu avg</th><th>cpu worst</th><th>upd / drop</th><th>latency</th>
                </tr>
              </thead>
              <tbody>
                {bench.results.map((r) => (
                  <tr key={r.mode}>
                    <td>{r.mode}</td>
                    <td>{fmt(r.fps)}</td>
                    <td>{fmt(r.avgFrameMs)}</td>
                    <td>{fmt(r.worstFrameMs)}</td>
                    <td>{fmt(r.avgCpuMs, 2)}</td>
                    <td>{fmt(r.worstCpuMs)}</td>
                    <td>{Number.isFinite(r.updates) ? `${r.updates} / ${r.dropped}` : "–"}</td>
                    <td>{fmt(r.latencyMs, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      </aside>
    </div>
  );
}
