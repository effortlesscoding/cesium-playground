// /#/faces: paste cube face definitions ({ n, u, v } objects) and see whether they are correct.
// No Cesium here; it's a tiny orthographic SVG 3D view plus the math checks.
import { useMemo, useRef, useState } from "react";
import "./FacesPage.css";

const DEFAULT_TEXT = `const faces = [
    { n: [1, 0, 0], u: [0, 1, 0], v: [0, 0, 1] },
    { n: [-1, 0, 0], u: [0, 0, 1], v: [0, 1, 0] },
    { n: [0, 1, 0], u: [0, 0, 1], v: [1, 0, 0] },
    { n: [0, -1, 0], u: [1, 0, 0], v: [0, 0, 1] },
    { n: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] },
    { n: [0, 0, -1], u: [0, 1, 0], v: [1, 0, 0] },
  ];`;

// Same as DEFAULT_TEXT, but face 3 has u and v swapped, so its triangles would face inward.
const BROKEN_TEXT = DEFAULT_TEXT.replace(
  "{ n: [0, 1, 0], u: [0, 0, 1], v: [1, 0, 0] }",
  "{ n: [0, 1, 0], u: [1, 0, 0], v: [0, 0, 1] }",
);

// The corners of each face quad, in (u, v) coordinates, in the order RedBoxRenderer emits them.
const CORNERS = [[-1, -1], [1, -1], [1, 1], [-1, 1]];

// --- tiny vector helpers ---------------------------------------------------------------------
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const near = (a, b, eps = 1e-6) => Math.abs(a - b) < eps;
const fmt = (v) => `[${v.map((x) => +x.toFixed(3)).join(", ")}]`;

/** Pulls every `{ n: [...], u: [...], v: [...] }` out of pasted text. No eval. */
function parseFaces(text) {
  const faces = [];
  for (const block of text.matchAll(/\{[^{}]*\}/g)) {
    const grab = (key) => {
      const m = new RegExp(`\\b${key}\\s*:\\s*\\[([^\\]]*)\\]`).exec(block[0]);
      if (!m) return null;
      const v = m[1].split(",").map((s) => Number(s.trim()));
      return v.length === 3 && v.every(Number.isFinite) ? v : null;
    };
    const n = grab("n"), u = grab("u"), v = grab("v");
    if (n && u && v) faces.push({ n, u, v });
  }
  return faces;
}

/** Builds the quad for a face and runs every check. */
function analyze(face, index) {
  const { n, u, v } = face;
  const center = scale(n, 0.5);
  // p = 0.5 n + 0.5 (su u + sv v), exactly as buildCube() in RedBoxRenderer.js does.
  const corners = CORNERS.map(([su, sv]) => add(center, add(scale(u, 0.5 * su), scale(v, 0.5 * sv))));
  const handed = dot(cross(u, v), n);
  // Direction the quad's first triangle faces: (p1 - p0) x (p2 - p0). Counter-clockwise seen from
  // outside means this points the same way as the outward normal.
  const winding = dot(cross(sub(corners[1], corners[0]), sub(corners[2], corners[0])), n);
  const unit = [n, u, v].every((a) => near(len(a), 1));
  const orthogonal = near(dot(n, u), 0) && near(dot(n, v), 0) && near(dot(u, v), 0);
  const checks = [
    { ok: unit, label: "n, u, v are unit length" },
    { ok: orthogonal, label: "n, u, v are perpendicular" },
    { ok: near(handed, 1), label: `u × v = n  (right-handed; dot = ${+handed.toFixed(2)})` },
    { ok: winding > 0, label: `corners 0→1→2→3 run counter-clockwise from outside  (dot = ${+winding.toFixed(2)})` },
  ];
  return { index, ...face, center, corners, checks, ok: checks.every((c) => c.ok) };
}

/** Do the faces cover +x, -x, +y, -y, +z, -z exactly once each? */
function coverage(faces) {
  const axes = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  const names = ["+x", "-x", "+y", "-y", "+z", "-z"];
  return axes.map((a, i) => ({
    name: names[i],
    count: faces.filter((f) => f.n.every((c, k) => near(c, a[k]))).length,
  }));
}

// --- 3D view ---------------------------------------------------------------------------------
// Orthographic. Rotate about z (yaw), then tilt about x (pitch). The viewer looks along +y, so a
// smaller y means closer to the viewer.
function makeProjector(yaw, pitch) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const rotate = ([x, y, z]) => {
    const x1 = cy * x - sy * y;
    const y1 = sy * x + cy * y;
    return [x1, cp * y1 - sp * z, sp * y1 + cp * z]; // [right, depth, up]
  };
  return {
    rotate,
    point: (p) => {
      const [x, depth, up] = rotate(p);
      return { x, y: -up, depth };
    },
  };
}

// Face colors avoid red, which is reserved for failures.
const PALETTE = ["#2ec4b6", "#e9c46a", "#9b5de5", "#00bbf9", "#f15bb5", "#90be6d"];
const hue = (i) => PALETTE[i % PALETTE.length];
const pts = (list) => list.map((p) => `${p.x},${p.y}`).join(" ");

function Arrow({ from, to, color, label, width = 0.02 }) {
  const dx = to.x - from.x, dy = to.y - from.y;
  const l = Math.hypot(dx, dy) || 1;
  const ux = dx / l, uy = dy / l, head = 0.07;
  const tip = to;
  const base = { x: tip.x - ux * head, y: tip.y - uy * head };
  return (
    <g>
      <line x1={from.x} y1={from.y} x2={base.x} y2={base.y} stroke={color} strokeWidth={width} />
      <polygon
        points={`${tip.x},${tip.y} ${base.x - uy * head * 0.45},${base.y + ux * head * 0.45} ${base.x + uy * head * 0.45},${base.y - ux * head * 0.45}`}
        fill={color}
      />
      {label && (
        <text x={tip.x + ux * 0.07} y={tip.y + uy * 0.07 + 0.03} fill={color} fontSize="0.1" fontWeight="700" textAnchor="middle">
          {label}
        </text>
      )}
    </g>
  );
}

function CubeView({ faces, selected, options }) {
  const [view, setView] = useState({ yaw: -0.6, pitch: 0.55 });
  const drag = useRef(null);
  const proj = makeProjector(view.yaw, view.pitch);

  // With a face selected, only that face gets the arrows and corner numbers; otherwise all do.
  const showDetail = (index) => selected === null || selected === index;

  const drawn = faces
    .map((f) => {
      const corners = f.corners.map(proj.point);
      const center = proj.point(f.center);
      const normalDepth = proj.rotate(f.n)[1]; // < 0: the face points toward the viewer
      return { f, corners, center, frontFacing: normalDepth < -1e-9, depth: center.depth };
    })
    .filter((d) => options.showBack || d.frontFacing || d.f.index === selected) // a selected face is always shown
    .sort((a, b) => b.depth - a.depth); // far to near

  const onPointerDown = (e) => {
    drag.current = { x: e.clientX, y: e.clientY, ...view };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e) => {
    if (!drag.current) return;
    const d = drag.current;
    setView({
      yaw: d.yaw + (e.clientX - d.x) * 0.01,
      pitch: Math.max(-1.5, Math.min(1.5, d.pitch + (e.clientY - d.y) * 0.01)),
    });
  };

  return (
    <div className="cube">
      <svg
        viewBox="-1.5 -1.5 3 3"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={() => (drag.current = null)}
      >
        {drawn.map(({ f, corners, center, frontFacing }) => {
          const isSel = selected === f.index;
          const color = f.ok ? hue(f.index) : "#ff4d4d";
          const dim = selected !== null && !isSel;
          return (
            <g key={f.index} opacity={dim ? 0.35 : 1}>
              <polygon
                points={pts(corners)}
                fill={color}
                fillOpacity={frontFacing ? 0.32 : 0.14}
                stroke={color}
                strokeWidth={isSel ? 0.035 : 0.015}
                strokeDasharray={frontFacing ? undefined : "0.05 0.04"}
              />
              {options.showWinding && showDetail(f.index) && (
                <g>
                  {/* Path 0 -> 1 -> 2 -> 3, pulled toward the face center so it doesn't overlap the edges. */}
                  {corners.map((c, i) => {
                    const next = corners[(i + 1) % 4];
                    const inset = (p) => ({ x: p.x + (center.x - p.x) * 0.22, y: p.y + (center.y - p.y) * 0.22 });
                    return <Arrow key={i} from={inset(c)} to={inset(next)} color="#fff" width={0.012} />;
                  })}
                  {corners.map((c, i) => {
                    const p = { x: c.x + (center.x - c.x) * 0.12, y: c.y + (center.y - c.y) * 0.12 };
                    return (
                      <g key={i}>
                        <circle cx={p.x} cy={p.y} r="0.065" fill="#111" stroke="#fff" strokeWidth="0.01" />
                        <text x={p.x} y={p.y + 0.03} fill="#fff" fontSize="0.085" textAnchor="middle">{i}</text>
                      </g>
                    );
                  })}
                </g>
              )}
              {options.showAxes && showDetail(f.index) && (
                <g>
                  <Arrow from={center} to={proj.point(add(f.center, scale(f.n, 0.5)))} color="#4ea1ff" label="n" />
                  <Arrow from={center} to={proj.point(add(f.center, scale(f.u, 0.3)))} color="#ff5c5c" label="u" />
                  <Arrow from={center} to={proj.point(add(f.center, scale(f.v, 0.3)))} color="#4cd964" label="v" />
                </g>
              )}
              <text x={center.x} y={center.y - 0.12} fill="#fff" fontSize="0.1" textAnchor="middle" fontWeight="700">
                {f.index}
              </text>
            </g>
          );
        })}
        {/* World axes at the cube's corner for orientation. */}
        {options.showWorld &&
          [["x", [1, 0, 0], "#ff5c5c"], ["y", [0, 1, 0], "#4cd964"], ["z", [0, 0, 1], "#4ea1ff"]].map(([name, dir, color]) => (
            <Arrow key={name} from={proj.point([-0.9, -0.9, -0.5])} to={proj.point(add([-0.9, -0.9, -0.5], scale(dir, 0.5)))} color={color} label={name} width={0.012} />
          ))}
      </svg>
      <button className="link" onClick={() => setView({ yaw: -0.6, pitch: 0.55 })}>Reset view</button>
    </div>
  );
}

export default function FacesPage() {
  const [text, setText] = useState(DEFAULT_TEXT);
  const [selected, setSelected] = useState(null);
  const [options, setOptions] = useState({ showAxes: true, showWinding: true, showBack: false, showWorld: true });

  const faces = useMemo(() => parseFaces(text).map(analyze), [text]);
  const cover = useMemo(() => coverage(faces), [faces]);
  const allOk = faces.length > 0 && faces.every((f) => f.ok) && cover.every((c) => c.count === 1);
  const toggle = (key) => setOptions((o) => ({ ...o, [key]: !o[key] }));

  return (
    <div className="faces">
      <header>
        <a href="#/">← Playground</a>
        <h1>Cube face checker</h1>
      </header>

      <section className="explain">
        <p>
          Each face is described by <b className="n">n</b> (the outward normal), and two in-plane axes{" "}
          <b className="u">u</b> and <b className="v">v</b>. The four corners are{" "}
          <code>0.5·n + 0.5·(±u ± v)</code>, emitted in the order (−u−v), (+u−v), (+u+v), (−u+v) and drawn as two
          triangles. A face is correct when:
        </p>
        <ol>
          <li><b>u × v = n</b>. Otherwise the corners wind the wrong way.</li>
          <li><b>Corners 0→1→2→3 go counter-clockwise when you look at the face from outside.</b> That is what makes it front-facing, so back-face culling keeps it (and removes the far side).</li>
          <li>All six directions (±x, ±y, ±z) are covered once.</li>
        </ol>
        <p className="hint">Drag the cube to rotate. Click a face's card to show only its arrows and corner order. Red faces fail a check; dashed outlines face away from you.</p>
      </section>

      <div className="grid">
        <div className="left">
          <CubeView faces={faces} selected={selected} options={options} />
          <div className="options">
            {[["showAxes", "n / u / v arrows"], ["showWinding", "corner order 0→3"], ["showBack", "back faces"], ["showWorld", "world axes"]].map(([k, label]) => (
              <label key={k}><input type="checkbox" checked={options[k]} onChange={() => toggle(k)} /> {label}</label>
            ))}
          </div>
        </div>

        <div className="right">
          <div className={`verdict ${allOk ? "ok" : "bad"}`}>
            {faces.length === 0
              ? "No faces found. Paste objects like { n: [1,0,0], u: [0,1,0], v: [0,0,1] }."
              : allOk
                ? "✓ All faces are correct and the cube is complete."
                : "✗ Something is wrong. See the red faces and the checks below."}
          </div>

          <div className="coverage">
            {cover.map((c) => (
              <span key={c.name} className={c.count === 1 ? "ok" : "bad"}>
                {c.name}: {c.count}
              </span>
            ))}
          </div>

          <div className="cards">
            {faces.map((f) => (
              <div
                key={f.index}
                className={`card ${f.ok ? "" : "failed"} ${selected === f.index ? "selected" : ""}`}
                onClick={() => setSelected(selected === f.index ? null : f.index)}
              >
                <div className="title">
                  <span className="swatch" style={{ background: f.ok ? hue(f.index) : "#ff4d4d" }} />
                  Face {f.index} {f.ok ? "✓" : "✗"}
                </div>
                <div className="vecs">
                  <span className="n">n {fmt(f.n)}</span>
                  <span className="u">u {fmt(f.u)}</span>
                  <span className="v">v {fmt(f.v)}</span>
                </div>
                <ul>
                  {f.checks.map((c) => (
                    <li key={c.label} className={c.ok ? "ok" : "bad"}>{c.ok ? "✓" : "✗"} {c.label}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </div>

      <section className="input">
        <div className="row">
          <h2>Paste your faces</h2>
          <button className="link" onClick={() => setText(DEFAULT_TEXT)}>Load correct example</button>
          <button className="link" onClick={() => setText(BROKEN_TEXT)}>Load broken example (face 2 has u and v swapped)</button>
        </div>
        <textarea value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} rows={10} />
      </section>
    </div>
  );
}
