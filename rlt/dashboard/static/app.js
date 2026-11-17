"use strict";
// RL Trading Lab dashboard. Plain DOM + SVG; every data string is inserted with textContent.

const $ = (s) => document.querySelector(s);
const SLOTS = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"];
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const S = {
  exp: null, runs: [], summary: null, baselines: [], bench: {}, selected: new Map(), metric: null, // summary null = loading
  progress: {}, ledgers: {}, sortKey: "run", sortDir: 1, views: {}, timer: null, lastEvalCount: -1,
};

async function api(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(path + " " + r.status);
  return r.json();
}

// ------------------------------------------------------------------------------------------ helpers
const isNum = (v) => typeof v === "number" && isFinite(v);
const f2 = (v, d = 2) => (isNum(v) ? v.toFixed(d) : "–");
const pct = (v, d = 1) => (isNum(v) ? (100 * v).toFixed(d) + "%" : "–");
function el(tag, attrs = {}, text) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else e.setAttribute(k, v);
  }
  if (text != null) e.textContent = text;
  return e;
}
const NS = "http://www.w3.org/2000/svg";
function sv(tag, attrs = {}) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}
function stext(x, y, str, attrs = {}) {
  const t = sv("text", { x, y, ...attrs });
  t.textContent = str;
  return t;
}
const linear = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / ((d1 - d0) || 1)) * (r1 - r0);
function niceTicks(lo, hi, n = 5) {
  if (!isNum(lo) || !isNum(hi)) return [];
  if (lo === hi) { lo -= 1; hi += 1; }
  const raw = (hi - lo) / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const e = raw / mag;
  const step = (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9 * step; t += step) out.push(+t.toFixed(10));
  return out;
}
function fmtTick(v) {
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + "M";
  if (a >= 1e4) return (v / 1e3).toFixed(0) + "K";
  if (a >= 1000) return v.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (a === 0) return "0";
  if (a < 0.01) return v.toExponential(0);
  if (a < 1) return v.toFixed(2);
  if (a < 10) return v.toFixed(1);
  return v.toFixed(0);
}
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function fmtDate(ms, full = false) {
  const d = new Date(ms);
  return full ? `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}` : `${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}
function dateTicks(x0, x1, maxTicks) {
  const years = (x1 - x0) / (365.25 * 864e5);
  const stepM = years > 8 ? 24 : years > 4 ? 12 : years > 1.5 ? 6 : 3;
  const d = new Date(x0);
  let y = d.getUTCFullYear(), m = Math.ceil(d.getUTCMonth() / stepM) * stepM;
  const out = [];
  for (let k = 0; k < 200; k++) {
    const t = Date.UTC(y + Math.floor(m / 12), m % 12, 1);
    if (t > x1) break;
    if (t >= x0) out.push(t);
    m += stepM;
  }
  while (out.length > maxTicks) out.splice(1, 1);
  return out;
}
function nearest(pts, x) {
  if (!pts.length) return null;
  let lo = 0, hi = pts.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (pts[mid][0] < x) lo = mid; else hi = mid; }
  return Math.abs(pts[lo][0] - x) <= Math.abs(pts[hi][0] - x) ? pts[lo] : pts[hi];
}
function table(headers, rows, numericCols = new Set()) {
  const t = el("table");
  const tr = el("tr");
  headers.forEach((h, i) => tr.append(el("th", { class: numericCols.has(i) ? "num" : "" }, h)));
  const thead = el("thead"); thead.append(tr);
  const tb = el("tbody");
  for (const r of rows) {
    const row = el("tr");
    r.forEach((c, i) => row.append(el("td", { class: numericCols.has(i) ? "num" : "" }, c)));
    tb.append(row);
  }
  t.append(thead, tb);
  const w = el("div", { class: "table-wrap" }); w.append(t);
  return w;
}
function withView(host, chartFn, tableFn) {
  const card = host.closest(".card");
  const view = S.views[host.id] || "chart";
  host.replaceChildren();
  if (view === "table") host.append(tableFn()); else chartFn();
  const btn = card && card.querySelector(".toggle");
  if (btn) {
    btn.textContent = view === "table" ? "Chart" : "Table";
    btn.onclick = () => { S.views[host.id] = view === "table" ? "chart" : "table"; withView(host, chartFn, tableFn); };
  }
}

// -------------------------------------------------------------------------------------- line chart
function lineChart(host, series, opts = {}) {
  host.replaceChildren();
  const valid = series.filter((s) => s.pts.some((p) => isNum(p[1])));
  if (!valid.length) { host.append(el("div", { class: "empty" }, opts.empty || "No data yet")); return; }
  if (valid.length >= 2) {
    const lg = el("div", { class: "legend" });
    for (const s of valid) {
      const it = el("span", { class: "item" });
      const k = el("span", { class: "key" }); k.style.background = s.color;
      it.append(k, document.createTextNode(s.name));
      lg.append(it);
    }
    host.append(lg);
  }
  const W = Math.max(300, host.clientWidth), H = opts.height || 260;
  const m = { l: 50, r: 14, t: 10, b: 26 };
  const xs = [], ys = [];
  for (const s of valid) for (const [x, y] of s.pts) if (isNum(y)) { xs.push(x); ys.push(y); }
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (opts.base != null) { y0 = Math.min(y0, opts.base); y1 = Math.max(y1, opts.base); }
  const pad = (y1 - y0) * 0.06 || 0.5;
  y0 -= pad; y1 += pad;
  const X = linear(x0, x1, m.l, W - m.r), Y = linear(y0, y1, H - m.b, m.t);
  const svg = sv("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.aria || "line chart" });
  for (const t of niceTicks(y0, y1, 5)) {
    svg.append(sv("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: css("--grid"), "stroke-width": 1 }));
    svg.append(stext(m.l - 6, Y(t) + 4, fmtTick(t), { "text-anchor": "end", class: "axis-label" }));
  }
  const maxT = Math.max(2, Math.floor(W / 110));
  const xt = opts.dates ? dateTicks(x0, x1, maxT) : niceTicks(x0, x1, maxT);
  for (const t of xt) {
    const x = X(t);
    const anchor = x > W - m.r - 34 ? "end" : x < m.l + 20 ? "start" : "middle";
    svg.append(stext(x, H - 7, opts.dates ? fmtDate(t) : fmtTick(t), { "text-anchor": anchor, class: "axis-label" }));
  }
  svg.append(sv("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: css("--axis"), "stroke-width": 1 }));
  if (opts.base != null && y0 < opts.base && y1 > opts.base)
    svg.append(sv("line", { x1: m.l, x2: W - m.r, y1: Y(opts.base), y2: Y(opts.base), stroke: css("--axis"), "stroke-width": 1 }));
  const ordered = [...valid].sort((a, b) => (b.ref ? 1 : 0) - (a.ref ? 1 : 0));
  for (const s of ordered) {
    let d = "", pen = false;
    for (const [x, y] of s.pts) {
      if (!isNum(y)) { pen = false; continue; }
      d += (pen ? "L" : "M") + X(x).toFixed(1) + "," + Y(y).toFixed(1);
      pen = true;
    }
    svg.append(sv("path", { d, fill: "none", stroke: s.color, "stroke-width": s.ref ? 1.5 : 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
  }
  const cross = sv("line", { y1: m.t, y2: H - m.b, stroke: css("--axis"), "stroke-width": 1, visibility: "hidden" });
  svg.append(cross);
  const dots = valid.map((s) => { const c = sv("circle", { r: 4, fill: s.color, stroke: css("--surface-1"), "stroke-width": 2, visibility: "hidden" }); svg.append(c); return c; });
  const hit = sv("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" });
  svg.append(hit);
  const wrap = el("div"); wrap.style.position = "relative"; wrap.append(svg);
  const tip = el("div", { class: "tooltip" }); wrap.append(tip);
  host.append(wrap);
  const fmtV = opts.fmt || ((v) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3)));
  hit.addEventListener("pointermove", (ev) => {
    const r = svg.getBoundingClientRect();
    const xv = x0 + ((ev.clientX - r.left - m.l) / (W - m.l - m.r)) * (x1 - x0);
    let best = null;
    for (const s of valid) { const p = nearest(s.pts, xv); if (p && (best === null || Math.abs(p[0] - xv) < Math.abs(best - xv))) best = p[0]; }
    if (best === null) return;
    cross.setAttribute("x1", X(best)); cross.setAttribute("x2", X(best)); cross.setAttribute("visibility", "visible");
    tip.replaceChildren(el("div", { class: "t-head" }, opts.dates ? fmtDate(best, true) : `${opts.xName || "step"} ${Math.round(best).toLocaleString("en-US")}`));
    valid.forEach((s, i) => {
      const p = nearest(s.pts, best);
      const tol = (x1 - x0) * 0.02;
      if (!p || !isNum(p[1]) || Math.abs(p[0] - best) > tol) { dots[i].setAttribute("visibility", "hidden"); return; }
      dots[i].setAttribute("cx", X(p[0])); dots[i].setAttribute("cy", Y(p[1])); dots[i].setAttribute("visibility", "visible");
      const row = el("div", { class: "row" });
      const k = el("span", { class: "key" }); k.style.background = s.color;
      row.append(k, el("span", { class: "v" }, fmtV(p[1])), el("span", { class: "n" }, s.name));
      tip.append(row);
    });
    tip.style.display = "block";
    const left = X(best) + 14;
    tip.style.left = (left + tip.offsetWidth > W ? X(best) - 14 - tip.offsetWidth : left) + "px";
    tip.style.top = m.t + 4 + "px";
  });
  hit.addEventListener("pointerleave", () => {
    cross.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); tip.style.display = "none";
  });
}

// --------------------------------------------------------------------------------------- bar chart
function barPath(xa, xb, yc, h, r) {
  const y0 = yc - h / 2, y1 = yc + h / 2;
  const len = Math.abs(xb - xa);
  r = Math.min(r, len, h / 2);
  if (len < 0.5) return "";
  if (xb >= xa)
    return `M${xa},${y0}H${xb - r}Q${xb},${y0} ${xb},${y0 + r}V${y1 - r}Q${xb},${y1} ${xb - r},${y1}H${xa}Z`;
  return `M${xa},${y0}H${xb + r}Q${xb},${y0} ${xb},${y0 + r}V${y1 - r}Q${xb},${y1} ${xb + r},${y1}H${xa}Z`;
}
function barChart(host, rows, refs) {
  host.replaceChildren();
  if (!rows.length) { host.append(el("div", { class: "empty" }, "No finished runs yet")); return; }
  const rowH = 28, W = Math.max(320, host.clientWidth);
  const m = { l: 160, r: 48, t: 30, b: 24 };
  const H = m.t + m.b + rows.length * rowH;
  const vals = rows.flatMap((r) => [r.value, r.lo, r.hi]).concat(refs.map((r) => r.value)).filter(isNum);
  let x0 = Math.min(0, ...vals), x1 = Math.max(0, ...vals);
  const pad = (x1 - x0) * 0.05 || 0.1; x0 -= pad; x1 += pad;
  const X = linear(x0, x1, m.l, W - m.r);
  const svg = sv("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "out-of-sample Sharpe by algorithm" });
  for (const t of niceTicks(x0, x1, 5)) {
    svg.append(sv("line", { x1: X(t), x2: X(t), y1: m.t - 6, y2: H - m.b, stroke: css("--grid"), "stroke-width": 1 }));
    svg.append(stext(X(t), H - 7, fmtTick(t), { "text-anchor": "middle", class: "axis-label" }));
  }
  svg.append(sv("line", { x1: X(0), x2: X(0), y1: m.t - 6, y2: H - m.b, stroke: css("--axis"), "stroke-width": 1 }));
  const wrap = el("div"); wrap.style.position = "relative";
  const tip = el("div", { class: "tooltip" });
  rows.forEach((r, i) => {
    const yc = m.t + i * rowH + rowH / 2;
    svg.append(stext(m.l - 8, yc + 4, r.label, { "text-anchor": "end", fill: css("--text-secondary"), "font-size": 12 }));
    const bar = sv("path", { d: barPath(X(0), X(r.value), yc, 14, 4), fill: css("--s1") });
    svg.append(bar);
    if (isNum(r.lo) && isNum(r.hi) && r.hi > r.lo) {
      const wc = css("--text-secondary");
      svg.append(sv("line", { x1: X(r.lo), x2: X(r.hi), y1: yc, y2: yc, stroke: wc, "stroke-width": 1 }));
      svg.append(sv("line", { x1: X(r.lo), x2: X(r.lo), y1: yc - 4, y2: yc + 4, stroke: wc, "stroke-width": 1 }));
      svg.append(sv("line", { x1: X(r.hi), x2: X(r.hi), y1: yc - 4, y2: yc + 4, stroke: wc, "stroke-width": 1 }));
    }
    const endX = Math.max(X(r.value), isNum(r.hi) ? X(r.hi) : -1e9, X(0));
    svg.append(stext(endX + 6, yc + 4, f2(r.value), { fill: css("--text-primary"), "font-size": 12, "font-weight": 600 }));
    const hit = sv("rect", { x: 0, y: yc - rowH / 2, width: W, height: rowH, fill: "transparent" });
    hit.addEventListener("pointermove", (ev) => {
      bar.setAttribute("opacity", "0.8");
      tip.replaceChildren(el("div", { class: "t-head" }, r.label));
      const rowsT = [["mean Sharpe", f2(r.value)], ["worst / best seed", `${f2(r.lo)} / ${f2(r.hi)}`]].concat(r.extra || []);
      for (const [n, v] of rowsT) { const row = el("div", { class: "row" }); row.append(el("span", { class: "v" }, v), el("span", { class: "n" }, n)); tip.append(row); }
      tip.style.display = "block";
      const rr = svg.getBoundingClientRect();
      const lx = ev.clientX - rr.left + 14;
      tip.style.left = (lx + tip.offsetWidth > W ? lx - 28 - tip.offsetWidth : lx) + "px";
      tip.style.top = yc + 10 + "px";
    });
    hit.addEventListener("pointerleave", () => { bar.removeAttribute("opacity"); tip.style.display = "none"; });
    svg.append(hit);
  });
  for (const ref of refs) {
    if (!isNum(ref.value)) continue;
    svg.append(sv("line", { x1: X(ref.value), x2: X(ref.value), y1: m.t - 10, y2: H - m.b, stroke: css("--ref"), "stroke-width": 1 }));
    svg.append(stext(X(ref.value), m.t - 14, `${ref.label} ${f2(ref.value)}`, { "text-anchor": "middle", fill: css("--text-secondary"), "font-size": 11 }));
  }
  wrap.append(svg, tip);
  host.append(wrap);
}

// ------------------------------------------------------------------------------------------ heatmap
function hexRgb(h) { h = h.replace("#", ""); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); }
function mix(a, b, t) { const A = hexRgb(a), B = hexRgb(b); return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(",")})`; }
function divColor(v, L) {
  if (!isNum(v)) return css("--surface-1");
  const t = Math.max(-1, Math.min(1, v / L));
  return t >= 0 ? mix(css("--div-mid"), css("--div-pos"), t) : mix(css("--div-mid"), css("--div-neg"), -t);
}
function foldLabel(f) {
  const fd = S.summary && S.summary.fold_dates && S.summary.fold_dates[f];
  if (fd) return `${fd[0].slice(0, 4)}–${fd[1].slice(2, 4)}`;
  const m = /_(\d{4})$/.exec(f || "");
  return m ? `${m[1]}–${String(+m[1] + 1).slice(2)}` : f;
}
function heatmap(host, data) {
  host.replaceChildren();
  if (!data || !data.labels || !data.labels.length) { host.append(el("div", { class: "empty" }, "No finished runs yet")); return; }
  const W = Math.max(320, host.clientWidth), cellH = 24, gap = 2;
  const m = { l: 160, r: 8, t: 22, b: 4 };
  const nc = data.folds.length, cw = (W - m.l - m.r) / nc;
  const H = m.t + m.b + data.labels.length * (cellH + gap);
  const all = data.matrix.flat().filter(isNum);
  const L = Math.min(2, Math.max(0.5, ...all.map(Math.abs)));
  const svg = sv("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "test Sharpe by fold" });
  data.folds.forEach((f, j) => svg.append(stext(m.l + j * cw + cw / 2, m.t - 8, foldLabel(f), { "text-anchor": "middle", class: "axis-label" })));
  const wrap = el("div"); wrap.style.position = "relative";
  const tip = el("div", { class: "tooltip" });
  data.labels.forEach((lab, i) => {
    const y = m.t + i * (cellH + gap);
    svg.append(stext(m.l - 8, y + cellH / 2 + 4, lab, { "text-anchor": "end", fill: css("--text-secondary"), "font-size": 12 }));
    data.folds.forEach((f, j) => {
      const v = data.matrix[i][j];
      const c = sv("rect", { x: m.l + j * cw + gap / 2, y, width: cw - gap, height: cellH, rx: 3, fill: divColor(v, L) });
      c.addEventListener("pointermove", (ev) => {
        c.setAttribute("stroke", css("--text-primary")); c.setAttribute("stroke-width", "1");
        tip.replaceChildren(el("div", { class: "t-head" }, `${lab} · test ${foldLabel(f)}`));
        const row = el("div", { class: "row" }); row.append(el("span", { class: "v" }, f2(v)), el("span", { class: "n" }, "Sharpe"));
        tip.append(row);
        tip.style.display = "block";
        const rr = svg.getBoundingClientRect();
        const lx = ev.clientX - rr.left + 14;
        tip.style.left = (lx + tip.offsetWidth > W ? lx - 28 - tip.offsetWidth : lx) + "px";
        tip.style.top = y + cellH + 4 + "px";
      });
      c.addEventListener("pointerleave", () => { c.removeAttribute("stroke"); tip.style.display = "none"; });
      svg.append(c);
    });
  });
  wrap.append(svg, tip);
  const sc = el("div", { class: "scale" });
  const bar = el("span", { class: "bar" });
  bar.style.background = `linear-gradient(90deg, ${css("--div-neg")}, ${css("--div-mid")}, ${css("--div-pos")})`;
  sc.append(el("span", {}, `−${L.toFixed(1)}`), bar, el("span", {}, `+${L.toFixed(1)}`), el("span", {}, "Sharpe (0 = gray)"));
  host.append(wrap, sc);
}

// --------------------------------------------------------------------------------------- rendering
function filtered() {
  const fa = $("#f-algo").value, ff = $("#f-fold").value, ft = $("#f-tag").value, fs = $("#f-state").value;
  return S.runs.filter((r) => (!fa || r.algo === fa) && (!ff || r.fold === ff) && (!ft || (r.tag || "") === ft) && (!fs || r.state === fs));
}
function setOptions(sel, values, labelAll) {
  const cur = sel.value;
  sel.replaceChildren();
  if (labelAll) sel.append(el("option", { value: "" }, labelAll));
  for (const v of values) sel.append(el("option", { value: v }, v === "" ? "(none)" : v));
  if ([...sel.options].some((o) => o.value === cur)) sel.value = cur;
}
function uniq(a) { return [...new Set(a)].sort(); }

function renderFilters() {
  setOptions($("#f-algo"), uniq(S.runs.map((r) => r.algo).filter(Boolean)), "All");
  setOptions($("#f-fold"), uniq(S.runs.map((r) => r.fold).filter(Boolean)), "All");
  setOptions($("#f-tag"), uniq(S.runs.map((r) => r.tag || "")), "All");
  setOptions($("#f-state"), uniq(S.runs.map((r) => r.state).filter(Boolean)), "All");
}

function benchSharpe() {
  const b = S.baselines.find((r) => r.strategy === "buy_hold") || S.baselines.find((r) => r.strategy === "equal_weight");
  return b ? b.sharpe : null;
}

function renderTiles() {
  const runs = filtered();
  const done = runs.filter((r) => r.state === "done");
  const running = runs.filter((r) => r.state === "running" || r.state === "starting").length;
  const failed = runs.filter((r) => r.state === "failed").length;
  const ts = done.map((r) => r.test_sharpe).filter(isNum).sort((a, b) => a - b);
  const med = ts.length ? ts[Math.floor(ts.length / 2)] : null;
  const best = done.filter((r) => isNum(r.test_sharpe)).sort((a, b) => b.test_sharpe - a.test_sharpe)[0];
  const beat = ts.length && isNum(benchSharpe()) ? ts.filter((v) => v > benchSharpe()).length : null;
  const tiles = [
    ["Runs", `${done.length} / ${runs.length}`, `${running} running · ${failed} failed`],
    ["Median test Sharpe (per run)", f2(med), "one 2-year test block per run"],
    ["Best single run", best ? f2(best.test_sharpe) : "–", best ? best.run : ""],
    ["Buy & hold, stitched OOS", f2(benchSharpe()), beat != null ? `${beat} of ${ts.length} runs above it (block vs full OOS)` : "run the baseline suite"],
  ];
  const host = $("#tiles");
  host.replaceChildren();
  for (const [l, v, n] of tiles) {
    const t = el("div", { class: "tile" });
    t.append(el("div", { class: "label" }, l), el("div", { class: "value" }, v), el("div", { class: "note" }, n));
    host.append(t);
  }
}

function renderSummary() {
  if (S.summary === null && S.runs.some((r) => r.state === "done")) {
    for (const id of ["#bars", "#heat"]) $(id).replaceChildren(el("div", { class: "empty" }, "Loading summary…"));
    return;
  }
  const sm = S.summary || {};
  const tab = (sm.table || []).slice().sort((a, b) => b.sharpe - a.sharpe);
  const rows = tab.map((r) => ({
    label: r.label, value: r.sharpe, lo: r.ci_lo, hi: r.ci_hi,
    extra: [["seeds", String(r.seeds)], ["seed range", `${f2(r.seed_min)} to ${f2(r.seed_max)}`], ["CAGR", pct(r.cagr)],
            ["max drawdown", pct(r.max_drawdown)], ["turnover / yr", f2(r.turnover, 1)], ["avg gross", f2(r.avg_gross)]],
  }));
  const refs = [];
  const b = S.baselines.find((r) => r.strategy === "buy_hold") || S.baselines.find((r) => r.strategy === "equal_weight");
  if (b) refs.push({ label: b.strategy === "buy_hold" ? "Buy & hold" : "Equal weight", value: b.sharpe });
  withView($("#bars"), () => barChart($("#bars"), rows, refs), () => table(
    ["algorithm / variant", "seeds", "ensemble Sharpe", "CI low", "CI high", "worst seed", "best seed", "CAGR", "max DD",
     "turnover/yr", "avg gross"],
    tab.map((r) => [r.label, String(r.seeds), f2(r.sharpe), f2(r.ci_lo), f2(r.ci_hi), f2(r.seed_min), f2(r.seed_max),
                    pct(r.cagr), pct(r.max_drawdown), f2(r.turnover, 1), f2(r.avg_gross)]),
    new Set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])));
  withView($("#heat"), () => heatmap($("#heat"), sm), () => table(
    ["algorithm / variant", ...(sm.folds || []).map(foldLabel)],
    (sm.labels || []).map((l, i) => [l, ...sm.matrix[i].map((v) => f2(v))]),
    new Set((sm.folds || []).map((_, j) => j + 1))));
}

const COLS = [
  ["", "sel"], ["run", "run"], ["algo", "algo"], ["fold", "fold"], ["seed", "seed"], ["variant", "tag"], ["state", "state"],
  ["progress", "step"], ["best val Sharpe", "best_val_sharpe"], ["test Sharpe", "test_sharpe"], ["test CAGR", "test_cagr"],
  ["test max DD", "test_max_drawdown"], ["turnover/yr", "test_turnover"], ["avg gross", "test_avg_gross"],
];
const NUMK = new Set(["seed", "step", "best_val_sharpe", "test_sharpe", "test_cagr", "test_max_drawdown", "test_turnover", "test_avg_gross"]);

function renderRuns() {
  const runs = filtered().slice().sort((a, b) => {
    const k = S.sortKey, va = a[k], vb = b[k];
    if (isNum(va) && isNum(vb)) return (va - vb) * S.sortDir;
    return String(va ?? "").localeCompare(String(vb ?? "")) * S.sortDir;
  });
  const t = el("table");
  const hr = el("tr");
  for (const [name, key] of COLS) {
    const th = el("th", { class: NUMK.has(key) ? "num" : "" }, name + (S.sortKey === key ? (S.sortDir > 0 ? " ↑" : " ↓") : ""));
    if (key !== "sel") th.onclick = () => { S.sortDir = S.sortKey === key ? -S.sortDir : (NUMK.has(key) ? -1 : 1); S.sortKey = key; renderRuns(); };
    hr.append(th);
  }
  const thead = el("thead"); thead.append(hr);
  const tb = el("tbody");
  for (const r of runs) {
    const tr = el("tr", { class: S.selected.has(r.run) ? "sel" : "" });
    const td0 = el("td");
    const cb = el("input", { type: "checkbox", "aria-label": `select ${r.run}` });
    cb.checked = S.selected.has(r.run);
    cb.onchange = () => toggleSelect(r.run, cb.checked);
    td0.append(cb);
    if (S.selected.has(r.run)) { const sw = el("span", { class: "swatch" }); sw.style.background = css(SLOTS[S.selected.get(r.run)]); td0.append(sw); }
    tr.append(td0);
    const icon = r.state === "done" ? "✓" : r.state === "failed" ? "✕" : r.state === "running" ? "●" : "○";
    const st = el("span", { class: `state ${r.state}` }); st.append(el("span", { class: "ic" }, icon), document.createTextNode(r.state || ""));
    if (r.error) st.title = r.error;
    const prog = r.state === "done" ? "done" : isNum(r.step) && isNum(r.total_steps) ? `${Math.round((100 * r.step) / r.total_steps)}%` : "–";
    const cells = [r.run, r.algo, r.fold, String(r.seed ?? ""), r.tag || "", st, prog, f2(r.best_val_sharpe), f2(r.test_sharpe),
                   pct(r.test_cagr), pct(r.test_max_drawdown), f2(r.test_turnover, 1), f2(r.test_avg_gross)];
    cells.forEach((c, i) => {
      const key = COLS[i + 1][1];
      const td = el("td", { class: NUMK.has(key) ? "num" : "" });
      if (c instanceof Node) td.append(c); else td.textContent = c;
      tr.append(td);
    });
    tb.append(tr);
  }
  t.append(thead, tb);
  $("#runs").replaceChildren(t);
  $("#sel-hint").textContent = `${S.selected.size} of 8 selected · ${runs.length} runs shown`;
}

function toggleSelect(run, on) {
  if (on) {
    if (S.selected.size >= 8) { renderRuns(); return; }
    const used = new Set(S.selected.values());
    let slot = 0; while (used.has(slot)) slot++;
    S.selected.set(run, slot);
  } else {
    S.selected.delete(run);
  }
  renderRuns();
  refreshSelected(true);
}

async function refreshSelected(force = false) {
  const jobs = [];
  for (const run of S.selected.keys()) {
    const r = S.runs.find((x) => x.run === run);
    const live = r && (r.state === "running" || r.state === "starting");
    if (force || live || !S.progress[run]) jobs.push(api(`/api/progress?exp=${encodeURIComponent(S.exp)}&run=${encodeURIComponent(run)}`).then((p) => { S.progress[run] = p; }));
    if (r && r.state === "done" && !S.ledgers[run]) jobs.push(api(`/api/ledger?exp=${encodeURIComponent(S.exp)}&run=${encodeURIComponent(run)}`).then((l) => { S.ledgers[run] = l; }));
  }
  await Promise.all(jobs).catch(() => {});
  renderCurves();
  await renderEquity();
}

const METRIC_LABELS = {
  val_sharpe: "validation Sharpe", best_val_sharpe: "best validation Sharpe so far", ep_log_return: "training episode log return",
  ep_sharpe: "training episode Sharpe", ep_gross: "training avg gross exposure", ep_turnover: "training turnover / yr",
  ep_cost: "training cost per episode", val_turnover: "validation turnover / yr", val_avg_gross: "validation avg gross",
  val_max_drawdown: "validation max drawdown", pg_loss: "policy loss", v_loss: "value loss", entropy: "policy entropy",
  policy_std: "policy std", kl: "approx KL", explained_var: "explained variance", loss: "TD loss", q_mean: "mean Q",
  critic_loss: "critic loss", actor_loss: "actor loss", alpha: "SAC temperature", train_sharpe: "RRL train Sharpe",
  u_recurrence: "RRL recurrence weight", eps: "epsilon", visited_states: "visited states",
};

function renderCurves() {
  const host = $("#curves");
  const keys = new Set();
  for (const run of S.selected.keys()) for (const row of S.progress[run] || []) for (const [k, v] of Object.entries(row)) if (isNum(v) && !["step", "wall", "episodes"].includes(k)) keys.add(k);
  const sel = $("#metric");
  const opts = [...keys].sort();
  const cur = S.metric && keys.has(S.metric) ? S.metric : keys.has("val_sharpe") ? "val_sharpe" : opts[0];
  sel.replaceChildren(...opts.map((k) => el("option", { value: k }, METRIC_LABELS[k] ? `${METRIC_LABELS[k]} (${k})` : k)));
  if (cur) sel.value = cur;
  S.metric = cur;
  const series = [...S.selected.entries()].map(([run, slot]) => ({
    name: run, color: css(SLOTS[slot]),
    pts: (S.progress[run] || []).filter((r) => isNum(r[cur]) && isNum(r.step)).map((r) => [r.step, r[cur]]).sort((a, b) => a[0] - b[0]),
  }));
  withView(host, () => lineChart(host, series, { xName: "step", empty: S.selected.size ? "No logged values for this metric yet" : "Select runs in the table", aria: cur }),
    () => table(["run", "step", cur || "value"], series.flatMap((s) => s.pts.map((p) => [s.name, String(p[0]), f2(p[1], 4)])), new Set([1, 2])));
}

async function renderEquity() {
  const host = $("#equity");
  const series = [];
  const folds = new Set();
  for (const [run, slot] of S.selected.entries()) {
    const l = S.ledgers[run];
    const r = S.runs.find((x) => x.run === run);
    if (!l || !l.dates) continue;
    if (r) folds.add(r.fold);
    series.push({ name: run, color: css(SLOTS[slot]), pts: l.dates.map((d, i) => [Date.parse(d), l.equity[i]]) });
  }
  const uni = (S.summary && S.summary.universe) || (S.runs[0] && S.runs[0].universe);
  if (uni && folds.size) {
    const strat = ["etf9", "fut4"].includes(uni) ? "equal_weight" : "buy_hold";
    const key = `${uni}|${strat}`;
    if (!S.bench[key]) S.bench[key] = await api(`/api/baseline?universe=${encodeURIComponent(uni)}&strategy=${strat}`).catch(() => ({}));
    const b = S.bench[key] || {};
    for (const f of folds) if (b[f]) series.unshift({ name: `${strat === "buy_hold" ? "buy & hold" : "equal weight"} ${foldLabel(f)}`, color: css("--muted"), ref: true, pts: b[f].dates.map((d, i) => [Date.parse(d), b[f].equity[i]]) });
  }
  withView(host, () => lineChart(host, series, { dates: true, base: 1, fmt: (v) => v.toFixed(3), empty: S.selected.size ? "Selected runs have no test ledger yet (still training)" : "Select runs in the table", aria: "test equity" }),
    () => table(["series", "start", "end", "final equity"], series.map((s) => [s.name, fmtDate(s.pts[0][0], true), fmtDate(s.pts[s.pts.length - 1][0], true), f2(s.pts[s.pts.length - 1][1], 3)]), new Set([3])));
}

// ------------------------------------------------------------------------------------------- data
async function loadExperiments() {
  const exps = await api("/api/experiments");
  const sel = $("#f-exp");
  const cur = S.exp || sel.value;
  sel.replaceChildren(...exps.map((e) => {
    const running = (e.states.running || 0) + (e.states.starting || 0);
    return el("option", { value: e.name }, `${e.name} (${e.runs}${running ? `, ${running} running` : ""})`);
  }));
  if (exps.some((e) => e.name === cur)) sel.value = cur;
  else if (exps.length) sel.value = exps[0].name;
  S.exp = sel.value || null;
}

async function loadExperimentData(full = false) {
  if (!S.exp) return;
  const exp = S.exp;
  S.runs = await api(`/api/runs?exp=${encodeURIComponent(exp)}`);
  const evalCount = S.runs.filter((r) => r.state === "done").length;
  if (full || evalCount !== S.lastEvalCount) {
    S.lastEvalCount = evalCount;
    // the summary aggregates every test ledger and can be slow while training saturates the CPU:
    // render the runs table now and fill the summary charts in when it arrives
    const uni = S.runs[0] && S.runs[0].universe;
    Promise.all([
      api(`/api/summary?exp=${encodeURIComponent(exp)}`).catch(() => ({})),
      uni ? api(`/api/baselines?universe=${encodeURIComponent(uni)}`).catch(() => []) : Promise.resolve([]),
    ]).then(([sm, bl]) => {
      if (S.exp !== exp) return;
      S.summary = sm;
      S.baselines = bl;
      renderTiles();
      renderSummary();
    });
  }
}

function renderAll() {
  renderFilters();
  renderTiles();
  renderSummary();
  renderRuns();
}

async function tick() {
  try {
    await loadExperiments();
    await loadExperimentData();
    renderAll();
    await refreshSelected();
  } catch (e) {
    console.error(e);
  }
  const live = S.runs.some((r) => r.state === "running" || r.state === "starting");
  $("#live").classList.toggle("on", live);
  $("#live-text").textContent = live ? "live · refreshing every 5 s" : "idle · refreshing every 30 s";
  clearTimeout(S.timer);
  S.timer = setTimeout(tick, live ? 5000 : 30000);
}

$("#f-exp").addEventListener("change", async () => {
  S.exp = $("#f-exp").value; S.selected.clear(); S.progress = {}; S.ledgers = {}; S.lastEvalCount = -1; S.summary = null;
  await loadExperimentData(true); renderAll(); await refreshSelected(true);
});
for (const id of ["#f-algo", "#f-fold", "#f-tag", "#f-state"]) $(id).addEventListener("change", () => { renderTiles(); renderRuns(); });
$("#metric").addEventListener("change", () => { S.metric = $("#metric").value; renderCurves(); });
$("#theme").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  renderSummary(); renderRuns(); renderCurves(); renderEquity();
});
let resizeT;
window.addEventListener("resize", () => { clearTimeout(resizeT); resizeT = setTimeout(() => { renderSummary(); renderCurves(); renderEquity(); }, 150); });
tick();
