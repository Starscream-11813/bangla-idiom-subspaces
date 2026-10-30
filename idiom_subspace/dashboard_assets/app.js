"use strict";
/*
 * Idiom Observatory front end. Reads the read-only JSON API (or an embedded
 * snapshot written by `dashboard.py --snapshot`) and renders progress, results
 * and logs. Nothing here can change an experiment.
 */
(() => {
  const SNAPSHOT = window.__OBSERVATORY_SNAPSHOT__ || null;
  const POLL_MS = 3000;
  const NS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);
  const COLORS = {MLR: "#0a8272", LEACE: "#df8a44", COV: "#587eaa", CPCA: "#9579a9", LDA: "#bd7264",
                  RAND: "#99a797", PCA: "#a5a251", AMBIENT: "#263f4a"};
  const METHOD_ORDER = ["MLR", "LDA", "CPCA", "COV", "LEACE", "PCA", "RAND"];
  const SPLIT_COLORS = {space: "#0b8274", probe: "#8ab8a0", validation: "#e4ad6d", test: "#587da6"};
  const state = {overview: null, detail: null, selected: null, metric: "R", projection: "orthogonal",
                 busy: false, connected: false, primaryOnly: true, estimatorLayer: null};

  // ---------------------------------------------------------------- helpers
  const finite = (v) => v !== null && v !== "" && v !== undefined && Number.isFinite(Number(v));
  const number = (v) => (finite(v) ? Number(v) : null);
  const fmtInt = (v) => (finite(v) ? Number(v).toLocaleString(undefined, {maximumFractionDigits: 0}) : "—");
  const fmt = (v, digits = 3) => (finite(v) ? Number(v).toFixed(digits) : "—");
  const pretty = (v) => String(v || "waiting").replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
  const color = (name) => COLORS[name] || "#6b9691";
  const source = (row) => row.source_condition ?? row.e ?? "";
  const target = (row) => row.condition ?? row.p ?? "";
  function duration(v) {
    if (!finite(v)) return "—";
    const s = Math.max(0, Math.floor(Number(v)));
    if (s >= 86400) return `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`;
    if (s >= 3600) return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
    if (s >= 60) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
    return `${s}s`;
  }
  function node(tag, attrs = {}, content) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === null) continue;
      if (k === "class") el.className = v; else el.setAttribute(k, v);
    }
    if (content !== undefined) el.textContent = content;
    return el;
  }
  function svg(tag, attrs = {}, content) {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) el.setAttribute(k, String(v));
    if (content !== undefined) el.textContent = content;
    return el;
  }
  function setText(id, value) { const el = $(id); if (el) el.textContent = value; }
  function store(key, value) {
    try {
      if (value === undefined) return localStorage.getItem(key);
      if (value === null) localStorage.removeItem(key); else localStorage.setItem(key, value);
    } catch (_) { /* storage unavailable (private mode, file://) */ }
    return null;
  }
  function empty(host, title, message) {
    host.replaceChildren();
    const box = node("div", {class: "empty-chart"});
    box.append(node("span", {class: "empty-symbol", "aria-hidden": "true"}, "⌁"), node("strong", {}, title), node("p", {}, message));
    host.append(box);
  }

  // ---------------------------------------------------------------- theme
  function applyTheme(theme) {
    if (theme === "dark" || theme === "light") document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
    setText("theme-toggle", theme === "dark" ? "◑ Dark" : theme === "light" ? "◐ Light" : "◐ Auto");
  }
  const urlTheme = new URLSearchParams(location.search).get("theme");
  applyTheme(["dark", "light"].includes(urlTheme) ? urlTheme : store("idiom-observatory-theme"));
  $("theme-toggle").addEventListener("click", () => {
    const order = [null, "dark", "light"];
    const current = store("idiom-observatory-theme");
    const next = order[(order.indexOf(current) + 1) % order.length];
    store("idiom-observatory-theme", next);
    applyTheme(next);
  });

  // ---------------------------------------------------------------- data
  async function getJSON(url) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 12000);
    try {
      const response = await fetch(url, {cache: "no-store", signal: controller.signal});
      if (!response.ok) throw new Error(`Local server returned ${response.status}`);
      return await response.json();
    } finally { clearTimeout(timer); }
  }
  const api = {
    overview: async () => (SNAPSHOT ? SNAPSHOT.overview : getJSON("/api/overview")),
    job: async (id) => (SNAPSHOT ? (SNAPSHOT.details || {})[id] || null : getJSON("/api/job?id=" + encodeURIComponent(id))),
  };

  // ---------------------------------------------------------------- job helpers
  const modelName = (job) => job?.context?.model || job?.model || job?.relative_path?.split("/").pop() || "Unnamed model";
  function jobState(job) {
    if (job?.stale) return "stale";
    if (job?.status === "running" && (job.heartbeat_unknown || job.telemetry_available === false || job.stage === "legacy_run" || job.live === false)) return "unknown";
    return job?.status || "unknown";
  }
  function stageLabel(stage) {
    const labels = {starting: "Preparing experiment", legacy_run: "Legacy run", prepare: "Preparing corpus & splits",
      extract: "Extracting hidden states", cache_validation: "Validating feature cache", preprocess: "Preprocessing layer",
      grid: "Retention / leakage grid", "batched_grid readouts": "Batched grid probes", layer_complete: "Layer saved",
      nulls: "Null draws", "batched_null draws": "Batched null draws", S2: "Language disentanglement",
      S3: "Transfer baselines", S4: "Literal separation & open class", S4_open_class: "Open-class sweep",
      "batched_open-class readouts": "Batched open-class probes", robustness: "Probe robustness",
      lexical_baselines: "Lexical baselines", completed: "Results saved", extracted: "Features cached"};
    return labels[stage] || pretty(stage);
  }
  function layerMilestone(job) {
    const m = job?.milestones?.layer;
    if (m && finite(m.total) && Number(m.total) > 0) return m;
    const done = (job?.completed_layers || []).length;
    return done ? {done, total: null} : null;
  }
  function jobFraction(job) {
    // Layer grids are most of a job; inference and later studies make up the rest,
    // so an unfinished job never shows 100%.
    if (!job) return null;
    if (job.status === "completed") return 1;
    const m = layerMilestone(job);
    if (m && finite(m.total)) {
      const inner = job.stage === "grid" && finite(job.fraction) ? Number(job.fraction) : 0;
      return Math.min(.9, (.9 * (Number(m.done) + inner)) / Number(m.total));
    }
    return finite(job.fraction) ? Math.min(.95, Number(job.fraction)) : null;
  }

  // ---------------------------------------------------------------- sidebar + whole run
  function renderSidebar() {
    const ov = state.overview || {};
    const jobs = ov.jobs || [];
    setText("job-count", jobs.length);
    setText("root-path", ov.root || "No workspace connected");
    $("root-path").title = ov.root || "";
    $("no-jobs").hidden = jobs.length > 0;
    const list = $("job-list");
    const scrollTop = list.scrollTop;
    list.replaceChildren();
    if (!jobs.length) { list.append(node("p", {class: "sidebar-empty"}, "No model jobs found yet.")); return; }
    for (const job of jobs) {
      const status = jobState(job);
      const button = node("button", {type: "button", class: "job-card" + (job.id === state.selected ? " selected" : ""),
        "aria-pressed": String(job.id === state.selected), "data-job-id": job.id, title: job.relative_path || modelName(job)});
      const top = node("div", {class: "job-card-top"});
      top.append(node("span", {class: "job-dot " + status, "aria-hidden": "true"}), node("span", {class: "job-card-name"}, modelName(job)));
      const ctx = job.context || {};
      const meta = [ctx.seed !== undefined ? `split ${ctx.seed}` : null, ctx.label_seed !== undefined ? `taxonomy ${ctx.label_seed}` : null, pretty(status)];
      const m = layerMilestone(job);
      if (m && finite(m.total)) meta.push(`layer ${m.done}/${m.total}`);
      button.append(top, node("div", {class: "job-card-meta"}, meta.filter(Boolean).join(" · ")),
                    node("div", {class: "job-card-stage"}, stageLabel(job.stage)));
      const fraction = jobFraction(job);
      const mini = node("div", {class: "job-mini"}); const fill = node("div");
      fill.style.width = fraction === null ? "0%" : `${100 * fraction}%`;
      mini.append(fill); button.append(mini);
      button.addEventListener("click", () => {
        if (state.selected === job.id) return;
        state.selected = job.id; state.detail = null; state.estimatorLayer = null;
        store("idiom-observatory-job", job.id);
        renderSidebar(); renderDetail();
        if (!state.busy) refresh();
      });
      list.append(button);
    }
    list.scrollTop = scrollTop;
  }

  function renderRunBand() {
    const ov = state.overview || {};
    const jobs = ov.jobs || [];
    const summary = ov.summary || {};
    const counts = {running: 0, completed: 0, failed: 0, stale: 0, unknown: 0};
    let partial = 0;
    for (const job of jobs) {
      const status = jobState(job);
      if (status === "completed") counts.completed += 1;
      else if (status === "failed" || status === "interrupted") counts.failed += 1;
      else if (status === "stale") counts.stale += 1;
      else if (status === "running") counts.running += 1;
      else counts.unknown += 1;
      if (status !== "completed") partial += jobFraction(job) || 0;
    }
    const planned = Math.max(Number(summary.planned_jobs) || 0, jobs.length);
    const fraction = planned ? Math.min(1, (counts.completed + partial) / planned) : null;
    $("run-fill").style.width = fraction === null ? "0%" : `${100 * fraction}%`;
    setText("run-count", planned ? `${counts.completed} of ${planned} jobs complete · ${Math.round(100 * (fraction || 0))}% (layer-weighted)` : "No jobs yet");
    const chips = $("run-chips");
    chips.replaceChildren();
    for (const [name, value] of [["planned", planned], ["running", counts.running], ["completed", counts.completed],
                                 ["failed", counts.failed], ["stale", counts.stale], ["queued", summary.queued]]) {
      if (!value && name !== "planned") continue;
      const chip = node("span", {class: "run-chip " + name}, pretty(name));
      chip.append(node("strong", {}, String(value || 0)));
      chips.append(chip);
    }
    const eta = $("run-eta");
    if (finite(summary.longest_running_layer_eta_seconds) && Number(summary.longest_running_layer_eta_seconds) > 0) {
      eta.hidden = false;
      eta.textContent = `≈ ${duration(summary.longest_running_layer_eta_seconds)} left in running layer grids`;
      eta.title = "Estimate from measured per-layer durations of the running jobs; inference and later studies are not included.";
    } else eta.hidden = true;
    const queue = $("queue");
    queue.replaceChildren();
    for (const launcher of ov.launchers || []) {
      const finished = new Map((launcher.finished || []).map((f) => [f.model, f]));
      const running = new Set((launcher.running || []).map((r) => r.model));
      for (const model of launcher.planned || []) {
        const done = finished.get(model);
        const cls = done ? (done.exit_code === 0 ? "done" : "failed") : running.has(model) ? "running" : "";
        const label = running.has(model) ? `▶ ${model}` : done ? `${done.exit_code === 0 ? "✓" : "✗"} ${model}` : `· ${model}`;
        queue.append(node("span", {class: cls, title: "GPU launcher queue"}, label));
      }
    }
  }

  // ---------------------------------------------------------------- selected job
  function renderDetail() {
    const data = state.detail;
    const job = data?.job;
    const ctx = job?.context || {};
    setText("model-name", job ? modelName(job) : "Waiting for an experiment");
    const chips = $("experiment-context");
    chips.replaceChildren();
    const analysis = job?.analysis || {};
    for (const value of [ctx.seed !== undefined ? `Split seed ${ctx.seed}` : null, ctx.label_seed !== undefined ? `Taxonomy seed ${ctx.label_seed}` : null,
                         analysis.split_mode ? `${pretty(analysis.split_mode)} split` : null,
                         analysis.probe_backend ? `Probes: ${analysis.probe_backend}` : null,
                         analysis.ambient_space ? `Ambient: ${analysis.ambient_space}` : null, job?.relative_path]) {
      if (value) chips.append(node("span", {class: "context-chip"}, value));
    }
    const status = job ? jobState(job) : "neutral";
    $("job-status").className = "status-pill " + status;
    setText("job-status", job ? pretty(status) : "NO JOB SELECTED");

    // stage card
    setText("stage-name", job ? stageLabel(job.stage) : "Waiting");
    const done = number(job?.completed), total = number(job?.total);
    const fraction = total > 0 && done !== null ? Math.min(1, Math.max(0, done / total)) : null;
    $("stage-fill").style.width = fraction === null ? "0%" : `${fraction * 100}%`;
    setText("stage-count", total > 0 ? `${fmtInt(done)} / ${fmtInt(total)} ${job.unit || "tasks"}` : done ? `${fmtInt(done)} ${job.unit || "tasks"}` : "Count not available");
    setText("stage-rate", finite(job?.rate_per_second) ? `${Number(job.rate_per_second).toFixed(2)} ${job.unit || "tasks"}/s` : "stage only");
    setText("stage-eta", finite(job?.eta_seconds) ? `≈ ${duration(job.eta_seconds)} left in this stage (throughput estimate)` : "No estimate yet");

    // layers card
    const m = layerMilestone(job);
    if (m && finite(m.total)) {
      setText("layer-count", `${m.done} / ${m.total}`);
      $("layer-fill").style.width = `${100 * Math.min(1, Number(m.done) / Number(m.total))}%`;
      const parts = [];
      if (finite(m.mean_seconds)) parts.push(`${duration(m.mean_seconds)} per layer`);
      if (finite(m.eta_seconds) && Number(m.done) < Number(m.total)) parts.push(`≈ ${duration(m.eta_seconds)} for remaining layers`);
      setText("layer-eta", parts.join(" · ") || "Timing after the first layer");
    } else {
      setText("layer-count", job ? String((job.completed_layers || []).length) : "—");
      $("layer-fill").style.width = "0%";
      setText("layer-eta", (job?.completed_layers || []).length ? "Layers " + job.completed_layers.join(", ") : "Layer timing pending");
    }
    setText("elapsed", duration(job?.elapsed_seconds));
    setText("stage-elapsed", job && finite(job.stage_elapsed_seconds) ? `${duration(job.stage_elapsed_seconds)} in this stage` : "No timing data yet");
    const pool = job?.pooling || {};
    setText("valid-count", pool.items ? fmtInt(pool.valid) : "—");
    setText("valid-note", pool.items ? `${fmtInt(pool.items)} total · ${fmtInt(pool.invalid)} excluded (absent/ambiguous/truncated spans)` : "Tokenization audit pending");
    setText("stage-message", job?.message || "Progress comes from the running process. Completed files remain available after a run stops.");
    const age = number(job?.heartbeat_age_seconds);
    setText("heartbeat", job?.heartbeat_unknown || job?.telemetry_available === false || job?.stage === "legacy_run" ? "Live telemetry unavailable"
      : job?.stale ? `Heartbeat stale · ${duration(age)} ago` : age !== null ? `Heartbeat ${duration(age)} ago` : "No heartbeat yet");
    const task = $("active-task");
    task.replaceChildren();
    for (const [key, value] of Object.entries(ctx)) {
      if (["model", "seed", "label_seed", "split_mode", "kind", "cache_path", "mode", "cpu_threads"].includes(key) || value === null || value === undefined) continue;
      task.append(node("span", {}, `${pretty(key)}: ${String(value)}`));
    }
    renderPipeline(job);
    renderLayerStrip(job);
    renderFilters();
    renderCharts();
    renderPopulation();
    renderTiming(job);
    renderInference();
    renderBaselines();
    renderContrasts();
    renderSelection(job);
    renderLog(data);
  }

  function renderPipeline(job) {
    const host = $("pipeline");
    host.replaceChildren();
    const canvas = svg("svg", {viewBox: "0 0 880 102", role: "img", "aria-label": "Extract representations, estimate on SPACE, fit probes on PROBE, select with VALIDATION, evaluate on TEST"});
    const stages = [
      {title: "Represent", sub: "EXACT-SPAN FEATURES", pattern: /extract|feature|token|cache|prepare/},
      {title: "Estimate", sub: "SPACE", pattern: /preprocess|subspace|estimat/},
      {title: "Fit probes", sub: "PROBE", pattern: /grid|probe|fit|batched_grid/},
      {title: "Select", sub: "VALIDATION", pattern: /select|valid|layer_complete/},
      {title: "Evaluate", sub: "TEST", pattern: /inference|null|bootstrap|contrast|analysis|result|complete|S[2-5]|robust|open/i},
    ];
    let active = job?.status === "completed" ? 4 : stages.findIndex((s) => s.pattern.test(job?.stage || ""));
    if (active < 0 && job) active = 0;
    stages.forEach((stage, i) => {
      const x = i * 177 + 1;
      const on = i === active ? " active" : "";
      if (i < 4) canvas.append(svg("path", {d: `M${x + 149} 39h22m-5-4 5 4-5 4`, class: "pipe-arrow", "stroke-width": 1.4}));
      canvas.append(svg("rect", {x, y: 12, width: 149, height: 58, rx: 8, class: "pipe-box" + on}));
      canvas.append(svg("circle", {cx: x + 20, cy: 33, r: 8, class: "pipe-dot" + on}));
      canvas.append(svg("text", {x: x + 20, y: 36, "text-anchor": "middle", class: "pipe-num" + on}, i + 1));
      canvas.append(svg("text", {x: x + 36, y: 37, class: "pipe-title" + on}, stage.title));
      canvas.append(svg("text", {x: x + 20, y: 56, class: "pipe-sub"}, stage.sub));
    });
    host.append(canvas);
  }

  function knownLayers() {
    const grid = state.detail?.grid || [];
    const layers = new Set(grid.map((r) => r.layer).filter(finite).map(Number));
    for (const l of state.detail?.job?.completed_layers || []) layers.add(Number(l));
    return [...layers].sort((a, b) => a - b);
  }

  function renderLayerStrip(job) {
    const host = $("layer-strip");
    host.replaceChildren();
    const done = new Set((job?.completed_layers || []).map(Number));
    const m = layerMilestone(job);
    const current = number(job?.context?.layer);
    const selected = number(job?.selection?.layer);
    const layers = [...done];
    if (current !== null && !done.has(current)) layers.push(current);
    layers.sort((a, b) => a - b);
    for (const layer of layers) {
      const cls = done.has(layer) ? "done" : "current";
      host.append(node("span", {class: `layer-cell ${cls}${layer === selected ? " selected" : ""}`,
                                title: `Layer ${layer}${layer === selected ? " · validation-selected" : ""}`}, String(layer)));
    }
    const remaining = m && finite(m.total) ? Number(m.total) - layers.length : 0;
    for (let i = 0; i < remaining; i += 1) host.append(node("span", {class: "layer-cell", title: "Pending"}, "·"));
    if (!host.children.length) host.append(node("span", {class: "muted compact"}, "No layer results yet"));
  }

  // ---------------------------------------------------------------- filters
  function updateSelect(id, values, preferred, allLabel) {
    const el = $(id);
    const previous = el.value;
    const list = [...new Set(values.filter((v) => v !== null && v !== undefined && String(v) !== "").map(String))];
    if (id === "filter-rank" || id === "estimator-layer") list.sort((a, b) => Number(a) - Number(b)); else list.sort();
    const signature = JSON.stringify([list, allLabel]);
    if (el.dataset.options !== signature) {
      el.dataset.options = signature;
      el.replaceChildren();
      if (allLabel) el.append(node("option", {value: "*"}, allLabel));
      for (const value of list) el.append(node("option", {value}, value));
      if (!list.length && !allLabel) el.append(node("option", {value: ""}, "—"));
    }
    if ([...el.options].some((o) => o.value === previous)) el.value = previous;
    else if ([...el.options].some((o) => o.value === String(preferred))) el.value = String(preferred);
    else if (allLabel) el.value = "*";
    else el.value = list[0] || "";
  }

  function renderFilters() {
    const d = state.detail || {};
    const rows = [...(d.grid || []), ...(d.ssi || []), ...(d.inference || []), ...(d.openclass || [])].filter((r) => r.method !== "AMBIENT");
    updateSelect("filter-concept", rows.map((r) => r.concept), "F");
    const conceptRows = rows.filter((r) => r.concept === $("filter-concept").value);
    updateSelect("filter-method", conceptRows.map((r) => r.method), "*", "All estimators");
    const method = $("filter-method").value;
    const filtered = conceptRows.filter((r) => method === "*" || r.method === method);
    const ranks = filtered.map((r) => r.M).filter(finite);
    const supervised = filtered.filter((r) => !["RAND", "PCA"].includes(r.method)).map((r) => r.M).filter(finite);
    const preferred = supervised.length ? Math.max(...supervised) : ranks.length ? Math.max(...ranks) : "";
    updateSelect("filter-rank", ranks, preferred);
    updateSelect("filter-source", filtered.map(source), "en-idiom");
    updateSelect("filter-target", filtered.map(target), "bn-fig");
    const layers = knownLayers();
    const selected = number(d.job?.selection?.layer);
    updateSelect("estimator-layer", layers, state.estimatorLayer ?? (selected ?? layers[layers.length - 1]));
  }
  const filters = () => ({concept: $("filter-concept").value, method: $("filter-method").value, rank: $("filter-rank").value,
                          source: $("filter-source").value, target: $("filter-target").value});
  function selectedRows(rows, {ignoreSource = false, ignoreRank = false, ignoreMethod = false} = {}) {
    const f = filters();
    return rows.filter((r) => (!r.group || r.group === "all") && r.concept === f.concept
      && (ignoreMethod || f.method === "*" || r.method === f.method) && r.method !== "AMBIENT"
      && (ignoreRank || String(r.M) === f.rank) && (ignoreSource || source(r) === f.source) && target(r) === f.target);
  }

  // ---------------------------------------------------------------- charts
  function legend(id, names) {
    const host = $(id);
    host.replaceChildren();
    for (const name of [...new Set(names)]) {
      const el = node("span", {class: "legend-item"});
      const line = node("span", {class: "legend-line"});
      line.style.backgroundColor = color(name);
      el.append(line, document.createTextNode(name));
      host.append(el);
    }
  }

  function baseChart(host, {width = 600, height = 270, xLabel, yLabel, xMin, xMax, yMin = 0, yMax = 1, xTicks, yTicks, logX = false}) {
    host.replaceChildren();
    const s = svg("svg", {viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `${yLabel} by ${xLabel}`});
    const m = {l: 64, r: 15, t: 15, b: 43};
    const plotW = width - m.l - m.r, plotH = height - m.t - m.b;
    const tx = (v) => (logX ? Math.log10(v) : v);
    const x = (v) => m.l + ((tx(v) - tx(xMin)) / ((tx(xMax) - tx(xMin)) || 1)) * plotW;
    const y = (v) => m.t + ((yMax - v) / ((yMax - yMin) || 1)) * plotH;
    const ticks = yTicks || Array.from({length: 5}, (_, i) => yMin + ((yMax - yMin) * i) / 4);
    for (const v of ticks) {
      s.append(svg("line", {x1: m.l, x2: width - m.r, y1: y(v), y2: y(v), class: "grid-line"}));
      s.append(svg("text", {x: m.l - 10, y: y(v) + 3.5, "text-anchor": "end", class: "axis-text"},
                   Math.abs(v) < 1e-6 ? "0" : Number(v).toLocaleString(undefined, {maximumFractionDigits: 2})));
    }
    for (const v of xTicks || []) s.append(svg("text", {x: x(v), y: height - 23, "text-anchor": "middle", class: "axis-text"},
                                               Number(v).toLocaleString(undefined, {maximumFractionDigits: 2})));
    s.append(svg("line", {x1: m.l, x2: width - m.r, y1: y(yMin), y2: y(yMin), class: "axis-line"}));
    s.append(svg("text", {x: m.l + plotW / 2, y: height - 3, "text-anchor": "middle", class: "axis-label"}, xLabel));
    s.append(svg("text", {transform: `translate(12 ${m.t + plotH / 2}) rotate(-90)`, "text-anchor": "middle", class: "axis-label"}, yLabel));
    host.append(s);
    return {s, x, y, m, plotW, plotH, width, height};
  }

  function point(s, x, y, c, label, r = 3) {
    const p = svg("circle", {cx: x, cy: y, r, fill: c, stroke: "white", "stroke-width": 1.5, tabindex: 0, role: "img", "aria-label": label});
    p.append(svg("title", {}, label));
    s.append(p);
  }

  function meanBy(rows, key, value) {
    const groups = new Map();
    for (const row of rows) {
      if (!finite(row[key]) || !finite(row[value])) continue;
      const k = Number(row[key]);
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(Number(row[value]));
    }
    return [...groups].sort((a, b) => a[0] - b[0]).map(([x, vs]) => ({x, y: vs.reduce((a, b) => a + b, 0) / vs.length, n: vs.length}));
  }

  function line(chart, values, cls, stroke) {
    if (values.length < 1) return;
    const d = values.map((p, i) => `${i ? "L" : "M"}${chart.x(p.x)} ${chart.y(p.y)}`).join(" ");
    chart.s.append(svg("path", {d, class: cls, stroke}));
  }

  function seriesPlot(host, rows, {xKey, yKey, xLabel, yLabel, logX = false, references = [], marker = null}) {
    const valid = rows.filter((r) => finite(r[xKey]) && finite(r[yKey]) && (!logX || r[xKey] > 0));
    if (!valid.length) {
      empty(host, "No saved observations", "This view fills in as matching results are saved. Try another condition, estimator or rank.");
      return [];
    }
    const refValues = references.flatMap((ref) => ref.values);
    const xs = [...new Set([...valid.map((r) => Number(r[xKey])), ...refValues.map((p) => p.x)])].sort((a, b) => a - b);
    let xmin = xs[0], xmax = xs[xs.length - 1];
    if (xmin === xmax) { xmin = logX ? xmin / 1.5 : xmin - .5; xmax = logX ? xmax * 1.5 : xmax + .5; }
    const ys = [...valid.map((r) => Number(r[yKey])), ...refValues.map((p) => p.y)];
    const ymin = yKey === "SSI" ? Math.min(0, Math.floor(Math.min(...ys) * 10) / 10) : 0;
    const ymax = yKey === "SSI" ? Math.max(1, Math.ceil(Math.max(...ys) * 10) / 10) : Math.min(1, Math.max(.2, Math.ceil(Math.max(...ys) * 10) / 10));
    const chart = baseChart(host, {xLabel, yLabel, xMin: xmin, xMax: xmax, yMin: ymin, yMax: ymax,
      xTicks: xs.length <= 12 ? xs : xs.filter((_, i) => i % Math.ceil(xs.length / 8) === 0), logX});
    for (const ref of references) line(chart, ref.values, ref.cls, ref.stroke);
    if (marker !== null && marker >= xmin && marker <= xmax) {
      const mx = chart.x(marker);
      chart.s.append(svg("line", {x1: mx, x2: mx, y1: chart.m.t, y2: chart.y(ymin), stroke: "#df8a44", "stroke-dasharray": "2 3", "stroke-width": 1.2}));
      chart.s.append(svg("text", {x: mx + 4, y: chart.m.t + 10, class: "axis-text", fill: "#df8a44"}, "◆ selected"));
    }
    const groups = new Map();
    for (const row of valid) {
      const key = row.method || "MLR";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(row);
    }
    for (const [name, data] of groups) {
      const values = meanBy(data, xKey, yKey);
      if (values.length > 1) chart.s.append(svg("path", {d: values.map((p, i) => `${i ? "L" : "M"}${chart.x(p.x)} ${chart.y(p.y)}`).join(" "),
                                                          fill: "none", stroke: color(name), "stroke-width": 2, "stroke-linejoin": "round"}));
      for (const p of values) point(chart.s, chart.x(p.x), chart.y(p.y), color(name), `${name}: ${xLabel} ${p.x}, ${yLabel} ${p.y.toFixed(3)}${p.n > 1 ? `, mean of ${p.n}` : ""}`);
    }
    return [...groups.keys()].sort((a, b) => METHOD_ORDER.indexOf(a) - METHOD_ORDER.indexOf(b));
  }

  function renderCharts() {
    const d = state.detail || {};
    const f = filters();
    const selectedLayer = number(d.job?.selection?.layer);
    let rows, references = [];
    if (state.metric === "SSI") {
      rows = selectedRows(d.ssi?.length ? d.ssi : d.inference || []);
    } else {
      rows = selectedRows(d.grid || []);
      const ambient = (d.grid || []).filter((r) => r.method === "AMBIENT" && r.concept === f.concept && target(r) === f.target && (!r.group || r.group === "all"));
      if (state.metric === "R") references.push({values: meanBy(ambient, "layer", "R"), cls: "ref-line"});
      references.push({values: meanBy(rows, "layer", "R0"), cls: "floor-line"});
    }
    const yLabel = {R: "Retention (accuracy)", L: "Leakage (accuracy)", SSI: "Shared-subspace index"}[state.metric];
    const names = seriesPlot($("layer-chart"), rows, {xKey: "layer", yKey: state.metric, xLabel: "Layer", yLabel, references, marker: selectedLayer});
    legend("layer-legend", names);
    setText("layer-description", f.source && f.target ? `${f.concept} · ${f.source} → ${f.target} · rank ${f.rank}` : "Concept recovery across the network");
    setText("layer-footnote", state.metric === "SSI"
      ? "SSI = (R(e→p) − R₀)/(R(p→p) − R₀) at matched rank; undefined values are omitted and values can exceed 1. ◆ validation-selected layer."
      : "Held-out accuracy. Dashed: ambient probe (full space); dotted: majority floor; ◆ validation-selected layer.");
    renderScatter(selectedRows(d.grid || []));
    renderEstimators();
    const open = selectedRows(d.openclass || [], {ignoreSource: true});
    legend("open-legend", seriesPlot($("open-chart"), open, {xKey: "N", yKey: $("open-metric").value, xLabel: "Estimation idiom types N (log scale)",
      yLabel: $("open-metric").value === "R" ? "Retention (accuracy)" : "Leakage (accuracy)", logX: true}));
  }

  function renderScatter(rows) {
    const valid = rows.filter((r) => finite(r.R) && finite(r.L));
    if (!valid.length) { empty($("scatter-chart"), "Containment awaits results", "Retention and leakage are both needed for this view."); return; }
    const c = baseChart($("scatter-chart"), {width: 520, height: 280, xLabel: "Leakage (accuracy) ↓", yLabel: "Retention (accuracy) ↑", xMin: 0, xMax: 1, xTicks: [0, .25, .5, .75, 1]});
    for (const row of valid) point(c.s, c.x(Number(row.L)), c.y(Number(row.R)), color(row.method),
      `${row.method}, layer ${row.layer}, rank ${row.M}: retention ${Number(row.R).toFixed(3)}, leakage ${Number(row.L).toFixed(3)}`, 4.5);
  }

  function renderEstimators() {
    const d = state.detail || {};
    const f = filters();
    const layer = number($("estimator-layer").value);
    const host = $("estimator-chart");
    const rows = (d.grid || []).filter((r) => (!r.group || r.group === "all") && r.concept === f.concept && Number(r.layer) === layer
      && source(r) === f.source && target(r) === f.target && String(r.M) === f.rank && r.method !== "AMBIENT");
    setText("estimator-description", layer !== null ? `${f.concept} · ${f.source} → ${f.target} · rank ${f.rank} · layer ${layer}` : "Retention and leakage by estimator");
    if (!rows.length) { empty(host, "No estimator results at this layer", "Choose a saved layer or a supported rank."); return; }
    const methods = [...new Set(rows.map((r) => r.method))].sort((a, b) => METHOD_ORDER.indexOf(a) - METHOD_ORDER.indexOf(b));
    const ambient = (d.grid || []).find((r) => r.method === "AMBIENT" && r.concept === f.concept && target(r) === f.target && Number(r.layer) === layer && (!r.group || r.group === "all"));
    const values = methods.map((m) => rows.find((r) => r.method === m));
    const ymax = Math.min(1, Math.max(.2, ...values.flatMap((r) => [number(r.R) || 0, number(r.L) || 0]), number(ambient?.R) || 0) + .05);
    const c = baseChart(host, {width: 560, height: 270, xLabel: "Estimator", yLabel: "Accuracy", xMin: 0, xMax: methods.length, yMax: ymax});
    const band = c.plotW / methods.length;
    values.forEach((row, i) => {
      const x0 = c.m.l + i * band + band * .18, w = band * .3;
      for (const [k, key, opacity] of [[0, "R", 1], [1, "L", .38]]) {
        if (!finite(row[key])) continue;
        const v = Number(row[key]);
        const rect = svg("rect", {x: x0 + k * w, y: c.y(v), width: w - 2, height: Math.max(0, c.y(0) - c.y(v)), fill: color(row.method), "fill-opacity": opacity, rx: 2, tabindex: 0});
        rect.append(svg("title", {}, `${row.method} ${key === "R" ? "retention" : "leakage"} ${v.toFixed(3)}`));
        c.s.append(rect);
      }
      c.s.append(svg("text", {x: c.m.l + i * band + band / 2, y: c.height - 23, "text-anchor": "middle", class: "axis-text"}, row.method));
    });
    if (finite(ambient?.R)) {
      const ay = c.y(Number(ambient.R));
      c.s.append(svg("line", {x1: c.m.l, x2: c.width - c.m.r, y1: ay, y2: ay, class: "ref-line"}));
    }
    if (finite(values[0]?.R0)) {
      const fy = c.y(Number(values[0].R0));
      c.s.append(svg("line", {x1: c.m.l, x2: c.width - c.m.r, y1: fy, y2: fy, class: "floor-line"}));
    }
  }

  function renderPopulation() {
    const d = state.detail || {};
    const split = d.split || {};
    const counts = split.counts || {};
    const host = $("split-chart");
    const pairs = ["space", "probe", "validation", "test"].map((k) => [k, Number(counts[k] || 0)]);
    const total = pairs.reduce((sum, p) => sum + p[1], 0);
    $("split-legend").replaceChildren();
    $("data-facts").replaceChildren();
    $("split-warning").hidden = true;
    if (!total) { empty(host, "Split audit not saved yet", "Population and leakage-graph diagnostics appear here when available."); return; }
    host.replaceChildren();
    const heading = node("div", {class: "split-total"}, fmtInt(total));
    heading.append(node("span", {}, "source records retained"));
    const track = node("div", {class: "split-track", role: "img", "aria-label": pairs.map(([k, n]) => `${k}: ${n}`).join(", ")});
    for (const [name, count] of pairs) {
      const segment = node("span", {title: `${pretty(name)}: ${fmtInt(count)}`});
      segment.style.width = `${(100 * count) / total}%`;
      segment.style.backgroundColor = SPLIT_COLORS[name];
      track.append(segment);
      const key = node("div", {class: "split-key"});
      const swatch = node("span", {class: "swatch"});
      swatch.style.backgroundColor = SPLIT_COLORS[name];
      key.append(swatch, document.createTextNode(pretty(name)), node("strong", {}, fmtInt(count)));
      $("split-legend").append(key);
    }
    host.append(heading, track);
    const facts = [["Split mode", split.mode ? pretty(split.mode) : null], ["Components", split.components],
                   ["Largest", finite(split.largest_fraction) ? `${(100 * split.largest_fraction).toFixed(1)}%` : null],
                   ["Identical inputs across splits", split.cross_split_identical_inputs],
                   ["F classes", d.labels?.effective_k], ["Label source", d.labels?.backend]];
    for (const [name, value] of facts) {
      if (value === undefined || value === null) continue;
      const item = node("span", {}, name + " ");
      item.append(node("strong", {}, typeof value === "number" ? fmtInt(value) : String(value)));
      $("data-facts").append(item);
    }
    if (split.coverage_warning || Number(split.largest_fraction) > .2) {
      $("split-warning").hidden = false;
      setText("split-warning", split.coverage_warning || `The largest connected component contains ${(100 * Number(split.largest_fraction)).toFixed(1)}% of records. Interpret test coverage with the split audit.`);
    }
  }

  function renderTiming(job) {
    const host = $("layer-timing");
    const m = job?.milestones?.layer;
    const durations = (m?.durations || []).filter(finite).map(Number);
    if (!durations.length) empty(host, "No completed layer yet", "Per-layer durations appear after the first layer is saved.");
    else {
      const layers = (job?.completed_layers || []).map(Number);
      const max = Math.max(...durations, 1);
      const c = baseChart(host, {width: 560, height: 170, xLabel: "Completed layer (in order)", yLabel: "Minutes", xMin: 0, xMax: durations.length, yMax: max / 60 * 1.1});
      const band = c.plotW / durations.length;
      durations.forEach((sec, i) => {
        const v = sec / 60;
        const rect = svg("rect", {x: c.m.l + i * band + band * .15, y: c.y(v), width: band * .7, height: Math.max(0, c.y(0) - c.y(v)), fill: "#0a8272", "fill-opacity": .75, rx: 2});
        rect.append(svg("title", {}, `${layers[i] !== undefined ? `layer ${layers[i]}` : `unit ${i + 1}`}: ${duration(sec)}`));
        c.s.append(rect);
        if (durations.length <= 16) c.s.append(svg("text", {x: c.m.l + i * band + band / 2, y: c.height - 23, "text-anchor": "middle", class: "axis-text"}, layers[i] ?? i + 1));
      });
    }
    const history = $("stage-history");
    const rows = (job?.history || []).slice(-6).filter((r) => finite(r.duration_seconds));
    if (job && finite(job.stage_elapsed_seconds)) rows.push({stage: job.stage, duration_seconds: job.stage_elapsed_seconds, current: job.status === "running"});
    history.replaceChildren();
    if (!rows.length) return;
    const max = Math.max(...rows.map((r) => Number(r.duration_seconds)), 1);
    for (const row of rows) {
      const el = node("div", {class: "timing-row" + (row.current ? " current" : "")});
      const top = node("div", {class: "timing-top"});
      top.append(node("span", {}, stageLabel(row.stage) + (row.current ? " · current" : "")), node("span", {}, duration(row.duration_seconds)));
      const track = node("div", {class: "timing-track"});
      const fill = node("div", {class: "timing-fill"});
      fill.style.width = `${Math.max(.5, (100 * Number(row.duration_seconds)) / max)}%`;
      track.append(fill);
      el.append(top, track);
      history.append(el);
    }
  }

  // ---------------------------------------------------------------- tables
  function table(host, columns, rows, rowClass) {
    host.replaceChildren();
    if (!rows.length) { host.append(node("p", {class: "table-empty"}, "No saved rows yet.")); return; }
    const t = node("table", {class: "data-table"});
    const head = node("tr");
    for (const col of columns) head.append(node("th", {title: col.title || ""}, col.label));
    const thead = node("thead"); thead.append(head);
    const body = node("tbody");
    for (const row of rows) {
      const tr = node("tr", {class: rowClass ? rowClass(row) : undefined});
      for (const col of columns) {
        const value = col.value(row);
        const td = node("td", {class: (col.num ? "num " : "") + (col.cls ? col.cls(row) : "")});
        td.textContent = value === null || value === undefined ? "—" : value;
        tr.append(td);
      }
      body.append(tr);
    }
    t.append(thead, body);
    host.append(t);
  }
  const pCell = (v) => (finite(v) ? (Number(v) < .001 ? "<.001" : Number(v).toFixed(3)) : "—");
  const ci = (est, lo, hi) => (finite(est) ? `${fmt(est, 2)} [${fmt(lo, 2)}, ${fmt(hi, 2)}]` : "—");

  function renderInference() {
    const d = state.detail || {};
    let rows = (d.inference || []).filter((r) => !r.group || r.group === "all");
    if (state.primaryOnly) {
      const primary = rows.filter((r) => r.primary_test === true || r.primary_test === "True");
      if (primary.length) rows = primary;
    }
    rows = rows.slice().sort((a, b) => String(a.concept).localeCompare(String(b.concept)) || String(source(a)).localeCompare(String(source(b))));
    table($("inference-table"), [
      {label: "Concept", value: (r) => r.concept},
      {label: "Direction", value: (r) => `${source(r)} → ${target(r)}`},
      {label: "M", value: (r) => r.M, num: true},
      {label: "n", value: (r) => fmtInt(r.n), num: true, title: "test items (component clusters below)"},
      {label: "R(e→p)", value: (r) => fmt(r.R_ep), num: true},
      {label: "R(p→p)", value: (r) => fmt(r.R_pp), num: true},
      {label: "Floor", value: (r) => fmt(r.R0), num: true},
      {label: "SSI [95% CI]", value: (r) => ci(r.SSI, r.ci_lo, r.ci_hi), num: true},
      {label: "RAND null SSI", value: (r) => fmt(r.RAND_SSI_mean, 2), num: true},
      {label: "p (RAND)", value: (r) => pCell(finite(r.RAND_p) ? r.RAND_p : r.RAND_p_diagnostic), num: true},
      {label: "Holm", value: (r) => pCell(r.RAND_p_holm), num: true, cls: (r) => (finite(r.RAND_p_holm) && Number(r.RAND_p_holm) < .05 ? "sig" : "ns")},
      {label: "p (PERM)", value: (r) => (finite(r.PERM_p) ? pCell(r.PERM_p) : r.PERM_status ? pretty(r.PERM_status) : "—"), num: true},
      {label: "Null-adj. SSI", value: (r) => fmt(r.SSI_adjusted_RAND ?? r.SSI_adjusted, 2), num: true},
    ], rows, (r) => (r.primary_test === true || r.primary_test === "True" ? "primary" : ""));
  }

  function renderBaselines() {
    const d = state.detail || {};
    const rows = (d.baselines || []).slice().sort((a, b) => String(source(a)).localeCompare(String(source(b))));
    table($("baseline-table"), [
      {label: "Pair", value: (r) => `${source(r)} → ${target(r)}`},
      {label: "Centering", value: (r) => pretty(r.centering)},
      {label: "n", value: (r) => fmtInt(r.n), num: true},
      {label: "CKA", value: (r) => fmt(r.CKA), num: true},
      {label: "Top-1", value: (r) => fmt(r.retr_top1), num: true},
      {label: "MRR", value: (r) => fmt(r.retr_MRR), num: true},
      {label: "Chance", value: (r) => fmt(r.chance_top1, 4), num: true},
    ], rows);
  }

  function renderContrasts() {
    const d = state.detail || {};
    table($("contrast-table"), [
      {label: "Contrast", value: (r) => r.contrast},
      {label: "Concept", value: (r) => r.concept},
      {label: "Pair", value: (r) => `${source(r)} → ${target(r)}`},
      {label: "ΔSSI [95% CI]", value: (r) => ci(r.estimate, r.ci_lo, r.ci_hi), num: true},
      {label: "n a / b", value: (r) => `${fmtInt(r.n_a)} / ${fmtInt(r.n_b)}`, num: true},
    ], (d.contrasts || []).slice(0, 60));
  }

  function renderSelection(job) {
    const host = $("selection");
    host.replaceChildren();
    const sel = job?.selection || {};
    if (!finite(sel.layer)) { host.append(node("p", {class: "table-empty"}, "Selected after all layer grids are saved.")); return; }
    host.append(node("div", {class: "big"}, `Layer ${sel.layer}`));
    host.append(node("div", {class: "muted compact"}, `Validation score ${fmt(sel.validation_score)} · ${sel.criterion || ""}`));
    const chips = node("div", {class: "candidates"});
    for (const pair of sel.candidates || []) {
      const [score, layer] = pair;
      chips.append(node("span", {class: Number(layer) === Number(sel.layer) ? "best" : ""}, `L${layer}: ${fmt(score)}`));
    }
    host.append(chips);
  }

  function renderLog(data) {
    const pre = $("log-tail");
    const lines = data?.log_tail || [];
    const nearBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 40;
    pre.textContent = lines.length ? lines.join("\n") : "No job.log found for this job (logs are written by launch_gpu_jobs.py, or redirect your run's output to job.log).";
    if (nearBottom) pre.scrollTop = pre.scrollHeight;
  }

  // ---------------------------------------------------------------- operators (illustrative)
  function renderProjection() {
    const oblique = state.projection === "oblique", host = $("projection-chart");
    host.replaceChildren();
    const s = svg("svg", {viewBox: "0 0 340 245", role: "img", "aria-label": oblique ? "Illustrative oblique projection: (2,1) becomes (1,0), leaving residual (1,1)" : "Illustrative orthogonal projection: (2,1) becomes (2,0), leaving residual (0,1)"});
    const ox = 53, oy = 190, scale = 87;
    const xp = ox + 2 * scale, yp = oy - scale, pp = ox + (oblique ? 1 : 2) * scale;
    const defs = svg("defs");
    for (const [id, c] of [["input", "#0a8272"], ["projected", "#df8a44"], ["residual", "#829a9d"]]) {
      const marker = svg("marker", {id: "arrow-" + id, viewBox: "0 0 8 8", refX: 6, refY: 4, markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse"});
      marker.append(svg("path", {d: "M0 0L8 4L0 8z", fill: c}));
      defs.append(marker);
    }
    s.append(defs);
    for (let i = 0; i < 4; i += 1) s.append(svg("line", {x1: ox + i * scale, y1: 25, x2: ox + i * scale, y2: oy + 8, class: "grid-line"}));
    for (let i = 0; i < 3; i += 1) s.append(svg("line", {x1: ox - 8, y1: oy - i * scale, x2: 320, y2: oy - i * scale, class: "grid-line"}));
    s.append(svg("path", {d: `M${ox} 21V${oy}H322`, class: "axis-line", fill: "none", "stroke-width": 1.2}));
    s.append(svg("line", {x1: ox, y1: oy, x2: xp, y2: yp, stroke: "#0a8272", "stroke-width": 2, "marker-end": "url(#arrow-input)"}));
    s.append(svg("line", {x1: ox, y1: oy, x2: pp, y2: oy, stroke: "#df8a44", "stroke-width": 3, "marker-end": "url(#arrow-projected)"}));
    s.append(svg("line", {x1: pp, y1: oy, x2: xp, y2: yp, stroke: "#829a9d", "stroke-width": 1.6, "stroke-dasharray": "5 4", "marker-end": "url(#arrow-residual)"}));
    s.append(svg("circle", {cx: xp, cy: yp, r: 4, fill: "#0a8272"}), svg("circle", {cx: pp, cy: oy, r: 4, fill: "#df8a44"}));
    const label = (x, y, t, fill) => s.append(svg("text", {x, y, class: "axis-text", fill}, t));
    label(ox - 17, oy + 17, "0"); label(xp + 10, yp - 7, "x = (2, 1)", "#0a8272");
    label(pp - 20, oy + 22, oblique ? "Px = (1, 0)" : "Px = (2, 0)", "#c78443"); label(ox + 4, 16, "y"); label(321, oy - 9, "x");
    label(oblique ? pp + 34 : xp + 10, oy - 35, "residual", "#829a9d");
    host.append(s);
    setText("projection-value", oblique ? "(1, 0)" : "(2, 0)");
    setText("residual-value", oblique ? "(1, 1)" : "(0, 1)");
    setText("projection-note", oblique ? "An oblique projection follows a tilted complement. The image remains on the x-axis." : "An orthogonal projection drops the perpendicular component.");
  }

  // ---------------------------------------------------------------- refresh loop
  function setConnection(ok) {
    state.connected = ok;
    const badge = $("connection-badge");
    const live = state.detail?.job?.live === true;
    badge.className = "feed-badge" + (SNAPSHOT ? " snapshot" : ok ? (live ? "" : " snapshot") : " offline");
    badge.replaceChildren(node("span"), document.createTextNode(SNAPSHOT ? "SNAPSHOT" : ok ? (live ? "LIVE" : "LOCAL SNAPSHOT") : "RECONNECTING"));
  }

  async function refresh() {
    if (state.busy) return;
    state.busy = true;
    try {
      const overview = await api.overview();
      state.overview = overview;
      const jobs = overview.jobs || [];
      if (!jobs.some((j) => j.id === state.selected)) {
        state.selected = (jobs.find((j) => j.live === true) || jobs.find((j) => j.status === "running" && !j.stale) || jobs[0])?.id || null;
      }
      renderSidebar();
      renderRunBand();
      if (state.selected) {
        const id = state.selected;
        const detail = await api.job(id);
        if (id === state.selected) { state.detail = detail; renderDetail(); }
      } else { state.detail = null; renderDetail(); }
      setConnection(true);
      $("connection-error").hidden = true;
      setText("last-sync", SNAPSHOT ? `Exported ${new Date(SNAPSHOT.exported_at).toLocaleString()}` : `Last sync ${new Date().toLocaleTimeString()}`);
    } catch (err) {
      setConnection(false);
      $("connection-error").hidden = false;
      setText("connection-error", "The local feed is unavailable. The last saved view remains visible; reconnecting automatically.");
    } finally { state.busy = false; }
  }

  state.selected = store("idiom-observatory-job");
  for (const id of ["filter-concept", "filter-method", "filter-rank", "filter-source", "filter-target", "open-metric"]) {
    $(id).addEventListener("change", () => { if (id === "filter-concept" || id === "filter-method") renderFilters(); renderCharts(); });
  }
  $("estimator-layer").addEventListener("change", () => { state.estimatorLayer = number($("estimator-layer").value); renderEstimators(); });
  $("primary-only").addEventListener("change", (event) => { state.primaryOnly = event.target.checked; renderInference(); });
  document.querySelectorAll("[data-metric]").forEach((button) => button.addEventListener("click", () => {
    state.metric = button.dataset.metric;
    document.querySelectorAll("[data-metric]").forEach((b) => { const on = b === button; b.classList.toggle("selected", on); b.setAttribute("aria-pressed", String(on)); });
    renderCharts();
  }));
  document.querySelectorAll("[data-projection]").forEach((button) => button.addEventListener("click", () => {
    state.projection = button.dataset.projection;
    document.querySelectorAll("[data-projection]").forEach((b) => { const on = b === button; b.classList.toggle("selected", on); b.setAttribute("aria-pressed", String(on)); });
    renderProjection();
  }));
  if (SNAPSHOT) {
    setText("mode-label", "OFFLINE SNAPSHOT");
    setText("poll-note", "Static export; reopen a new snapshot to update");
    setText("footer-note", "Offline snapshot · no server · read-only");
  }
  renderProjection();
  renderDetail();
  refresh();
  if (!SNAPSHOT) setInterval(refresh, POLL_MS);
})();
