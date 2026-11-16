"use strict";

// Control room: send a video from a phone in resumable chunks, then follow it through
// the pipeline. Plain JS, no build step (same as the review page).

const CHUNK = 8 * 1024 * 1024;
const POLL_MS = 3000;
const $ = (id) => document.getElementById(id);

let steps = [];
let activeUpload = null; // { id, cancelled }

// --- small helpers -------------------------------------------------------------------

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v);
  }
  for (const c of children) if (c) node.append(c);
  return node;
}

function gb(bytes) {
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(2)} GB` : `${(bytes / 1e6).toFixed(0)} MB`;
}

function ago(iso) {
  if (!iso) return "";
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
}

function duration(sec) {
  const m = Math.floor(sec / 60), s = Math.round(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

async function api(path, options = {}) {
  const headers = { "X-Pokepipe": "1", ...(options.headers || {}) };
  const res = await fetch(path, { ...options, headers });
  if (res.status === 401) {
    location.href = "/login";
    throw new Error("sign in first");
  }
  const body = res.status === 204 ? null : await res.json().catch(() => null);
  if (!res.ok) {
    const err = new Error((body && body.detail) || `${res.status} ${res.statusText}`);
    err.status = res.status;
    err.body = body;
    throw err;
  }
  return body;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function storeGet(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function storeSet(key, value) {
  try { localStorage.setItem(key, value); } catch { /* private mode */ }
}

// --- sets ----------------------------------------------------------------------------

async function loadSets() {
  const data = await api("/api/sets");
  const select = $("set");
  select.replaceChildren();
  for (const s of data.sets) select.append(el("option", { value: s.id, text: `${s.name} (${s.id})` }));
  const remembered = storeGet("pokepipe.set");
  const pick = [remembered, data.default].find((id) => id && data.sets.some((s) => s.id === id));
  if (pick) select.value = pick;
  select.addEventListener("change", () => storeSet("pokepipe.set", select.value));
  if (!data.sets.length) {
    select.append(el("option", { value: "", text: "No sets cached — run pokepipe cards fetch" }));
  }
}

// --- upload --------------------------------------------------------------------------

function showUpload(name, received, size, detail) {
  $("upload").hidden = false;
  $("up-name").textContent = name;
  const pct = size ? Math.floor((received / size) * 100) : 0;
  $("up-pct").textContent = `${pct}%`;
  $("up-bar").style.width = `${pct}%`;
  $("up-detail").textContent = detail;
}

function note(text, kind = "") {
  const n = $("up-note");
  n.textContent = text;
  n.className = `note ${kind}`;
  n.hidden = !text;
}

async function keepAwake() {
  try { return await navigator.wakeLock.request("screen"); } catch { return null; }
}

async function upload(file) {
  const setId = $("set").value;
  if (!setId) { note("Choose a set first.", "bad"); return; }
  note("");
  $("file").disabled = true;
  $("up-cancel").hidden = false;
  let lock = await keepAwake();
  const relock = async () => {
    if (document.visibilityState === "visible" && activeUpload) lock = await keepAwake();
  };
  document.addEventListener("visibilitychange", relock);

  try {
    const up = await api("/api/uploads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: file.name, size: file.size, set_id: setId }),
    });
    activeUpload = { id: up.id, cancelled: false };
    if (up.received > 0) note(`Resuming: ${gb(up.received)} already on the PC.`);

    let received = up.received;
    let failures = 0;
    const t0 = performance.now();
    const startBytes = received;
    while (received < file.size) {
      if (activeUpload.cancelled) return;
      const end = Math.min(file.size, received + CHUNK);
      try {
        const res = await api(`/api/uploads/${up.id}?offset=${received}`, {
          method: "PUT",
          headers: { "Content-Type": "application/octet-stream" },
          body: file.slice(received, end),
        });
        received = res.received;
        failures = 0;
      } catch (err) {
        if (err.status === 409 && err.body) { received = err.body.received; continue; }
        if (err.status && err.status < 500) throw err;
        failures += 1; // network drop or the PC restarting: wait, then ask where we are
        const wait = Math.min(30, 2 ** failures);
        showUpload(file.name, received, file.size, `Connection lost, retrying in ${wait} s…`);
        await sleep(wait * 1000);
        try { received = (await api(`/api/uploads/${up.id}`)).received; } catch { /* retry */ }
        continue;
      }
      const secs = (performance.now() - t0) / 1000;
      const rate = (received - startBytes) / Math.max(secs, 0.001);
      const left = rate > 0 ? (file.size - received) / rate : 0;
      showUpload(file.name, received, file.size,
        `${gb(received)} of ${gb(file.size)} · ${(rate / 1e6).toFixed(1)} MB/s · ${duration(left)} left`);
    }

    showUpload(file.name, file.size, file.size, "Checking the video…");
    const done = await api(`/api/uploads/${up.id}/finish`, { method: "POST" });
    const c = done.check;
    const facts = c ? `${c.width}×${c.height}, ${Math.round(c.fps)} fps, ${duration(c.duration)}` : "";
    showUpload(file.name, file.size, file.size, `Sent · ${facts}`);
    if (c && c.warning) note(c.warning, "warn");
    else note("In the inbox. The watcher picks it up within a few seconds.", "ok");
    refresh();
  } catch (err) {
    note(err.message, "bad");
  } finally {
    activeUpload = null;
    $("file").disabled = false;
    $("file").value = "";
    $("up-cancel").hidden = true;
    document.removeEventListener("visibilitychange", relock);
    if (lock) lock.release().catch(() => {});
  }
}

$("file").addEventListener("change", () => {
  const file = $("file").files[0];
  if (file) upload(file);
});

$("up-cancel").addEventListener("click", async () => {
  if (!activeUpload) return;
  activeUpload.cancelled = true;
  await api(`/api/uploads/${activeUpload.id}`, { method: "DELETE" }).catch(() => {});
  $("upload").hidden = true;
  note("");
});

// --- status --------------------------------------------------------------------------

function watcherPill(w) {
  const pill = $("watcher");
  if (!w.running) {
    pill.textContent = "Watcher not running here";
    pill.className = "pill off";
    pill.title = "Start it on the PC: pokepipe watch run --lan";
    return;
  }
  const busy = w.working_on && w.last_message;
  pill.textContent = busy ? `Working · ${w.working_on}` : "Watching";
  pill.className = "pill on";
  pill.title = w.last_message || "";
}

function pcPanel(pc) {
  const box = $("pc");
  if (!pc) { box.hidden = true; return; }
  box.hidden = false;
  const body = $("pc-body");
  body.replaceChildren();
  if (pc.lan && pc.phone_urls.length) {
    body.append(el("p", {}, "On a phone on the same Wi-Fi, open ",
      el("strong", { class: "num", text: pc.phone_urls.join(" or ") }),
      " and enter the code ", el("strong", { class: "code", text: pc.code })));
  } else {
    body.append(el("p", { class: "muted" }, "Phones can't reach this page yet. Restart with ",
      el("code", { text: "pokepipe watch run --lan" }), " to let them in."));
  }
  if (pc.review_url) {
    body.append(el("p", {}, el("a", { href: pc.review_url, target: "_blank", text: "Open the review page" }),
      el("span", { class: "muted", text: " (PC only)" })));
  }
}

function stepper(v) {
  const ol = el("ol", { class: "steps" });
  steps.forEach((label, i) => {
    let state = i < v.step ? "done" : i === v.step ? "now" : "todo";
    if (v.status === "done") state = "done";
    if (v.failed && i === v.step) state = "fail";
    if (state === "now" && v.status === "awaiting_review") state = "wait";
    ol.append(el("li", { class: state, title: label }, el("span", { text: label })));
  });
  return ol;
}

function headline(v) {
  if (v.failed) return `Failed at ${steps[v.step].toLowerCase()}`;
  if (v.status === "done") return v.posts.length === 1 ? "1 post ready" : `${v.posts.length} posts ready`;
  if (v.status === "awaiting_review") {
    return v.review_pending ? `Waiting for review · ${v.review_pending} to check` : "Waiting for review";
  }
  return steps[v.step];
}

function videoCard(v) {
  const facts = [];
  if (v.pulls != null) facts.push(`${v.pulls} cards in ${v.packs} packs`);
  if (v.cost_usd) facts.push(`$${v.cost_usd.toFixed(2)} spent`);
  facts.push(`started ${ago(v.started_at)}`);

  const card = el("article", { class: `video ${v.failed ? "is-failed" : ""} ${v.status === "done" ? "is-done" : ""}` },
    el("div", { class: "video-head" },
      el("div", {},
        el("h3", { text: v.slug }),
        el("p", { class: "muted small", text: `${v.set_name} · ${facts.join(" · ")}` })),
      el("span", { class: `status s-${v.failed ? "failed" : v.status}`, text: headline(v) })),
    stepper(v));

  if (v.failed && v.error) {
    const retry = el("button", { type: "button", class: "primary", text: "Retry from this step" });
    retry.addEventListener("click", async () => {
      retry.disabled = true;
      try { await api(`/api/jobs/${v.slug}/retry`, { method: "POST" }); refresh(); }
      catch (err) { retry.disabled = false; alertBox(card, err.message); }
    });
    card.append(el("div", { class: "error-box" }, el("p", { text: v.error }), retry));
  } else if (v.message && v.status !== "done") {
    card.append(el("p", { class: "muted small last", text: `${v.message} · ${ago(v.updated_at)}` }));
  }

  if (v.posts.length) {
    const row = el("div", { class: "posts" });
    for (const n of v.posts) {
      row.append(el("a", { class: "post-link", href: `/media/${v.slug}/posts/${n}.mp4`, target: "_blank",
        text: `▶ Part ${n}` }));
    }
    card.append(row);
  }
  return card;
}

// The lime "Needs you" card: the review is the pipeline's only human stop, so it goes
// above everything else. It links to the review page only on the PC (loopback-only).
function needsCard(data) {
  const card = $("needs");
  const waiting = data.videos.filter((v) => !v.failed && v.status === "awaiting_review");
  if (!waiting.length) { card.hidden = true; return; }
  card.hidden = false;
  const first = waiting[0];
  const cards = waiting.reduce((n, v) => n + (v.review_pending || 0), 0);
  const what = cards ? `${cards} card${cards === 1 ? "" : "s"} to check` : "A review to finish";
  $("needs-title").textContent = waiting.length === 1 ? `${what} in ${first.slug}` : `${what} in ${waiting.length} videos`;
  const reviewUrl = data.pc && data.pc.review_url;
  if (reviewUrl) {
    card.href = `${reviewUrl}#${encodeURIComponent(first.slug)}`;
    card.target = "_blank";
    $("needs-sub").textContent = "Open the review page →";
  } else {
    card.removeAttribute("href");
    $("needs-sub").textContent = "Open the review page on the PC";
  }
}

function alertBox(parent, text) {
  parent.append(el("p", { class: "error", text }));
}

async function refresh() {
  let data;
  try { data = await api("/api/status"); } catch { return; }
  steps = data.steps;
  watcherPill(data.watcher);
  pcPanel(data.pc);

  const inbox = $("inbox");
  inbox.hidden = !data.inbox.length;
  inbox.textContent = data.inbox.length
    ? `In the inbox, waiting for the watcher: ${data.inbox.map((f) => f.name).join(", ")}`
    : "";

  needsCard(data);

  const list = $("videos");
  list.replaceChildren();
  if (!data.videos.length) {
    list.append(el("p", { class: "muted", text: "No videos yet. Send one above and it appears here." }));
  }
  for (const v of data.videos) list.append(videoCard(v));
}

loadSets().catch((err) => note(err.message, "bad"));
refresh();
setInterval(() => { if (document.visibilityState === "visible") refresh(); }, POLL_MS);
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") refresh(); });
