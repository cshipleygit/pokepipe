// Pull review page. Vanilla JS: state lives on the server (review.json); after each
// edit we refetch and re-render, which keeps the client trivially consistent.
"use strict";

// Mirrors pokepipe.cards.rarity.is_special: Double Rare and above, including rarity
// names a new era invents (e.g. "Mega Hyper Rare").
const BASE_RARITIES = new Set(["Common", "Uncommon", "Rare", "Rare Holo"]);
const isHit = (rarity) => !!rarity && !BASE_RARITIES.has(rarity) && rarity.includes("Rare");
const LOW_CONFIDENCE = 0.7;

const $ = (sel) => document.querySelector(sel);
const state = { slug: null, data: null, cards: {}, cardList: [], pack: 1, selected: 0 };

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  if (!res.ok) {
    const detail = (await res.json().catch(() => ({}))).detail || res.statusText;
    toast(typeof detail === "string" ? detail : "Request failed");
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toast.t);
  toast.t = setTimeout(() => (el.hidden = true), 2500);
}

const fmt = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
const card = (id) => state.cards[id] || { id, name: id, number: "?", rarity: null };
const cardImg = (id) => `/cards/${state.data.review.set_id}/${id}.png`;

// --- loading ---------------------------------------------------------------------------

async function init() {
  const videos = await api("/api/videos");
  const select = $("#video-select");
  if (!videos.length) {
    $("#empty").hidden = false;
    select.hidden = true;
    return;
  }
  // Label by slug: the watcher renames every file to source.<ext>, so file names repeat.
  select.innerHTML = videos
    .map((v) => `<option value="${esc(v.slug)}">${esc(videoLabel(v))}</option>`)
    .join("");
  setView("pack"); // Pack is the primary view; every visit starts there
  const fromHash = decodeURIComponent(location.hash.slice(1));
  select.value = videos.some((v) => v.slug === fromHash) ? fromHash : videos[0].slug;
  select.onchange = () => openVideo(select.value);
  await openVideo(select.value);
}

async function openVideo(slug) {
  state.slug = slug;
  location.hash = slug;
  await refresh();
  const setId = state.data.review.set_id;
  state.cardList = await api(`/api/sets/${setId}/cards`);
  state.cards = Object.fromEntries(state.cardList.map((c) => [c.id, c]));
  $("#video").src = `/media/${slug}/video`;
  state.pack = firstPackNeedingWork() ?? 1;
  render();
}

async function refresh() {
  state.data = await api(`/api/videos/${state.slug}`);
}

function firstPackNeedingWork() {
  const p = state.data.packs.find((p) => p.pending > 0 || !p.complete);
  return p ? p.index : null;
}

// --- rendering -------------------------------------------------------------------------

function packPulls() {
  return state.data.review.pulls
    .filter((p) => p.pack === state.pack)
    .sort((a, b) => a.time - b.time);
}

// Every pull still waiting for a decision, in viewing order: the review's to-do list.
function pendingPulls() {
  return state.data.review.pulls
    .filter((p) => p.status === "pending")
    .sort((a, b) => a.pack - b.pack || a.time - b.time);
}

function currentPull() {
  return packPulls()[state.selected] || null;
}

// After a decision, move to the next card that needs one: later in this pack first,
// then the first one anywhere. Stays put when nothing is left.
function goNextPending(fromId) {
  const queue = pendingPulls().filter((p) => p.id !== fromId);
  if (!queue.length) return;
  const here = pullById(fromId);
  const next = (here && queue.find((p) => p.pack === here.pack && p.time > here.time)) || queue[0];
  jumpTo(next);
}

function jumpTo(pull) {
  state.pack = pull.pack;
  state.selected = Math.max(0, packPulls().findIndex((p) => p.id === pull.id));
}

const QUEUE_SHOWN = 12;
const MAX_CHOICES = 3;

function render() {
  const { review, packs, detection } = state.data;
  const done = review.pulls.filter((p) => p.status !== "pending").length;
  const kept = review.pulls.filter((p) => p.status === "confirmed").length;
  const pending = pendingPulls();
  $("#progress").textContent = `${done}/${review.pulls.length} reviewed · ${kept} kept` +
    (review.completed ? " · finished" : "");
  $("#finish").disabled = done < review.pulls.length || review.completed;
  $("#needs-n").textContent = pending.length;
  $("#needs-label").textContent = pending.length === 1 ? "card needs you" : "cards need you";
  $(".needs-count").classList.toggle("clear", !pending.length);

  const cur = currentPull();
  $("#queue").innerHTML = pending.length
    ? pending.slice(0, QUEUE_SHOWN).map((p) => {
      const why = p.suggested ? "missed?" : p.detected_index == null ? "added" : "check";
      return `<li><button data-qid="${p.id}" aria-current="${!!cur && cur.id === p.id}">
        <span>Pack ${p.pack} · ${esc(card(p.card_id).name)}</span><small>${why}</small></button></li>`;
    }).join("") + (pending.length > QUEUE_SHOWN ? `<li class="clear">+${pending.length - QUEUE_SHOWN} more</li>` : "")
    : `<li class="clear">${review.completed ? "Review finished." : "Nothing left. Finish the review."}</li>`;

  $("#pack-list").innerHTML = packs.map((p) => {
    const pulls = review.pulls.filter((x) => x.pack === p.index).sort((a, b) => a.time - b.time);
    const slots = pulls.map((x) => {
      const hit = x.status !== "rejected" && isHit(card(x.card_id).rarity);
      return `<i class="slot ${x.status}${hit ? " hit" : ""}"></i>`;
    });
    for (let n = p.cards; n < p.expected; n++) slots.push(`<i class="slot missing" title="missing card"></i>`);
    const meta = p.pending ? `<span class="pack-meta todo">${p.pending} to check</span>`
      : !p.complete ? `<span class="pack-meta bad">${p.cards}/${p.expected}</span>`
      : `<span class="pack-meta">${p.cards}/${p.expected} ✓</span>`;
    return `<li><button data-pack="${p.index}" aria-current="${p.index === state.pack}">
      <span class="pack-row"><b>Pack ${p.index}</b>${meta}</span>
      <span class="slots" aria-hidden="true">${slots.join("")}</span></button></li>`;
  }).join("");

  const pack = packs.find((p) => p.index === state.pack);
  const detPack = detection.packs.find((p) => p.index === state.pack);
  $("#pack-title").textContent = pack
    ? `Pack ${pack.index} — ${pack.cards} of ${pack.expected} cards`
    : `Pack ${state.pack}`;
  const note = detPack && detPack.layout_note ? ` · ${detPack.layout_note}` : "";
  $("#clip-label").textContent = detPack ? `Clip ${fmt(detPack.clip_start)} – ${fmt(detPack.clip_end)}${note}` : "";
  $("#play-clip").disabled = !detPack;

  const pulls = packPulls();
  state.selected = Math.min(state.selected, Math.max(pulls.length - 1, 0));
  $("#pulls").innerHTML = pulls.map((p, i) => pullTile(p, i)).join("");
  renderFocus();
  renderTimeline();
  labelVideoOption();
}

// --- views: Pack (default), Card, Video ------------------------------------------------

const VIEWS = ["pack", "card", "video"];

function setView(view) {
  if (!VIEWS.includes(view)) return;
  state.view = view;
  $("#main").dataset.view = view;
  for (const tab of document.querySelectorAll(".tabs button")) {
    tab.setAttribute("aria-selected", String(tab.dataset.view === view));
  }
  if (view === "pack") $("#video").pause();
}

// The video's length, before its metadata loads: the latest moment we know of.
function videoLength() {
  const d = $("#video").duration;
  if (Number.isFinite(d) && d > 0) return d;
  const { detection, review } = state.data;
  return Math.max(1, ...detection.packs.map((p) => p.clip_end), ...review.pulls.map((p) => p.time));
}

const pct = (t) => `${Math.min(100, Math.max(0, (t / videoLength()) * 100)).toFixed(3)}%`;

// Video view: one tick per card (lime and taller while it needs you, foil for a hit),
// each pack's clip as a band behind them, and the packs as chapters underneath.
function renderTimeline() {
  if (!state.data) return;
  const { detection, review, packs } = state.data;
  const cur = currentPull();
  $("#bands").innerHTML = detection.packs.map((p) =>
    `<i class="band ${p.index === state.pack ? "current" : ""}" style="left:${pct(p.clip_start)};width:calc(${pct(p.clip_end)} - ${pct(p.clip_start)})"></i>`,
  ).join("");
  $("#ticks").innerHTML = review.pulls.map((p) => {
    const c = card(p.card_id);
    const hit = isHit(c.rarity) ? " hit" : "";
    const sel = cur && cur.id === p.id ? " selected" : "";
    return `<button class="tick ${p.status}${hit}${sel}" data-tick="${p.id}" style="left:${pct(p.time)}"
      title="Pack ${p.pack} · ${esc(c.name)} · ${fmt(p.time)}" aria-label="Pack ${p.pack}, ${esc(c.name)} at ${fmt(p.time)}"></button>`;
  }).join("");
  $("#chapters").innerHTML = detection.packs.map((p) => {
    const info = packs.find((x) => x.index === p.index);
    const todo = info && info.pending;
    return `<li><button class="${todo ? "todo" : ""}" data-chapter="${p.index}"><b>Pack ${p.index}</b>${fmt(p.clip_start)} – ${fmt(p.clip_end)}${todo ? ` · ${todo} need you` : ""}</button></li>`;
  }).join("");
  movePlayhead();
}

function movePlayhead() {
  $("#playhead").style.left = pct($("#video").currentTime || 0);
}

// The <select> is built once; keep the open video's label current as cards are decided.
function videoLabel(v) {
  const status = v.completed ? "finished" : `${v.pending} to check`;
  return `${v.slug} · ${v.packs} packs · ${status}`;
}

function labelVideoOption() {
  const opt = [...$("#video-select").options].find((o) => o.value === state.slug);
  if (!opt || !state.data) return;
  const { review, detection } = state.data;
  opt.textContent = videoLabel({
    slug: state.slug, packs: detection.packs.length, completed: review.completed,
    pending: review.pulls.filter((p) => p.status === "pending").length,
  });
}

// The card being decided, big, with the likely answers as buttons (keys 1-3).
function choicesFor(p, det) {
  const ids = [p.card_id];
  if (det) for (const id of [det.card_id, ...det.alternatives.map((a) => a.card_id)]) {
    if (!id.startsWith("decoy:") && !ids.includes(id)) ids.push(id);
  }
  return ids.slice(0, MAX_CHOICES);
}

function renderFocus() {
  const box = $("#focus");
  const p = currentPull();
  if (!p) {
    box.innerHTML = `<p class="done-note">No cards in this pack.</p>`;
    return;
  }
  const det = p.detected_index != null ? state.data.detection.pulls[p.detected_index] : null;
  const c = card(p.card_id);
  const hit = isHit(c.rarity);
  const question = p.status !== "pending" ? c.name
    : p.suggested ? `Missed card: ${c.name}?`
    : !det ? `Added: ${c.name}`
    : `Is this ${c.name}?`;
  // Explain why a card is waiting; once it's settled, just describe it.
  const open = p.status === "pending";
  const why = open && p.suggested ? `The recovery agent thinks this was missed. ${p.suggested}`
    : open && p.auto_note ? `Not checked automatically: ${p.auto_note}`
    : open && det && det.confidence < LOW_CONFIDENCE ? `Low confidence (${Math.round(det.confidence * 100)}%).`
    : [c.rarity, p.reverse_holo ? "reverse holo" : ""].filter(Boolean).join(" · ");
  const shot = det && det.thumbnail
    ? `<img src="/media/${state.slug}/${det.thumbnail}" alt="Frame from video">`
    : `<div class="no-shot">${p.suggested ? "Press space to watch the moment" : "Added by hand"}</div>`;
  const ref = `<img src="${cardImg(p.card_id)}" alt="${esc(c.name)} reference">`;
  const choices = choicesFor(p, det).map((id, i) => {
    const alt = card(id);
    return `<button class="choice ${id === p.card_id ? "current" : ""}" data-choice="${id}">
      <span>${esc(alt.name)}<small>#${alt.number} · ${alt.rarity || "?"}</small></span><kbd>${i + 1}</kbd></button>`;
  }).join("");
  box.innerHTML = `
    <div class="eyebrow"><span>Pack ${p.pack}${det && det.slot ? ` · slot ${det.slot}` : ""} · ${fmt(p.time)}</span>
      <span>${p.status === "pending" ? "needs you" : p.status}</span></div>
    <h1>${esc(question)}</h1>
    ${why ? `<p class="why">${esc(why)}</p>` : ""}
    <div class="imgs">
      <figure>${shot}<figcaption>From the video</figcaption></figure>
      <figure>${hit ? `<div class="holo">${ref}</div>` : ref}<figcaption>#${c.number} · ${c.rarity || ""}</figcaption></figure>
    </div>
    <div class="choices">${choices}</div>
    <div class="row">
      <button data-focus="reject" aria-pressed="${p.status === "rejected"}">Not a pull · x</button>
      <button data-focus="reverse" aria-pressed="${p.reverse_holo}">Reverse holo · r</button>
      <button data-focus="change">Something else · e</button>
      <button data-focus="seek">Play · space</button>
      ${det ? "" : `<button data-focus="delete">Delete added card</button>`}
    </div>`;
}

// Picking an answer settles the card: right card, kept, then on to the next one.
async function choose(p, cardId) {
  const body = { status: "confirmed" };
  if (cardId !== p.card_id) body.card_id = cardId;
  await patch(p.id, body);
  goNextPending(p.id);
  render();
}

// Model-written text (agent evidence) goes into an attribute: escape it.
function esc(s) {
  return String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
}

function pullTile(p, i) {
  const det = p.detected_index != null ? state.data.detection.pulls[p.detected_index] : null;
  const c = card(p.card_id);
  const badges = [];
  if (isHit(c.rarity)) badges.push(`<span class="badge hit">${c.rarity}</span>`);
  if (p.decided_by === "auto") badges.push(`<span class="badge auto" title="${esc(p.auto_note || "")}">auto</span>`);
  else if (p.status === "pending" && p.auto_note) badges.push(`<span class="badge low" title="${esc(p.auto_note)}">why: ${esc(p.auto_note.split(";")[0])}</span>`);
  if (det && det.confidence < LOW_CONFIDENCE && p.status === "pending") badges.push(`<span class="badge low">check</span>`);
  if (det && p.card_id !== p.detected_card_id) badges.push(`<span class="badge fixed">was ${card(p.detected_card_id).name}</span>`);
  if (!det && p.suggested) badges.push(`<span class="badge low" title="${esc(p.suggested)}">agent: missed card?</span>`);
  else if (!det) badges.push(`<span class="badge added">added</span>`);
  if (det && det.showings && det.showings.length) {
    const times = det.showings.map((s) => fmt(s.peak_time)).join(", ");
    badges.push(`<span class="badge rev" title="Shown again at ${times} — kept as footage for the edit">shown ${det.showings.length + 1}×</span>`);
  }
  if (p.reverse_holo) {
    const auto = det && det.reverse_guess;
    badges.push(`<span class="badge rev" title="${auto ? `Inferred from pack order (slot ${det.slot})` : "Set by you"}">reverse holo${auto ? " · auto" : ""}</span>`);
  }
  const shot = det && det.thumbnail
    ? `<img src="/media/${state.slug}/${det.thumbnail}" alt="Frame from video" loading="lazy">`
    : `<div class="no-shot">${p.suggested ? "suggested by the recovery agent — play to check" : "added by hand"}</div>`;
  const conf = det ? Math.round(det.confidence * 100) : 100;
  const stateLabel = { pending: "needs you", confirmed: "kept", rejected: "not a pull" }[p.status];
  // A tile is a summary: clicking it opens the card in the Card view, where the actions are.
  return `<li class="pull ${p.status} ${i === state.selected ? "selected" : ""}" data-id="${p.id}" data-i="${i}">
    <div class="imgs">
      <figure>${shot}<figcaption>video</figcaption></figure>
      <figure><img src="${cardImg(p.card_id)}" alt="${esc(c.name)} reference" loading="lazy"><figcaption>#${esc(c.number)}</figcaption></figure>
    </div>
    <div class="sub"><span class="name">${esc(c.name)}</span><span class="state">${stateLabel}</span></div>
    <div class="sub">${esc(c.rarity || "")} · ${fmt(p.time)}${det ? ` · ${conf}%` : ""} ${badges.join("")}</div>
    ${det ? `<div class="conf" title="Detection confidence ${conf}%"><i style="width:${conf}%"></i></div>` : ""}
  </li>`;
}

// --- actions ---------------------------------------------------------------------------

async function patch(id, body) {
  await api(`/api/videos/${state.slug}/pulls/${id}`, { method: "PATCH", body });
  await refresh();
  render();
}

function pullById(id) {
  return state.data.review.pulls.find((p) => p.id === id);
}

async function act(action, id) {
  const p = pullById(id);
  switch (action) {
    case "confirm": return patch(id, { status: p.status === "confirmed" ? "pending" : "confirmed" });
    case "reject": return patch(id, { status: p.status === "rejected" ? "pending" : "rejected" });
    case "reverse": return patch(id, { reverse_holo: !p.reverse_holo });
    case "change": return openPicker(p);
    case "seek": return seek(p.time - 1);
    case "delete":
      await api(`/api/videos/${state.slug}/pulls/${id}`, { method: "DELETE" });
      await refresh();
      return render();
  }
}

// Playback. `stopAt` lets "Play pack clip" end at the clip boundary instead of running
// on into the next pack; a plain seek clears it.
let stopAt = null;

function seek(t, until = null) {
  const v = $("#video");
  if (state.view === "pack") setView("video"); // the Pack view has no player
  stopAt = until;
  v.currentTime = Math.max(0, t);
  // The player is pinned on wide screens; on narrow ones it scrolls away, so bring it
  // back into view — otherwise playback starts off-screen and looks like nothing happened.
  const r = v.getBoundingClientRect();
  if (r.bottom < 0 || r.top > innerHeight) v.scrollIntoView({ behavior: "smooth", block: "start" });
  v.play().catch((err) => {
    if (err.name !== "AbortError") toast(`Can't play video: ${err.message}`);
  });
}

$("#video").addEventListener("timeupdate", (e) => {
  if (stopAt != null && e.target.currentTime >= stopAt) {
    e.target.pause();
    stopAt = null;
  }
  movePlayhead();
});
// Tick positions use the real duration once it is known.
$("#video").addEventListener("loadedmetadata", () => renderTimeline());

$("#video").addEventListener("error", (e) => {
  const code = e.target.error && e.target.error.code;
  const why = { 2: "network error — is `pokepipe review serve` still running?",
                3: "the file couldn't be decoded",
                4: "the source video is missing or in an unsupported format" }[code];
  toast(`Video failed to load: ${why || "unknown error"}`);
});

// Card picker: used both to correct a pull and to add a missed one.
let pickerDone = null;

function openPicker(pull) {
  const dlg = $("#picker");
  const det = pull && pull.detected_index != null ? state.data.detection.pulls[pull.detected_index] : null;
  $("#picker-title").textContent = pull ? `Change ${card(pull.card_id).name}` : `Add a missed card to pack ${state.pack}`;
  const alts = det ? [det.card_id, ...det.alternatives.map((a) => a.card_id)] : [];
  $("#picker-alts").innerHTML = alts.length
    ? alts.map((id) => pickButton(id)).join("")
    : "";
  $("#picker-search").value = "";
  renderResults("");
  pickerDone = async (cardId) => {
    dlg.close();
    if (pull) return patch(pull.id, { card_id: cardId });
    await api(`/api/videos/${state.slug}/pulls`, {
      method: "POST",
      body: { card_id: cardId, pack: state.pack, time: $("#video").currentTime || 0 },
    });
    await refresh();
    render();
    toast(`Added ${card(cardId).name}`);
  };
  dlg.showModal();
  $("#picker-search").focus();
}

function pickButton(id) {
  const c = card(id);
  return `<button type="button" data-pick="${id}"><img src="${cardImg(id)}" alt="" loading="lazy">
    <span>${c.name} <small>#${c.number} · ${c.rarity || ""}</small></span></button>`;
}

function renderResults(q) {
  q = q.trim().toLowerCase();
  const hits = !q ? [] : state.cardList
    .filter((c) => c.name.toLowerCase().includes(q) || c.number === q)
    .slice(0, 15);
  $("#picker-results").innerHTML = hits.map((c) => `<li>${pickButton(c.id)}</li>`).join("");
}

// --- events ----------------------------------------------------------------------------

$("#pack-list").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-pack]");
  if (!btn) return;
  state.pack = Number(btn.dataset.pack);
  state.selected = 0;
  render();
});

// A tile in the Pack view opens that card in the Card view.
$("#pulls").addEventListener("click", (e) => {
  const tile = e.target.closest(".pull");
  if (!tile) return;
  state.selected = Number(tile.dataset.i);
  setView("card");
  render();
});

$("#queue").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-qid]");
  if (!btn) return;
  jumpTo(pullById(Number(btn.dataset.qid)));
  setView("card");
  render();
});

$(".tabs").addEventListener("click", (e) => {
  const tab = e.target.closest("button[data-view]");
  if (tab) setView(tab.dataset.view);
});

// Video view: a tick selects its card and plays from just before it; anywhere else
// on the bar seeks there.
$("#timeline").addEventListener("click", (e) => {
  const tick = e.target.closest("button[data-tick]");
  if (tick) {
    const p = pullById(Number(tick.dataset.tick));
    jumpTo(p);
    render();
    return seek(p.time - 1);
  }
  const r = e.currentTarget.getBoundingClientRect();
  seek(((e.clientX - r.left) / r.width) * videoLength());
});

$("#chapters").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-chapter]");
  if (!btn) return;
  const p = state.data.detection.packs.find((x) => x.index === Number(btn.dataset.chapter));
  state.pack = p.index;
  state.selected = 0;
  render();
  seek(p.clip_start, p.clip_end);
});

$("#focus").addEventListener("click", (e) => {
  const p = currentPull();
  if (!p) return;
  const pick = e.target.closest("button[data-choice]");
  if (pick) return choose(p, pick.dataset.choice);
  const btn = e.target.closest("button[data-focus]");
  if (!btn) return;
  if (btn.dataset.focus === "reject" && p.status !== "rejected") {
    patch(p.id, { status: "rejected" }).then(() => { goNextPending(p.id); render(); });
  } else act(btn.dataset.focus, p.id);
});

$("#picker").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-pick]");
  if (btn && pickerDone) pickerDone(btn.dataset.pick);
});
$("#picker-search").addEventListener("input", (e) => renderResults(e.target.value));

$("#add-card").onclick = () => openPicker(null);
$("#confirm-pack").onclick = async () => {
  const r = await api(`/api/videos/${state.slug}/packs/${state.pack}/confirm`, { method: "POST" });
  await refresh();
  toast(`Confirmed ${r.confirmed} in pack ${state.pack}`);
  const next = firstPackNeedingWork();
  if (next) state.pack = next;
  render();
};
$("#play-clip").onclick = () => {
  const p = state.data.detection.packs.find((p) => p.index === state.pack);
  if (p) seek(p.clip_start, p.clip_end);
};
// Finishing the review also fixes which packs go into each post, so the pipeline
// only ever needs this one human stop.
$("#finish").onclick = async () => {
  const plan = await api(`/api/videos/${state.slug}/posts`);
  $("#posts-packs").innerHTML = Object.entries(plan.packs)
    .map(([pack, rarities]) => {
      const best = rarities.filter(isHit);
      return `<li><b>Pack ${pack}</b> ${best.length ? `<span class="badge hit">${best.join(", ")}</span>` : ""}</li>`;
    })
    .join("");
  $("#posts-spec").value = plan.spec;
  $("#posts-seconds").value = plan.seconds;
  $("#posts-note").textContent = plan.suggested
    ? "Suggested: packs with a big hit get their own post, the rest in threes."
    : "Your saved plan.";
  $("#posts").showModal();
  $("#posts-spec").focus();
};

$("#posts-form").addEventListener("submit", async (e) => {
  if (e.submitter && e.submitter.value !== "save") return;
  e.preventDefault();
  try {
    await api(`/api/videos/${state.slug}/posts`, {
      method: "PUT",
      body: { spec: $("#posts-spec").value, seconds: Number($("#posts-seconds").value) },
    });
    await api(`/api/videos/${state.slug}/complete`, { method: "POST" });
  } catch {
    return; // the error is already shown as a toast; keep the dialog open to fix it
  }
  $("#posts").close();
  await refresh();
  render();
  toast("Review finished — the pipeline takes it from here");
});

document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, select, dialog") || e.ctrlKey || e.metaKey || e.altKey) return;
  const pulls = state.data ? packPulls() : [];
  const current = pulls[state.selected];
  const keys = { c: "confirm", x: "reject", r: "reverse", e: "change", " ": "seek" };
  const views = { p: "pack", d: "card", v: "video" };
  if (views[e.key]) { e.preventDefault(); return setView(views[e.key]); }
  if (e.key === "j" || e.key === "ArrowRight") state.selected = Math.min(state.selected + 1, pulls.length - 1);
  else if (e.key === "k" || e.key === "ArrowLeft") state.selected = Math.max(state.selected - 1, 0);
  else if (/^[1-9]$/.test(e.key) && current) {
    const det = current.detected_index != null ? state.data.detection.pulls[current.detected_index] : null;
    const id = choicesFor(current, det)[Number(e.key) - 1];
    if (!id) return;
    e.preventDefault();
    choose(current, id);
    return;
  } else if (keys[e.key] && current) {
    e.preventDefault();
    // Keys set a status outright (no toggling), so mashing "c" down a pack is safe.
    const direct = { c: "confirmed", x: "rejected" }[e.key];
    const done = direct ? patch(current.id, { status: direct }) : act(keys[e.key], current.id);
    done.then(() => {
      // A pending card jumps to the next one that needs you; a settled one steps on.
      if (direct && current.status === "pending") goNextPending(current.id);
      else if (direct) state.selected = Math.min(state.selected + 1, pulls.length - 1);
      render();
    });
    return;
  } else return;
  e.preventDefault();
  render();
  document.querySelector(".pull.selected")?.scrollIntoView({ block: "nearest" });
});

init();
