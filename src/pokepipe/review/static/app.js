// Pull review page. Vanilla JS: state lives on the server (review.json); after each
// edit we refetch and re-render, which keeps the client trivially consistent.
"use strict";

const HIT_RARITIES = new Set([
  "Double Rare", "Illustration Rare", "Ultra Rare", "Special Illustration Rare", "Hyper Rare",
]);
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
  select.innerHTML = videos
    .map((v) => `<option value="${v.slug}">${v.video} ${v.completed ? "✓" : `(${v.pending} pending)`}</option>`)
    .join("");
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

function render() {
  const { review, packs, detection } = state.data;
  const done = review.pulls.filter((p) => p.status !== "pending").length;
  const kept = review.pulls.filter((p) => p.status === "confirmed").length;
  $("#progress").textContent = `${done}/${review.pulls.length} reviewed · ${kept} kept` +
    (review.completed ? " · finished ✓" : "");
  $("#finish").disabled = done < review.pulls.length || review.completed;

  $("#pack-list").innerHTML = packs.map((p) => {
    const cls = !p.complete ? "bad" : p.pending === 0 ? "done" : "";
    const meta = `${p.cards}/${p.expected}${p.pending ? ` · ${p.pending} left` : " ✓"}`;
    return `<li><button data-pack="${p.index}" aria-current="${p.index === state.pack}">
      Pack ${p.index}<span class="pack-meta ${cls}">${meta}</span></button></li>`;
  }).join("");

  const pack = packs.find((p) => p.index === state.pack);
  const detPack = detection.packs.find((p) => p.index === state.pack);
  $("#pack-title").textContent = pack
    ? `Pack ${pack.index} — ${pack.cards} of ${pack.expected} cards`
    : `Pack ${state.pack}`;
  $("#clip-label").textContent = detPack ? `Clip ${fmt(detPack.clip_start)} – ${fmt(detPack.clip_end)}` : "";
  $("#play-clip").disabled = !detPack;

  const pulls = packPulls();
  state.selected = Math.min(state.selected, Math.max(pulls.length - 1, 0));
  $("#pulls").innerHTML = pulls.map((p, i) => pullTile(p, i)).join("");
}

function pullTile(p, i) {
  const det = p.detected_index != null ? state.data.detection.pulls[p.detected_index] : null;
  const c = card(p.card_id);
  const badges = [];
  if (HIT_RARITIES.has(c.rarity)) badges.push(`<span class="badge hit">${c.rarity}</span>`);
  if (det && det.confidence < LOW_CONFIDENCE && p.status === "pending") badges.push(`<span class="badge low">check</span>`);
  if (det && p.card_id !== p.detected_card_id) badges.push(`<span class="badge fixed">was ${card(p.detected_card_id).name}</span>`);
  if (!det) badges.push(`<span class="badge added">added</span>`);
  if (p.reverse_holo) badges.push(`<span class="badge rev">reverse holo</span>`);
  const shot = det && det.thumbnail
    ? `<img src="/media/${state.slug}/${det.thumbnail}" alt="Frame from video" loading="lazy">`
    : `<div class="no-shot">added by hand</div>`;
  const conf = det ? Math.round(det.confidence * 100) : 100;
  return `<li class="pull ${p.status} ${i === state.selected ? "selected" : ""}" data-id="${p.id}" data-i="${i}">
    <div class="imgs">
      <figure>${shot}<figcaption>video</figcaption></figure>
      <figure><img src="${cardImg(p.card_id)}" alt="${c.name} reference" loading="lazy"><figcaption>#${c.number}</figcaption></figure>
    </div>
    <div class="name">${c.name}</div>
    <div class="sub">${c.rarity || ""} · ${fmt(p.time)}${det ? ` · ${conf}%` : ""} ${badges.join("")}</div>
    ${det ? `<div class="conf" title="Detection confidence ${conf}%"><i style="width:${conf}%"></i></div>` : ""}
    <div class="row">
      <button class="ok" data-act="confirm" aria-pressed="${p.status === "confirmed"}" title="Confirm (c)">✓ Keep</button>
      <button class="bad" data-act="reject" aria-pressed="${p.status === "rejected"}" title="Reject (x)">✗ Reject</button>
      <button class="rev" data-act="reverse" aria-pressed="${p.reverse_holo}" title="Reverse holo (r)">Rev</button>
      <button data-act="change" title="Change card (e)">Change</button>
      <button data-act="seek" title="Play from here (space)">▶</button>
      ${det ? "" : `<button class="bad" data-act="delete" title="Delete added card">Delete</button>`}
    </div>
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
});

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

$("#pulls").addEventListener("click", (e) => {
  const tile = e.target.closest(".pull");
  if (!tile) return;
  state.selected = Number(tile.dataset.i);
  const btn = e.target.closest("button[data-act]");
  if (btn) act(btn.dataset.act, Number(tile.dataset.id));
  else render();
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
$("#finish").onclick = async () => {
  await api(`/api/videos/${state.slug}/complete`, { method: "POST" });
  await refresh();
  render();
  toast("Review finished — ready for the next stage");
};

document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, select, dialog") || e.ctrlKey || e.metaKey || e.altKey) return;
  const pulls = state.data ? packPulls() : [];
  const current = pulls[state.selected];
  const keys = { c: "confirm", x: "reject", r: "reverse", e: "change", " ": "seek" };
  if (e.key === "j" || e.key === "ArrowRight") state.selected = Math.min(state.selected + 1, pulls.length - 1);
  else if (e.key === "k" || e.key === "ArrowLeft") state.selected = Math.max(state.selected - 1, 0);
  else if (keys[e.key] && current) {
    e.preventDefault();
    // Keys set a status outright (no toggling), so mashing "c" down a pack is safe.
    const direct = { c: "confirmed", x: "rejected" }[e.key];
    const done = direct ? patch(current.id, { status: direct }) : act(keys[e.key], current.id);
    done.then(() => {
      if (e.key === "c" || e.key === "x") state.selected = Math.min(state.selected + 1, pulls.length - 1);
      render();
    });
    return;
  } else return;
  e.preventDefault();
  render();
  document.querySelector(".pull.selected")?.scrollIntoView({ block: "nearest" });
});

init();
