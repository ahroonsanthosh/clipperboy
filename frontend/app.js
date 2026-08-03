/* Clipperboy front-end. No framework, no build step. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const el = {
    uploadSection: $("upload-section"),
    form: $("upload-form"),
    dropzone: $("dropzone"),
    fileInput: $("file-input"),
    fileChosen: $("file-chosen"),
    startBtn: $("start-btn"),
    startHint: $("start-hint"),
    uploadError: $("upload-error"),
    aspect: $("aspect"),
    clipCount: $("clip-count"),
    compress: $("compress"),
    captions: $("captions"),
    speed: $("speed"),
    quality: $("quality"),

    progressSection: $("progress-section"),
    progressHeading: $("progress-heading"),
    progressFile: $("progress-file"),
    progressBar: $("progress-bar"),
    progressPct: $("progress-pct"),
    progressMessage: $("progress-message"),
    stageList: $("stage-list"),
    cancelBtn: $("cancel-btn"),
    jobError: $("job-error"),

    resultsSection: $("results-section"),
    resultsHeading: $("results-heading"),
    resultsSummary: $("results-summary"),
    clipGrid: $("clip-grid"),
    downloadAll: $("download-all"),
    downloadTranscript: $("download-transcript"),
    newUpload: $("new-upload"),

    historySection: $("history-section"),
    historyList: $("history-list"),

    dialog: $("preview-dialog"),
    dialogTitle: $("preview-title"),
    dialogVideo: $("preview-video"),
    dialogCaption: $("preview-caption"),
    dialogTranscript: $("preview-transcript"),
    dialogClose: $("preview-close"),

    marqueeTrack: $("marquee-track"),
    navUpload: $("nav-upload"),
    navProgress: $("nav-progress"),
    navResults: $("nav-results"),
    railNew: $("rail-new"),
    stripPrev: $("strip-prev"),
    stripNext: $("strip-next"),
  };

  let selectedFile = null;
  let currentJobId = null;
  let pollTimer = null;
  let lastSignature = "";
  let announcedDone = false;
  let pollFailures = 0;
  let jobStartedAt = 0;

  // ---------- helpers ----------
  const fmtTime = (s) => {
    s = Math.max(0, Math.round(s));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`
             : `${m}:${String(sec).padStart(2, "0")}`;
  };
  const fmtSize = (mb) => (mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`);

  // How each clip was framed, in words a creator actually uses.
  const CROP_LABEL = {
    "speaker-tracked": "follows the speaker",
    "centred": "centre of frame",
    "full-frame": "whole frame",
    "audio-waveform": "audio, with waveform",
  };

  /** Turn engine-speak into something a human can act on. */
  function friendlyError(raw) {
    const text = String(raw || "");
    const low = text.toLowerCase();
    if (low.includes("cancelled")) return "You cancelled this episode.";
    if (low.includes("speech model") || low.includes("huggingface")) {
      return "The speech model couldn't be downloaded. Check your internet connection — " +
             "this is only needed the first time you run Clipperboy.";
    }
    if (low.includes("no speech")) return "No speech was found in this file.";
    if (low.includes("memory") || low.includes("allocate")) {
      return "Your computer ran short of memory. Close other apps and try again.";
    }
    if (low.includes("duration") || low.includes("valid media")) {
      return "That file couldn't be read as audio or video. Try exporting it again as MP4 or MP3.";
    }
    if (low.includes("ffmpeg") || low.includes("failed:")) {
      return "Something went wrong while editing this clip. Trying again usually fixes it.";
    }
    return text || "Something went wrong.";
  }

  function showError(node, message) {
    node.hidden = false;                 // reveal first, so role="alert" fires on the text change
    requestAnimationFrame(() => { node.textContent = message; });
  }
  function clearError(node) {
    node.textContent = "";
    node.hidden = true;
  }

  async function api(path, options) {
    const res = await fetch(path, options);
    if (!res.ok) {
      let detail = `Request failed (${res.status})`;
      try {
        const body = await res.json();
        if (body && body.detail) detail = body.detail;
      } catch (_) { /* non-JSON error body */ }
      throw new Error(detail);
    }
    return res.json();
  }

  // ---------- chrome: rail + marquee ----------

  /** The rail is the app's state display: the live step is a full colour block,
      the ones you can't reach yet are outlined and disabled. */
  function syncRail() {
    const steps = [
      [el.navUpload, el.uploadSection],
      [el.navProgress, el.progressSection],
      [el.navResults, el.resultsSection],
    ];
    // "current" = the furthest visible step, which is the one you're looking at.
    let current = null;
    steps.forEach(([, section]) => { if (!section.hidden) current = section; });
    steps.forEach(([btn, section]) => {
      btn.disabled = section.hidden;
      if (section === current) btn.setAttribute("aria-current", "step");
      else btn.removeAttribute("aria-current");
    });
  }

  function focusSection(section, heading) {
    if (section.hidden) return;
    section.scrollIntoView({ block: "start" });
    heading.focus();
  }
  el.navUpload.addEventListener("click", () => focusSection(el.uploadSection, document.getElementById("upload-heading")));
  el.navProgress.addEventListener("click", () => focusSection(el.progressSection, el.progressHeading));
  el.navResults.addEventListener("click", () => focusSection(el.resultsSection, el.resultsHeading));

  /** The ticker says what the app is actually doing — never marketing filler.
      It is aria-hidden; every fact in it is also in a live region or heading. */
  function setMarquee(items) {
    const list = items.filter(Boolean);
    if (!list.length) return;
    el.marqueeTrack.textContent = "";
    // Two identical runs, so the -50% keyframe loops seamlessly.
    for (let copy = 0; copy < 2; copy += 1) {
      list.forEach((text, i) => {
        const item = document.createElement("span");
        item.textContent = text;
        const sep = document.createElement("span");
        sep.className = "sep";
        sep.textContent = i % 2 ? "◆" : "⚡";
        el.marqueeTrack.append(item, sep);
      });
    }
  }

  const IDLE_MARQUEE = [
    "Local only — nothing leaves this computer",
    "Whisper transcribes, FFmpeg edits",
    "Drop an episode to begin",
    "Vertical, square or wide",
    "Word-by-word burned-in captions",
  ];

  function marqueeForJob(job) {
    if (!job) return setMarquee(IDLE_MARQUEE);
    const ready = job.clips.filter((c) => c.status === "ready").length;
    if (job.status === "done") {
      const shapes = [...new Set(job.clips.map((c) => c.aspect))].join(" / ");
      return setMarquee([
        `${ready} clip${ready === 1 ? "" : "s"} ready`,
        job.filename,
        `${fmtTime(job.duration)} episode`,
        shapes ? `exported ${shapes}` : "",
        "Download all as a zip",
      ]);
    }
    if (job.status === "failed") {
      return setMarquee([`${job.filename} stopped`, friendlyError(job.error), "Try another file"]);
    }
    return setMarquee([
      `${job.stage_label}`,
      job.message || "Working",
      `${Math.round((job.progress || 0) * 100)}% through ${job.filename}`,
      job.clips.length ? `${ready} of ${job.clips.length} clips cut` : "Looking for moments",
    ]);
  }

  // ---------- file selection ----------
  function chooseFile(file) {
    if (!file) return;
    selectedFile = file;
    el.fileChosen.hidden = false;
    el.fileChosen.textContent = `Selected: ${file.name} (${fmtSize(file.size / 1024 / 1024)})`;
    el.startHint.hidden = true;
    clearError(el.uploadError);
  }

  el.fileInput.addEventListener("change", () => chooseFile(el.fileInput.files[0]));

  // The <label> is the keyboard-reachable control, so it needs key handling.
  document.querySelector(".dropzone-browse").addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      el.fileInput.click();
    }
  });

  ["dragenter", "dragover"].forEach((evt) =>
    el.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      el.dropzone.classList.add("is-dragging");
    }));
  ["dragleave", "drop"].forEach((evt) =>
    el.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      el.dropzone.classList.remove("is-dragging");
    }));
  el.dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) {
      el.fileInput.files = e.dataTransfer.files;
      chooseFile(file);
    }
  });

  // ---------- submit ----------
  el.form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      showError(el.uploadError, "Choose a podcast file first — drag one in, or use “Browse your files”.");
      document.querySelector(".dropzone-browse").focus();
      return;
    }

    clearError(el.uploadError);
    el.startBtn.disabled = true;
    el.startBtn.textContent = "Uploading…";

    const body = new FormData();
    body.append("file", selectedFile);
    body.append("aspect", el.aspect.value);
    body.append("clip_count", el.clipCount.value);
    body.append("compress", el.compress.value);
    body.append("captions", el.captions.checked ? "true" : "false");
    body.append("speed", el.speed.value);
    body.append("quality", el.quality.value);

    try {
      const { job } = await api("/api/jobs", { method: "POST", body });
      currentJobId = job.id;
      announcedDone = false;
      pollFailures = 0;
      jobStartedAt = Date.now();
      lastSignature = "";
      el.uploadSection.hidden = true;
      el.progressSection.hidden = false;
      el.resultsSection.hidden = true;
      el.progressFile.textContent = job.filename;
      renderProgress(job);
      el.progressHeading.focus();
      poll();
    } catch (err) {
      showError(el.uploadError, friendlyError(err.message));
      el.startBtn.disabled = false;
    } finally {
      el.startBtn.textContent = "Find my clips";
    }
  });

  el.cancelBtn.addEventListener("click", async () => {
    if (!currentJobId) return resetToUpload();
    if (!confirm("Stop working on this episode and discard its clips?")) return;
    try { await api(`/api/jobs/${currentJobId}`, { method: "DELETE" }); } catch (_) {}
    resetToUpload();
  });

  el.newUpload.addEventListener("click", resetToUpload);
  el.railNew.addEventListener("click", resetToUpload);

  // Horizontal filmstrip: buttons for the mouse, native arrow keys / scrollbar
  // for the keyboard (the <ul> is a focusable region).
  const stripStep = () => Math.max(240, Math.round(el.clipGrid.clientWidth * 0.8));
  el.stripPrev.addEventListener("click", () => el.clipGrid.scrollBy({ left: -stripStep(), behavior: "smooth" }));
  el.stripNext.addEventListener("click", () => el.clipGrid.scrollBy({ left: stripStep(), behavior: "smooth" }));

  function resetToUpload() {
    clearTimeout(pollTimer);
    currentJobId = null;
    selectedFile = null;
    pollFailures = 0;
    el.fileInput.value = "";
    el.fileChosen.hidden = true;
    el.startHint.hidden = false;
    el.startBtn.disabled = false;
    clearError(el.uploadError);
    clearError(el.jobError);
    el.uploadSection.hidden = false;
    el.progressSection.hidden = true;
    el.resultsSection.hidden = true;
    lastSignature = "";
    syncRail();
    marqueeForJob(null);
    loadHistory();
    document.querySelector(".dropzone-browse").focus();
  }

  // ---------- polling ----------
  async function poll() {
    if (!currentJobId) return;
    try {
      const { job } = await api(`/api/jobs/${currentJobId}`);
      renderProgress(job);
      renderResults(job);

      if (job.status === "failed") {
        el.cancelBtn.textContent = "Start over";
        showError(el.jobError, friendlyError(job.error));
        return;
      }
      if (job.status === "done") {
        el.progressSection.hidden = true;
        if (!announcedDone) {
          announcedDone = true;
          el.resultsHeading.focus();
        }
        loadHistory();
        return;
      }
      // Contact is good — clear any earlier "can't reach" warning.
      if (pollFailures) {
        pollFailures = 0;
        clearError(el.jobError);
      }
    } catch (err) {
      pollFailures += 1;
      // A single missed poll means nothing — a laptop waking, a browser
      // throttling a background tab. Only say something once it persists.
      if (pollFailures === 4) {
        showError(el.jobError,
          "Can't reach Clipperboy at the moment — still trying. Your episode keeps " +
          "processing as long as the Clipperboy window is still open.");
      } else if (pollFailures === 20) {
        showError(el.jobError,
          "Clipperboy still isn't responding. Check the window it started in: if it " +
          "has closed or was stopped, reopen it and upload the episode again. If it " +
          "is still running, leave this page open — it will reconnect on its own.");
      }
    }
    // Back off while unreachable so a stopped app isn't polled twice a second.
    const delay = pollFailures > 4 ? Math.min(15000, 1500 * pollFailures) : 1500;
    pollTimer = setTimeout(poll, delay);
  }

  /** Time left, measured from this run's own rate rather than guessed from the
      machine. Only shown once there's enough progress for it to mean anything. */
  function etaText(job) {
    if (!jobStartedAt || !job.progress || job.progress < 0.06) return "";
    if (job.status === "done" || job.status === "failed") return "";
    const elapsed = (Date.now() - jobStartedAt) / 1000;
    const remaining = elapsed / job.progress - elapsed;
    if (!isFinite(remaining) || remaining < 20) return "";
    const mins = Math.round(remaining / 60);
    return mins < 1 ? "under a minute left"
         : mins === 1 ? "about a minute left"
         : `about ${mins} minutes left`;
  }

  function renderProgress(job) {
    const pct = Math.round((job.progress || 0) * 100);
    el.progressBar.style.width = `${pct}%`;
    el.progressBar.setAttribute("aria-valuenow", String(pct));
    el.progressPct.textContent = `${pct}%`;
    const eta = etaText(job);
    el.progressMessage.textContent =
      (job.message || job.stage_label || "Working…") + (eta ? ` · ${eta}` : "");

    const order = job.stages.map((s) => s.key);
    const activeIndex = order.indexOf(job.stage);
    el.stageList.innerHTML = "";
    job.stages.forEach((stage, i) => {
      const li = document.createElement("li");
      const done = i < activeIndex || job.status === "done";
      const active = i === activeIndex && job.status !== "done";
      li.className = done ? "is-done" : active ? "is-active" : "";
      const marker = document.createElement("span");
      marker.className = "marker";
      marker.setAttribute("aria-hidden", "true");
      marker.textContent = done ? "✓" : active ? "▶" : "·";
      li.append(marker, document.createTextNode(stage.label));
      if (done) li.append(document.createTextNode(" (done)"));
      el.stageList.append(li);
    });

    marqueeForJob(job);
    syncRail();
  }

  // ---------- results ----------
  function renderResults(job) {
    if (!job.clips.length) {
      // Nothing to show yet — but a *finished* job with no clips must not leave
      // the user staring at a blank page.
      el.resultsSection.hidden = job.status !== "done";
      if (job.status === "done") {
        el.resultsSummary.textContent =
          "No clips were produced from this episode. It may be too short, or " +
          "contain no clear speech. Try another file.";
        el.clipGrid.innerHTML = "";
        el.downloadAll.setAttribute("aria-disabled", "true");
        el.downloadAll.href = "#";
        el.downloadTranscript.hidden = !job.transcript_available;
      }
      syncRail();
      return;
    }
    el.resultsSection.hidden = false;
    syncRail();

    const signature = JSON.stringify(
      job.clips.map((c) => [c.id, c.status, c.aspect, c.title, c.error]));
    if (signature === lastSignature) return;
    lastSignature = signature;

    const ready = job.clips.filter((c) => c.status === "ready").length;
    const source = job.compressed
      ? `${fmtSize(job.source_size_mb)} → ${fmtSize(job.working_size_mb)} after shrinking`
      : fmtSize(job.source_size_mb);
    el.resultsSummary.textContent =
      `${ready} of ${job.clips.length} clips ready · from a ${fmtTime(job.duration)} ` +
      `${job.has_video ? "video" : "audio"} episode · ${source}`;

    const noneReady = ready === 0;
    el.downloadAll.href = noneReady ? "#" : `/api/jobs/${job.id}/download-all`;
    el.downloadAll.setAttribute("aria-disabled", String(noneReady));
    el.downloadTranscript.href = `/api/jobs/${job.id}/transcript.txt`;
    el.downloadTranscript.hidden = !job.transcript_available;

    // Remember what had focus / was expanded, so a background refresh doesn't
    // yank the page out from under someone mid-read.
    const activeId = document.activeElement ? document.activeElement.id : "";
    const scrollY = window.scrollY;
    const scrollX = el.clipGrid.scrollLeft;
    el.clipGrid.innerHTML = "";
    job.clips.forEach((clip) => el.clipGrid.append(buildClipCard(job, clip)));
    el.clipGrid.scrollLeft = scrollX;   // a background refresh must not rewind the strip

    if (activeId) {
      const restored = document.getElementById(activeId);
      if (restored) restored.focus({ preventScroll: true });
    }
    window.scrollTo({ top: scrollY, behavior: "auto" });
  }

  el.downloadAll.addEventListener("click", (e) => {
    if (el.downloadAll.getAttribute("aria-disabled") === "true") e.preventDefault();
  });

  function buildClipCard(job, clip) {
    const li = document.createElement("li");
    li.className = "clip-card";
    const uid = `${job.id}-${clip.id}`;

    // Every card gets a fixed-height header block whatever the clip's shape or
    // state, so the strip never turns into a ragged row of odd heights.
    if (clip.status === "ready") {
      const thumbBtn = document.createElement("button");
      thumbBtn.type = "button";
      thumbBtn.id = `play-${uid}`;
      thumbBtn.className = "clip-thumb-btn";
      thumbBtn.setAttribute("aria-label", `Play preview of clip ${clip.index}: ${clip.title}`);
      const img = document.createElement("img");
      img.src = clip.thumbnail;
      img.alt = "";
      img.loading = "lazy";
      const index = document.createElement("span");
      index.className = "clip-index";
      index.setAttribute("aria-hidden", "true");
      index.textContent = String(clip.index).padStart(2, "0");
      const aspect = document.createElement("span");
      aspect.className = "clip-aspect";
      aspect.setAttribute("aria-hidden", "true");
      aspect.textContent = clip.aspect;
      const badge = document.createElement("span");
      badge.className = "play-badge";
      badge.setAttribute("aria-hidden", "true");
      badge.textContent = "▶";
      thumbBtn.append(img, index, aspect, badge);
      thumbBtn.addEventListener("click", () => openPreview(clip));
      li.append(thumbBtn);
    } else {
      const slab = document.createElement("div");
      slab.className = `clip-slab is-${clip.status}`;
      slab.setAttribute("aria-hidden", "true");
      const slabIndex = document.createElement("span");
      slabIndex.className = "clip-index";
      slabIndex.textContent = String(clip.index).padStart(2, "0");
      slab.append(slabIndex);
      slab.append(
        clip.status === "failed" ? "Didn't render"
        : clip.status === "rendering" ? "Editing now…"
        : "Queued");
      li.append(slab);
    }

    const body = document.createElement("div");
    body.className = "clip-body";
    const foot = document.createElement("div");
    foot.className = "clip-foot";

    const h3 = document.createElement("h3");
    h3.className = "clip-title";
    h3.textContent = clip.title;   // the index lives in the badge on the poster
    body.append(h3);

    const meta = document.createElement("p");
    meta.className = "clip-meta";
    [
      `${Math.round(clip.duration)} seconds`,
      `at ${fmtTime(clip.start)} in the episode`,
      clip.aspect,
      clip.captions ? "captions burned in" : "no burned-in captions",
      CROP_LABEL[clip.crop_mode] || "",
    ].filter(Boolean).forEach((bit) => {
      const span = document.createElement("span");
      span.textContent = bit;
      meta.append(span);
    });
    body.append(meta);

    if (clip.summary) {
      const p = document.createElement("p");
      p.className = "clip-summary";
      p.textContent = clip.summary;
      body.append(p);
    }

    if (clip.reasons.length) {
      const ul = document.createElement("ul");
      ul.className = "chips";
      ul.setAttribute("aria-label", `Why clip ${clip.index} was picked`);
      clip.reasons.forEach((r) => {
        const chip = document.createElement("li");
        chip.className = "chip";
        chip.textContent = r;
        ul.append(chip);
      });
      body.append(ul);
    }

    if (clip.keywords.length) {
      const ul = document.createElement("ul");
      ul.className = "chips";
      ul.setAttribute("aria-label", `Topics in clip ${clip.index}`);
      clip.keywords.slice(0, 4).forEach((k) => {
        const chip = document.createElement("li");
        chip.className = "chip chip-key";
        chip.textContent = k;
        ul.append(chip);
      });
      body.append(ul);
    }

    if (clip.status === "ready") {
      const actions = document.createElement("div");
      actions.className = "clip-actions";

      const dl = document.createElement("a");
      dl.className = "pillbtn";
      dl.id = `dl-${uid}`;
      dl.href = `${clip.video}?download=true`;
      dl.setAttribute("download", "");
      dl.textContent = "Download";
      dl.setAttribute("aria-label", `Download clip ${clip.index}: ${clip.title}`);

      const srt = document.createElement("a");
      srt.className = "pillbtn pillbtn-quiet";
      srt.href = clip.srt;
      srt.setAttribute("download", "");
      srt.textContent = "Subtitles";
      srt.setAttribute("aria-label", `Download the subtitle file for clip ${clip.index}`);

      actions.append(dl, srt);
      foot.append(actions, buildReexport(job, clip, uid));
    } else {
      const status = document.createElement("p");
      status.className = `clip-status is-${clip.status}`;
      status.setAttribute("role", "status");
      status.textContent =
        clip.status === "failed" ? friendlyError(clip.error)
        : clip.status === "rendering" ? "Editing this clip now…"
        : "Waiting to be edited…";
      body.append(status);

      if (clip.status === "failed") {
        const retry = document.createElement("button");
        retry.type = "button";
        retry.id = `retry-${uid}`;
        retry.className = "pillbtn pillbtn-quiet";
        retry.textContent = "Try this clip again";
        retry.addEventListener("click", () => runRender(job, clip, clip.aspect, retry, "Trying again…"));
        foot.append(retry);
      }
    }

    // The transcript lives in the preview dialog: you read it when deciding about
    // one clip, not while scanning the strip. Keeping it out of the card is what
    // lets every card fit its content with nothing cut and nothing to scroll.
    if (clip.transcript) {
      const read = document.createElement("button");
      read.type = "button";
      read.id = `text-${uid}`;
      read.className = "clip-readbtn";
      read.textContent = "What's said in this clip";
      read.setAttribute("aria-label", `Read what's said in clip ${clip.index}: ${clip.title}`);
      read.addEventListener("click", () => openPreview(clip));
      body.append(read);
    }

    li.append(body);
    if (foot.childElementCount) li.append(foot);
    return li;
  }

  function buildReexport(job, clip, uid) {
    const wrap = document.createElement("div");
    wrap.className = "reexport";

    const selectId = `aspect-${uid}`;
    const label = document.createElement("label");
    label.setAttribute("for", selectId);
    label.textContent = "Also export as";

    const select = document.createElement("select");
    select.id = selectId;
    [["9:16", "Tall 9:16"], ["1:1", "Square 1:1"], ["16:9", "Wide 16:9"]].forEach(([value, text]) => {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = text;
      if (value === clip.aspect) opt.selected = true;
      select.append(opt);
    });

    const go = document.createElement("button");
    go.type = "button";
    go.id = `go-${uid}`;
    go.className = "pillbtn pillbtn-quiet";
    go.textContent = "Re-export";

    const status = document.createElement("span");
    status.setAttribute("role", "status");

    go.addEventListener("click", () => {
      if (select.value === clip.aspect) {
        status.textContent = "Already this shape.";
        return;
      }
      runRender(job, clip, select.value, go, "Re-exporting…", status);
    });

    wrap.append(label, select, go, status);
    return wrap;
  }

  async function runRender(job, clip, aspect, button, busyLabel, status) {
    const original = button.textContent;
    button.disabled = true;
    button.textContent = busyLabel;
    if (status) status.textContent = "Working — this takes a few seconds.";
    try {
      await api(`/api/jobs/${job.id}/clips/${clip.id}/render`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ aspect, captions: clip.captions }),
      });
      lastSignature = "";
      const { job: fresh } = await api(`/api/jobs/${job.id}`);
      renderResults(fresh);
    } catch (err) {
      button.disabled = false;
      button.textContent = original;
      if (status) status.textContent = friendlyError(err.message);
      else showError(el.jobError, friendlyError(err.message));
    }
  }

  // ---------- preview dialog ----------
  let lastFocused = null;

  function openPreview(clip) {
    lastFocused = document.activeElement;
    const playable = clip.status === "ready" && Boolean(clip.video);
    el.dialogTitle.textContent = `${clip.index}. ${clip.title}`;
    el.dialog.dataset.aspect = clip.aspect;
    el.dialogVideo.hidden = !playable;          // a queued clip has text but no film yet
    if (playable) el.dialogVideo.src = clip.video;
    el.dialogCaption.textContent =
      `${Math.round(clip.duration)} seconds · ${clip.aspect} · from ${fmtTime(clip.start)} in the episode`;
    el.dialogTranscript.textContent = clip.transcript;
    el.dialogTranscript.scrollTop = 0;
    if (typeof el.dialog.showModal === "function") el.dialog.showModal();
    else el.dialog.setAttribute("open", "");
    el.dialogClose.focus();
    if (playable) el.dialogVideo.play().catch(() => { /* autoplay blocked — controls are there */ });
  }

  function teardownPreview() {
    el.dialogVideo.pause();
    el.dialogVideo.removeAttribute("src");
    el.dialogVideo.load();
    if (lastFocused && document.contains(lastFocused)) lastFocused.focus();
  }

  el.dialogClose.addEventListener("click", () => {
    if (typeof el.dialog.close === "function") el.dialog.close();
    else { el.dialog.removeAttribute("open"); teardownPreview(); }
  });
  el.dialog.addEventListener("close", teardownPreview);
  el.dialog.addEventListener("click", (e) => {
    if (e.target === el.dialog && typeof el.dialog.close === "function") el.dialog.close();
  });
  // Keep Tab inside the dialog rather than letting it escape to the page behind.
  el.dialog.addEventListener("keydown", (e) => {
    if (e.key !== "Tab") return;
    const stops = [el.dialogClose, el.dialogVideo, el.dialogTranscript].filter((n) => !n.hidden);
    const i = stops.indexOf(document.activeElement);
    e.preventDefault();
    stops[(i + (e.shiftKey ? stops.length - 1 : 1) + stops.length) % stops.length].focus();
  });

  // ---------- history ----------
  async function loadHistory() {
    try {
      const { jobs: all } = await api("/api/jobs");
      const others = all.filter((j) => j.id !== currentJobId);
      el.historySection.hidden = others.length === 0;
      el.historyList.innerHTML = "";

      others.forEach((job) => {
        const li = document.createElement("li");
        const info = document.createElement("div");
        const name = document.createElement("strong");
        name.textContent = job.filename;
        const meta = document.createElement("div");
        meta.className = "history-meta";
        const readyCount = job.clips.filter((c) => c.status === "ready").length;
        meta.textContent = job.status === "done"
          ? `${readyCount} clips · ${fmtTime(job.duration)} episode`
          : job.status === "failed" ? friendlyError(job.error) : `${job.stage_label}…`;
        info.append(name, meta);

        const actions = document.createElement("div");
        actions.className = "history-actions";

        const open = document.createElement("button");
        open.type = "button";
        open.className = "pillbtn pillbtn-quiet";
        open.textContent = "Open";
        open.setAttribute("aria-label", `Open ${job.filename}`);
        open.addEventListener("click", async () => {
          clearTimeout(pollTimer);
          currentJobId = job.id;
          lastSignature = "";
          announcedDone = job.status === "done";
          clearError(el.jobError);
          el.uploadSection.hidden = true;
          el.progressFile.textContent = job.filename;
          const { job: fresh } = await api(`/api/jobs/${job.id}`);
          el.progressSection.hidden = fresh.status === "done";
          renderProgress(fresh);
          renderResults(fresh);
          loadHistory();
          if (fresh.status !== "done" && fresh.status !== "failed") poll();
          (fresh.status === "done" ? el.resultsHeading : el.progressHeading).focus();
        });

        const del = document.createElement("button");
        del.type = "button";
        del.className = "pillbtn pillbtn-quiet";
        del.textContent = "Delete";
        del.setAttribute("aria-label", `Delete ${job.filename} and its clips`);
        del.addEventListener("click", async () => {
          if (!confirm(`Delete "${job.filename}" and all of its clips?`)) return;
          await api(`/api/jobs/${job.id}`, { method: "DELETE" });
          loadHistory();
        });

        actions.append(open, del);
        li.append(info, actions);
        el.historyList.append(li);
      });
    } catch (_) {
      el.historySection.hidden = true;
    }
  }

  syncRail();
  marqueeForJob(null);
  loadHistory();
})();
