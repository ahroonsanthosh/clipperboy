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
    uploadError: $("upload-error"),
    aspect: $("aspect"),
    clipCount: $("clip-count"),
    compress: $("compress"),

    progressSection: $("progress-section"),
    progressFile: $("progress-file"),
    progressBar: $("progress-bar"),
    progressPct: $("progress-pct"),
    progressMessage: $("progress-message"),
    stageList: $("stage-list"),
    cancelBtn: $("cancel-btn"),
    jobError: $("job-error"),

    resultsSection: $("results-section"),
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
    dialogClose: $("preview-close"),
  };

  let selectedFile = null;
  let currentJobId = null;
  let pollTimer = null;
  let lastRenderedSignature = "";

  // ---------- helpers ----------
  const fmtTime = (s) => {
    s = Math.max(0, Math.round(s));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`
             : `${m}:${String(sec).padStart(2, "0")}`;
  };
  const fmtSize = (mb) => (mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`);

  function showError(node, message) {
    node.textContent = message;
    node.hidden = false;
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

  // ---------- file selection ----------
  function chooseFile(file) {
    if (!file) return;
    selectedFile = file;
    el.fileChosen.hidden = false;
    el.fileChosen.textContent = `Selected: ${file.name} (${fmtSize(file.size / 1024 / 1024)})`;
    el.startBtn.disabled = false;
    clearError(el.uploadError);
  }

  el.dropzone.addEventListener("click", () => el.fileInput.click());
  el.dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.fileInput.click(); }
  });
  el.fileInput.addEventListener("change", () => chooseFile(el.fileInput.files[0]));

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
    if (file) chooseFile(file);
  });

  // ---------- submit ----------
  el.form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!selectedFile) return;

    clearError(el.uploadError);
    el.startBtn.disabled = true;
    el.startBtn.textContent = "Uploading…";

    const body = new FormData();
    body.append("file", selectedFile);
    body.append("aspect", el.aspect.value);
    body.append("clip_count", el.clipCount.value);
    body.append("compress", el.compress.value);

    try {
      const { job } = await api("/api/jobs", { method: "POST", body });
      currentJobId = job.id;
      el.uploadSection.hidden = true;
      el.progressSection.hidden = false;
      el.resultsSection.hidden = true;
      el.progressFile.textContent = job.filename;
      lastRenderedSignature = "";
      renderProgress(job);
      el.progressSection.scrollIntoView({ block: "start" });
      poll();
    } catch (err) {
      showError(el.uploadError, err.message);
      el.startBtn.disabled = false;
    } finally {
      el.startBtn.textContent = "Find my clips";
    }
  });

  el.cancelBtn.addEventListener("click", async () => {
    if (!currentJobId) return resetToUpload();
    if (!confirm("Stop working on this episode and discard it?")) return;
    try { await api(`/api/jobs/${currentJobId}`, { method: "DELETE" }); } catch (_) {}
    resetToUpload();
  });

  el.newUpload.addEventListener("click", resetToUpload);

  function resetToUpload() {
    clearTimeout(pollTimer);
    currentJobId = null;
    selectedFile = null;
    el.fileInput.value = "";
    el.fileChosen.hidden = true;
    el.startBtn.disabled = true;
    clearError(el.uploadError);
    clearError(el.jobError);
    el.uploadSection.hidden = false;
    el.progressSection.hidden = true;
    el.resultsSection.hidden = true;
    loadHistory();
    el.uploadSection.scrollIntoView({ block: "start" });
    el.dropzone.focus();
  }

  // ---------- polling ----------
  async function poll() {
    if (!currentJobId) return;
    try {
      const { job } = await api(`/api/jobs/${currentJobId}`);
      renderProgress(job);

      if (job.status === "failed") {
        showError(el.jobError, job.error || "Processing failed.");
        return;
      }
      if (job.clips.length) renderResults(job);
      if (job.status === "done") { loadHistory(); return; }
    } catch (err) {
      showError(el.jobError, `Lost contact with the app: ${err.message}`);
    }
    pollTimer = setTimeout(poll, 1500);
  }

  function renderProgress(job) {
    const pct = Math.round((job.progress || 0) * 100);
    el.progressBar.style.width = `${pct}%`;
    el.progressBar.setAttribute("aria-valuenow", String(pct));
    el.progressPct.textContent = `${pct}%`;
    el.progressMessage.textContent = job.message || job.stage_label || "Working…";

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
  }

  // ---------- results ----------
  function renderResults(job) {
    const signature = JSON.stringify(job.clips.map((c) => [c.id, c.status, c.aspect, c.title]));
    el.resultsSection.hidden = false;
    if (signature === lastRenderedSignature) return;
    lastRenderedSignature = signature;

    const ready = job.clips.filter((c) => c.status === "ready").length;
    const source = job.compressed
      ? `${fmtSize(job.source_size_mb)} → ${fmtSize(job.working_size_mb)} after compression`
      : fmtSize(job.source_size_mb);
    el.resultsSummary.textContent =
      `${ready} of ${job.clips.length} clips ready · from ${fmtTime(job.duration)} of ${job.has_video ? "video" : "audio"} · ${source}`;

    el.downloadAll.href = `/api/jobs/${job.id}/download-all`;
    const noneReady = ready === 0;
    el.downloadAll.setAttribute("aria-disabled", String(noneReady));
    el.downloadAll.classList.toggle("btn-disabled", noneReady);
    el.downloadTranscript.href = `/api/jobs/${job.id}/transcript`;
    el.downloadTranscript.hidden = !job.transcript_available;

    el.clipGrid.innerHTML = "";
    job.clips.forEach((clip) => el.clipGrid.append(buildClipCard(job, clip)));
  }

  function buildClipCard(job, clip) {
    const li = document.createElement("li");
    li.className = "clip-card";

    if (clip.status === "ready") {
      const thumbBtn = document.createElement("button");
      thumbBtn.type = "button";
      thumbBtn.className = "clip-thumb-btn";
      thumbBtn.setAttribute("aria-label", `Play preview of clip ${clip.index}: ${clip.title}`);
      const img = document.createElement("img");
      img.src = clip.thumbnail;
      img.alt = "";
      img.loading = "lazy";
      const badge = document.createElement("span");
      badge.className = "play-badge";
      badge.setAttribute("aria-hidden", "true");
      badge.textContent = "▶";
      thumbBtn.append(img, badge);
      thumbBtn.addEventListener("click", () => openPreview(clip));
      li.append(thumbBtn);
    }

    const body = document.createElement("div");
    body.className = "clip-body";

    const h3 = document.createElement("h3");
    h3.className = "clip-title";
    h3.textContent = `${clip.index}. ${clip.title}`;
    body.append(h3);

    const meta = document.createElement("p");
    meta.className = "clip-meta";
    const bits = [
      `${Math.round(clip.duration)}s`,
      `at ${fmtTime(clip.start)}–${fmtTime(clip.end)}`,
      clip.aspect,
    ];
    if (clip.crop_mode === "speaker-tracked") bits.push("speaker tracked");
    else if (clip.crop_mode === "centred") bits.push("centre crop");
    else if (clip.crop_mode === "audio-waveform") bits.push("waveform visual");
    bits.push(`score ${clip.score.toFixed(1)}`);
    bits.forEach((b) => {
      const span = document.createElement("span");
      span.textContent = b;
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
      ul.setAttribute("aria-label", "Why this moment was picked");
      clip.reasons.forEach((r) => {
        const chipLi = document.createElement("li");
        chipLi.className = "chip";
        chipLi.textContent = r;
        ul.append(chipLi);
      });
      body.append(ul);
    }

    if (clip.keywords.length) {
      const ul = document.createElement("ul");
      ul.className = "chips";
      ul.setAttribute("aria-label", "Topics mentioned");
      clip.keywords.slice(0, 5).forEach((k) => {
        const chipLi = document.createElement("li");
        chipLi.className = "chip chip-key";
        chipLi.textContent = k;
        ul.append(chipLi);
      });
      body.append(ul);
    }

    const details = document.createElement("details");
    details.className = "clip-transcript";
    const summary = document.createElement("summary");
    summary.textContent = "What's said in this clip";
    const transcript = document.createElement("p");
    transcript.textContent = clip.transcript;
    details.append(summary, transcript);
    body.append(details);

    const actions = document.createElement("div");
    actions.className = "clip-actions";

    if (clip.status === "ready") {
      const dl = document.createElement("a");
      dl.className = "btn btn-primary";
      dl.href = `${clip.video}?download=true`;
      dl.setAttribute("download", "");
      dl.textContent = "Download";
      dl.setAttribute("aria-label", `Download clip ${clip.index}: ${clip.title}`);

      const srt = document.createElement("a");
      srt.className = "btn btn-secondary";
      srt.href = clip.srt;
      srt.setAttribute("download", "");
      srt.textContent = "Captions (.srt)";
      srt.setAttribute("aria-label", `Download subtitle file for clip ${clip.index}`);

      actions.append(dl, srt);
      body.append(actions);

      const reexport = document.createElement("div");
      reexport.className = "reexport";
      const selectId = `aspect-${job.id}-${clip.id}`;
      const label = document.createElement("label");
      label.setAttribute("for", selectId);
      label.textContent = "Re-export as";
      const select = document.createElement("select");
      select.id = selectId;
      ["9:16", "1:1", "16:9"].forEach((a) => {
        const opt = document.createElement("option");
        opt.value = a;
        opt.textContent = a;
        if (a === clip.aspect) opt.selected = true;
        select.append(opt);
      });
      const go = document.createElement("button");
      go.type = "button";
      go.className = "btn btn-secondary";
      go.textContent = "Go";
      go.addEventListener("click", async () => {
        if (select.value === clip.aspect) return;
        go.disabled = true;
        go.textContent = "Rendering…";
        try {
          await api(`/api/jobs/${job.id}/clips/${clip.id}/render`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ aspect: select.value, captions: true }),
          });
          lastRenderedSignature = "";
          const { job: fresh } = await api(`/api/jobs/${job.id}`);
          renderResults(fresh);
        } catch (err) {
          go.textContent = "Failed";
          alert(`Could not re-export: ${err.message}`);
        } finally {
          go.disabled = false;
        }
      });
      reexport.append(label, select, go);
      body.append(reexport);
    } else {
      const status = document.createElement("p");
      status.className = `clip-status is-${clip.status}`;
      status.setAttribute("role", "status");
      status.textContent =
        clip.status === "failed" ? `Could not render this clip: ${clip.error}`
        : clip.status === "rendering" ? "Rendering this clip…"
        : "Queued for rendering…";
      body.append(status);
    }

    li.append(body);
    return li;
  }

  // ---------- preview dialog ----------
  let lastFocused = null;

  function openPreview(clip) {
    lastFocused = document.activeElement;
    el.dialogTitle.textContent = `${clip.index}. ${clip.title}`;
    el.dialogVideo.src = clip.video;
    el.dialogCaption.textContent =
      `${Math.round(clip.duration)}s · ${clip.aspect} · from ${fmtTime(clip.start)} in the episode`;
    if (typeof el.dialog.showModal === "function") el.dialog.showModal();
    else el.dialog.setAttribute("open", "");
    el.dialogVideo.play().catch(() => { /* autoplay blocked — controls are there */ });
  }

  function closePreview() {
    el.dialogVideo.pause();
    el.dialogVideo.removeAttribute("src");
    el.dialogVideo.load();
    if (el.dialog.open && typeof el.dialog.close === "function") el.dialog.close();
    else el.dialog.removeAttribute("open");
    if (lastFocused) lastFocused.focus();
  }

  el.dialogClose.addEventListener("click", closePreview);
  el.dialog.addEventListener("close", closePreview);
  el.dialog.addEventListener("click", (e) => {
    if (e.target === el.dialog) closePreview();   // click on the backdrop
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
          : job.status === "failed" ? `Failed: ${job.error}` : `${job.stage_label}…`;
        info.append(name, meta);

        const actions = document.createElement("div");
        actions.className = "history-actions";

        const open = document.createElement("button");
        open.type = "button";
        open.className = "btn btn-secondary";
        open.textContent = "Open";
        open.addEventListener("click", async () => {
          clearTimeout(pollTimer);
          currentJobId = job.id;
          lastRenderedSignature = "";
          el.uploadSection.hidden = true;
          el.progressSection.hidden = job.status === "done";
          el.progressFile.textContent = job.filename;
          const { job: fresh } = await api(`/api/jobs/${job.id}`);
          renderProgress(fresh);
          renderResults(fresh);
          loadHistory();
          if (fresh.status !== "done" && fresh.status !== "failed") poll();
          el.resultsSection.scrollIntoView({ block: "start" });
        });

        const del = document.createElement("button");
        del.type = "button";
        del.className = "btn btn-quiet";
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

  loadHistory();
})();
