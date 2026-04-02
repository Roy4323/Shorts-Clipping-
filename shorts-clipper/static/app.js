(() => {
  'use strict';

  // ── State ──────────────────────────────────────────────────────────
  let selectedVideo  = null;
  let selectedAudio  = null;
  let activeJobId    = null;   // currently being polled
  let viewJobId      = null;   // currently shown in detail view
  let pollTimer      = null;
  let shortsCount    = 3;      // how many shorts to generate

  const LS_KEY = 'sc_active_job';

  const STAGE_ORDER = ['downloading','transcribing','scoring','clipping','reframing','subtitles','done'];
  const STAGE_LABEL = {
    queued:'Queued', downloading:'Downloading', transcribing:'Transcript',
    scoring:'Scoring', clipping:'Clipping', reframing:'Reframing',
    subtitles:'Subtitles', done:'Done', failed:'Failed',
  };

  // ── DOM shortcuts ──────────────────────────────────────────────────
  const $  = id => document.getElementById(id);
  const qs = sel => document.querySelector(sel);

  // ── Tabs ───────────────────────────────────────────────────────────
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      $(`${btn.dataset.tab}-tab`).classList.add('active');
    });
  });

  // ── Shorts count selector ──────────────────────────────────────────
  document.querySelectorAll('.count-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.count-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      shortsCount = parseInt(btn.dataset.n, 10);
    });
  });

  // ── Drop zones ─────────────────────────────────────────────────────
  function setupDrop(zoneId, inputId, type) {
    const zone  = $(zoneId);
    const input = $(inputId);
    zone.addEventListener('click', () => input.click());
    input.addEventListener('change', () => input.files[0] && pick(input.files[0]));
    zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
      e.preventDefault(); zone.classList.remove('drag-over');
      if (e.dataTransfer.files[0]) pick(e.dataTransfer.files[0]);
    });
    function pick(file) {
      if (type === 'video') selectedVideo = file; else selectedAudio = file;
      zone.querySelector('.dz-label').textContent = file.name.length > 22 ? file.name.slice(0,20)+'…' : file.name;
      zone.querySelector('.dz-hint').textContent  = fmtBytes(file.size);
      zone.classList.add('has-file');
      if (selectedVideo) $('start-upload-btn').classList.remove('hidden');
    }
  }
  setupDrop('video-drop-zone', 'video-input', 'video');
  setupDrop('audio-drop-zone', 'audio-input', 'audio');

  // ── Submit: Upload ─────────────────────────────────────────────────
  $('start-upload-btn').addEventListener('click', async () => {
    if (!selectedVideo) return;
    const ytUrl = $('upload-yt-url').value.trim();
    const form  = new FormData();
    form.append('video', selectedVideo);
    if (selectedAudio) form.append('audio', selectedAudio);
    if (ytUrl) form.append('youtube_url', ytUrl);
    try {
      const r = await fetch('/api/upload', { method: 'POST', body: form });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      startTracking((await r.json()).job_id);
      refreshHistory();
    } catch (e) { toast(e.message); }
  });

  // ── Submit: YouTube URL ────────────────────────────────────────────
  $('process-url-btn').addEventListener('click', submitUrl);
  $('yt-url').addEventListener('keydown', e => e.key === 'Enter' && submitUrl());

  async function submitUrl() {
    const url = $('yt-url').value.trim();
    if (!url) { toast('Please enter a YouTube URL.'); return; }
    try {
      const r = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, shorts_count: shortsCount }),
      });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      const { job_id } = await r.json();
      startTracking(job_id);
      refreshHistory();
    } catch (e) { toast(e.message); }
  }

  // ── Active job tracking ────────────────────────────────────────────
  function startTracking(jobId) {
    activeJobId = jobId;
    localStorage.setItem(LS_KEY, jobId);
    $('active-job-card').classList.remove('hidden');
    $('live-dot').classList.remove('idle');
    clearInterval(pollTimer);
    pollTimer = setInterval(() => pollActive(jobId), 2000);
    pollActive(jobId);
  }

  async function pollActive(jobId) {
    try {
      const r = await fetch(`/api/status/${jobId}`);
      if (!r.ok) return;
      const job = await r.json();
      renderActiveCard(job);
      if (job.stage === 'done' || job.stage === 'failed') {
        clearInterval(pollTimer);
        $('live-dot').classList.add('idle');
        localStorage.removeItem(LS_KEY);
        refreshHistory();
        if (job.stage === 'done') {
          if (job.scorer_warning) toastWarn(job.scorer_warning);
          loadDetailView(jobId);
        }
        if (job.stage === 'failed') toast('Job failed: ' + (job.error_message || 'unknown error'));
      }
    } catch (_) {}
  }

  function renderActiveCard(job) {
    const title = job.metadata?.title || job.url;
    $('active-job-title').textContent = title.length > 70 ? title.slice(0,68)+'…' : title;
    $('active-job-url').textContent   = job.url.startsWith('local://') ? 'Local file' : job.url;
    const pill = $('active-stage-pill');
    pill.textContent  = STAGE_LABEL[job.stage] || job.stage;
    pill.className    = 'stage-pill ' + pill_class(job.stage);
    $('active-progress-fill').style.width = job.progress_pct + '%';
    $('active-progress-pct').textContent  = job.progress_pct + '%';
    
    if (!$('detail-progress').classList.contains('hidden')) {
      $('detail-progress-fill').style.width = job.progress_pct + '%';
      $('detail-progress-pct').textContent  = job.progress_pct + '%';
    }

    updateStepper(job.stage);
    updateJobsBadge();
  }

  function pill_class(stage) {
    if (stage === 'done')   return 'done';
    if (stage === 'failed') return 'failed';
    if (['clipping','reframing','subtitles','scoring'].includes(stage)) return stage;
    return '';
  }

  function updateStepper(stage) {
    const idx = STAGE_ORDER.indexOf(stage);
    document.querySelectorAll('.step').forEach(el => {
      const si = STAGE_ORDER.indexOf(el.dataset.stage);
      el.classList.toggle('done',   si < idx);
      el.classList.toggle('active', si === idx);
    });
  }

  // ── Reconnect on page load ─────────────────────────────────────────
  async function reconnectIfNeeded() {
    const saved = localStorage.getItem(LS_KEY);
    if (!saved) return;
    try {
      const r = await fetch(`/api/status/${saved}`);
      if (!r.ok) { localStorage.removeItem(LS_KEY); return; }
      const job = await r.json();
      if (job.stage === 'done' || job.stage === 'failed') {
        localStorage.removeItem(LS_KEY);
      } else {
        startTracking(saved);
      }
    } catch (_) { localStorage.removeItem(LS_KEY); }
  }

  // ── Detail view ────────────────────────────────────────────────────
  async function loadDetailView(jobId) {
    if (viewJobId === jobId) return;
    viewJobId = jobId;

    const r = await fetch(`/api/status/${jobId}`);
    if (!r.ok) return;
    const job = await r.json();

    const section = $('job-detail');
    section.classList.remove('hidden');
    section.style.animation = 'none';
    requestAnimationFrame(() => {
      section.style.animation = 'fadeUp .4s cubic-bezier(.4,0,.2,1)';
    });

    // Header
    const title = job.metadata?.title || job.url;
    $('detail-title').textContent = title;
    $('detail-sub').textContent   =
      (job.metadata?.duration ? fmtDuration(job.metadata.duration) + ' · ' : '') +
      (job.metadata?.uploader || '') +
      (job.classification?.content_type ? ' · ' + job.classification.content_type.replace('_',' ') : '');

    // Video player
    const videoSrc = $('video-src');
    videoSrc.src = `/api/video/${jobId}`;
    $('video-player').load();

    // Audio player + download
    const audioSection = $('audio-section');
    const audioBtn = $('audio-download-btn');
    const audioPlayer = $('audio-player');
    const audioSrc = $('audio-src');
    try {
      const audioRes = await fetch(`/api/audio/${jobId}`, { method: 'HEAD' });
      if (audioRes.ok) {
        audioSection.classList.remove('hidden');
        audioSrc.src = `/api/audio/${jobId}`;
        audioPlayer.load();
        audioBtn.href = `/api/audio/${jobId}`;
      } else {
        audioSection.classList.add('hidden');
      }
    } catch (_) {
      audioSection.classList.add('hidden');
    }

    // Transcript
    renderTranscript(job.transcript);

    // Scoring results
    renderScoring(job);

    // Clips — playable + downloadable
    renderClips(jobId, job.clips || []);

    // Mark history card
    document.querySelectorAll('.history-card').forEach(c => {
      c.classList.toggle('active-view', c.dataset.jobId === jobId);
    });

    // Show/hide regenerate button
    $('regenerate-btn').style.display = job.transcript ? 'block' : 'none';
    $('detail-progress').classList.add('hidden');
  }

  function renderScoring(job) {
    const section = $('scoring-section');
    const windows = job.windows || [];
    if (!windows.length) { section.classList.add('hidden'); return; }
    section.classList.remove('hidden');

    // Meta row: engine used, count, warning
    const isReal   = job.clips?.some(c => c.hook);
    const engine   = isReal ? 'Multi-Signal (OpenAI + Audio + Dialogue)' : 'Stub (equal-time windows)';
    const warning  = job.scorer_warning
      ? `<div class="scoring-warning">⚡ ${escHtml(job.scorer_warning)}</div>` : '';

    $('scoring-meta').innerHTML = `
      <div class="scoring-meta-row">
        <span class="scoring-engine ${isReal ? 'engine-real' : 'engine-stub'}">${engine}</span>
        <span class="scoring-badge">${windows.length} window${windows.length !== 1 ? 's' : ''} scored</span>
      </div>
      ${warning}`;

    // Window rows
    $('scoring-windows').innerHTML = windows.map((w, i) => {
      const dur  = (w.end - w.start).toFixed(1);
      const pct  = Math.round((w.score || 0) * 100);
      const bar  = Math.max(4, pct);
      const clip = job.clips?.find(c => Math.abs(c.start_sec - w.start) < 1);
      const hook = clip?.hook ? `<div class="sw-hook">"${escHtml(clip.hook)}"</div>` : '';
      const eng  = clip?.engagement_type ? `<span class="eng-badge">${clip.engagement_type}</span>` : '';
      return `
        <div class="scoring-window">
          <div class="sw-rank">#${i + 1}</div>
          <div class="sw-info">
            <div class="sw-top">
              <span class="sw-time">${fmtTime(w.start)} – ${fmtTime(w.end)} · ${dur}s</span>
              ${eng}
            </div>
            ${hook}
          </div>
          <div class="sw-score-col">
            <div class="sw-bar-track"><div class="sw-bar-fill" style="width:${bar}%"></div></div>
            <span class="sw-score-num">${pct}%</span>
          </div>
        </div>`;
    }).join('');
  }

  function renderTranscript(transcript) {
    const body = $('transcript-body');
    const segs = transcript?.segments || [];
    $('seg-count').textContent = segs.length ? segs.length + ' segments' : '';

    if (!segs.length) {
      body.innerHTML = '<div class="transcript-empty">No transcript available.</div>';
      return;
    }

    body.innerHTML = segs.map((seg, i) => `
      <div class="transcript-row" data-time="${seg.start_sec}" data-idx="${i}">
        <span class="tr-time">${fmtTime(seg.start_sec)}</span>
        <span class="tr-text">${escHtml(seg.text.replace(/\n/g,' '))}</span>
      </div>`).join('');

    // Click → seek video
    body.querySelectorAll('.transcript-row').forEach(row => {
      row.addEventListener('click', () => {
        const player = $('video-player');
        player.currentTime = parseFloat(row.dataset.time);
        player.play().catch(() => {});
        body.querySelectorAll('.transcript-row').forEach(r => r.classList.remove('active'));
        row.classList.add('active');
      });
    });

    // Highlight active row while video plays
    $('video-player').addEventListener('timeupdate', () => {
      const t = $('video-player').currentTime;
      const rows = body.querySelectorAll('.transcript-row');
      let lastActive = null;
      rows.forEach(row => {
        if (parseFloat(row.dataset.time) <= t) lastActive = row;
      });
      if (lastActive) {
        rows.forEach(r => r.classList.remove('active'));
        lastActive.classList.add('active');
      }
    });
  }

  function renderClips(jobId, clips) {
    $('clips-count-label').textContent = clips.length ? clips.length + ' clips' : '';
    const grid = $('clips-grid');
    grid.innerHTML = clips.map(clip => {
      const dur = (clip.end_sec - clip.start_sec).toFixed(1);
      const clipUrl = `/api/clip/${jobId}/${clip.clip_number}`;
      return `
        <div class="clip-card">
          <div class="clip-preview">
            <video class="clip-video" preload="metadata" playsinline loop
              poster="/api/clip/${jobId}/${clip.clip_number}/thumb">
              <source src="${clipUrl}" type="video/mp4">
            </video>
            <div class="clip-play-overlay" data-clip-id="cv-${clip.clip_number}">
              <span class="play-icon">▶</span>
            </div>
          </div>
          <div class="clip-body">
            <div class="clip-num">Short #${clip.clip_number}${clip.engagement_type ? ` <span class="eng-badge">${clip.engagement_type}</span>` : ''}</div>
            <div class="clip-time">${fmtTime(clip.start_sec)} – ${fmtTime(clip.end_sec)} · ${dur}s · score ${clip.score.toFixed(2)}</div>
            ${clip.hook ? `<div class="clip-hook">"${escHtml(clip.hook)}"</div>` : ''}
          </div>
          <a class="clip-dl" href="${clipUrl}" download="short_${clip.clip_number}.mp4">↓ Download</a>
        </div>`;
    }).join('');

    // Click to play clips in modal
    grid.querySelectorAll('.clip-play-overlay').forEach(overlay => {
      overlay.addEventListener('click', (e) => {
        e.stopPropagation();
        const card = overlay.closest('.clip-card');
        const clipUrl = card.querySelector('.clip-video source').src;
        
        const modalVideo = $('modal-video');
        modalVideo.src = clipUrl;
        modalVideo.load();
        $('video-modal').classList.remove('hidden');
        modalVideo.play().catch(() => {});
      });
    });
  }

  // ── History ────────────────────────────────────────────────────────
  async function refreshHistory() {
    try {
      const r = await fetch('/api/jobs');
      if (!r.ok) return;
      const jobs = await r.json();
      renderHistory(jobs);
      updateJobsBadge(jobs.length);
    } catch (_) {}
  }

  function renderHistory(jobs) {
    const empty = $('history-empty');
    const list  = $('history-list');

    if (!jobs.length) { empty.style.display = ''; list.innerHTML = ''; return; }
    empty.style.display = 'none';

    // Update existing cards or add new ones
    const existing = {};
    list.querySelectorAll('.history-card').forEach(c => { existing[c.dataset.jobId] = c; });

    jobs.forEach(job => {
      const prev = existing[job.job_id];
      const card = buildHistoryCard(job);
      if (prev) {
        // Only re-render if stage changed
        if (prev.dataset.stage !== job.stage) list.replaceChild(card, prev);
        else {
          // Just update the mini progress bar
          const fill = prev.querySelector('.h-progress-fill');
          const pct  = prev.querySelector('.h-pct');
          if (fill) fill.style.width = job.progress_pct + '%';
          if (pct)  pct.textContent  = job.progress_pct + '%';
        }
      } else {
        list.appendChild(card);
      }
      delete existing[job.job_id];
    });
    Object.values(existing).forEach(el => el.remove());
  }

  function buildHistoryCard(job) {
    const jid     = job.job_id;
    const stage   = job.stage;
    const clips   = job.clips || [];
    const isDone  = stage === 'done';
    const isFail  = stage === 'failed';
    const title   = job.metadata?.title || job.url.replace('local://','');
    const time    = fmtAge(job.created_at);

    const card = document.createElement('div');
    card.className   = `history-card ${isDone ? 'done' : isFail ? 'failed' : 'running'}` +
                       (viewJobId === jid ? ' active-view' : '');
    card.dataset.jobId = jid;
    card.dataset.stage = stage;

    card.innerHTML = `
      <div class="h-dot ${stage}"></div>
      <div class="h-meta">
        <div class="h-title">${escHtml(title.length > 60 ? title.slice(0,58)+'…' : title)}</div>
        <div class="h-sub">${time} · ${STAGE_LABEL[stage] || stage}${isFail ? ' — <span style="color:var(--red)">' + escHtml(job.error_message || '') + '</span>' : ''}</div>
        <div class="h-progress">
          <div class="h-progress-fill" style="width:${job.progress_pct}%"></div>
        </div>
      </div>
      <div class="h-right">
        ${isDone ? `<span class="h-clips">${clips.length} clips</span>` : ''}
        ${!isDone && !isFail ? `<span class="h-pct">${job.progress_pct}%</span>` : ''}
        ${isDone ? `<span class="h-arrow">›</span>` : ''}
      </div>`;

    if (isDone) {
      card.addEventListener('click', () => loadDetailView(jid));
    }
    return card;
  }

  function updateJobsBadge(n) {
    if (n === undefined) {
      fetch('/api/jobs').then(r => r.json()).then(j => { $('jobs-badge').textContent = j.length + ' job' + (j.length !== 1 ? 's' : ''); }).catch(() => {});
    } else {
      $('jobs-badge').textContent = n + ' job' + (n !== 1 ? 's' : '');
    }
  }

  // ── Refresh button ─────────────────────────────────────────────────
  $('refresh-btn').addEventListener('click', refreshHistory);

  // ── Helpers ────────────────────────────────────────────────────────
  function fmtTime(s) {
    const m = Math.floor(s/60), sec = Math.floor(s%60);
    return `${m}:${String(sec).padStart(2,'0')}`;
  }
  function fmtDuration(s) {
    const m = Math.floor(s/60), sec = s%60;
    return sec ? `${m}m ${sec}s` : `${m}m`;
  }
  function fmtBytes(b) {
    return b < 1048576 ? (b/1024).toFixed(1)+' KB' : (b/1048576).toFixed(1)+' MB';
  }
  function fmtAge(iso) {
    if (!iso) return '';
    const s = Math.floor((Date.now() - new Date(iso)) / 1000);
    if (s < 60)   return s + 's ago';
    if (s < 3600) return Math.floor(s/60) + 'm ago';
    return Math.floor(s/3600) + 'h ago';
  }
  function escHtml(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }
  function toast(msg) {
    const t = document.createElement('div');
    t.className   = 'toast toast-error';
    t.textContent = '⚠ ' + msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 6000);
  }
  function toastWarn(msg) {
    const t = document.createElement('div');
    t.className   = 'toast toast-warn';
    t.textContent = '⚡ Scorer: ' + msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 10000);
  }

  // --- Global exposing for inline HTML handlers ---
  window.regenerateActiveJob = async function() {
    if (!viewJobId) return;
    try {
      const res = await fetch(`/api/generate/${viewJobId}/regenerate`, { method: 'POST' });
      if (!res.ok) {
        toast('Error: ' + (await res.text()));
        return;
      }
      toast('Regeneration started! Processing new clips...');
      
      // Show inline progress bar
      $('regenerate-btn').classList.add('hidden');
      $('detail-progress').classList.remove('hidden');
      $('detail-progress-pct').textContent = '35%';
      $('detail-progress-fill').style.width = '35%';

      localStorage.setItem(LS_KEY, viewJobId);
      clearInterval(pollTimer);
      pollTimer = setInterval(() => pollActive(viewJobId), 2000);
      pollActive(viewJobId);
    } catch (e) {
      toast('Regeneration failed: ' + e.message);
    }
  };

  // ── Modal ──────────────────────────────────────────────────────────
  function closeModal() {
    $('video-modal').classList.add('hidden');
    $('modal-video').pause();
    $('modal-video').src = '';
  }
  $('modal-close').addEventListener('click', closeModal);
  $('modal-backdrop').addEventListener('click', closeModal);

  // ── Boot ───────────────────────────────────────────────────────────
  setInterval(refreshHistory, 4000);
  reconnectIfNeeded();
  refreshHistory();

})();
