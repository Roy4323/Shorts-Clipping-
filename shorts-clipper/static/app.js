(() => {
  'use strict';

  // ── State ──────────────────────────────────────────────────────────
  let selectedVideo  = null;
  let selectedAudio  = null;
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
    renderClips(jobId, job.clips || [], job.metadata);

    // Mark history card
    document.querySelectorAll('.history-card').forEach(c => {
      c.classList.toggle('active-view', c.dataset.jobId === jobId);
    });

    // Show/hide regenerate + re-reframe buttons
    $('regenerate-btn').style.display = job.transcript ? 'block' : 'none';
    $('rereframe-btn').style.display = (job.windows && job.windows.length) ? 'block' : 'none';
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

  function _compactSubs(n) {
    if (!n) return '';
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1).replace(/\.0$/, '') + 'M';
    if (n >= 1_000)     return Math.floor(n / 1_000) + 'K';
    return String(n);
  }

  function renderClips(jobId, clips, jobMeta) {
    $('clips-count-label').textContent = clips.length ? clips.length + ' clips' : '';
    const grid = $('clips-grid');
    const defaultChannelName = escHtml(jobMeta?.uploader || '');
    const defaultSubCount    = _compactSubs(jobMeta?.channel_follower_count);

    grid.innerHTML = clips.map(clip => {
      const n      = clip.clip_number;
      const dur    = (clip.end_sec - clip.start_sec).toFixed(1);
      const clipUrl = `/api/clip/${jobId}/${n}`;
      return `
        <div class="clip-card" id="clip-card-${n}">
          <div class="clip-preview">
            <video class="clip-video" id="clip-video-${n}" preload="metadata" playsinline loop
              poster="/api/clip/${jobId}/${n}/thumb">
              <source src="${clipUrl}" type="video/mp4">
            </video>
            <div class="clip-play-overlay">
              <span class="play-icon">▶</span>
            </div>
          </div>
          <div class="clip-body">
            <div class="clip-num">Short #${n}${clip.engagement_type ? ` <span class="eng-badge">${clip.engagement_type}</span>` : ''}</div>
            <div class="clip-time">${fmtTime(clip.start_sec)} – ${fmtTime(clip.end_sec)} · ${dur}s · score ${clip.score.toFixed(2)}</div>
            ${clip.hook ? `<div class="clip-hook">"${escHtml(clip.hook)}"</div>` : ''}
          </div>

          <!-- ── CTA inline form ── -->
          <div class="cta-inline hidden" id="cta-inline-${n}">
            <div class="cta-inline-grid">
              <input type="text" class="cta-inp" id="cta-name-${n}"
                placeholder="Channel name" value="${defaultChannelName}">
              <input type="text" class="cta-inp" id="cta-subs-${n}"
                placeholder="Subscribers e.g. 20K" value="${defaultSubCount}">
            </div>
            <div class="cta-inline-row2">
              <div class="cta-swatches" id="cta-sw-${n}">
                <button class="cta-sw active" data-color="#CC0000" style="background:#CC0000" title="Red"></button>
                <button class="cta-sw" data-color="#000000" style="background:#000000" title="Black"></button>
                <button class="cta-sw" data-color="#1565C0" style="background:#1565C0" title="Blue"></button>
                <button class="cta-sw" data-color="#2E7D32" style="background:#2E7D32" title="Green"></button>
                <label class="cta-custom-lbl" title="Custom color">
                  <input type="color" id="cta-color-${n}" value="#CC0000">
                </label>
              </div>
              <div class="cta-logo-wrap">
                <div class="cta-lp" id="cta-lp-${n}">?</div>
                <button class="btn-ghost btn-xs" id="cta-logo-btn-${n}">Logo</button>
                <input type="file" id="cta-logo-inp-${n}" accept="image/*" hidden>
              </div>
            </div>
            <button class="btn-primary btn-sm cta-apply-btn" id="cta-apply-${n}">
              Apply Bumper ›
            </button>
            <div class="cta-status hidden" id="cta-status-${n}"></div>
          </div>

          <!-- ── Bottom action row ── -->
          <div class="clip-actions-row">
            <button class="btn-ghost btn-sm cta-toggle-btn" id="cta-toggle-${n}">
              + Add CTA Bumper
            </button>
            <a class="clip-dl" id="clip-dl-${n}" href="${clipUrl}" download="short_${n}.mp4">↓ Download</a>
          </div>
        </div>`;
    }).join('');

    // ── Bind per-clip CTA logic ────────────────────────────────────────────
    clips.forEach(clip => {
      const n = clip.clip_number;
      let accentColor = '#CC0000';
      let logoId      = null;

      // Toggle form open/close — auto-fill channel info on first open
      let ctaAutoFilled = false;
      document.getElementById(`cta-toggle-${n}`).addEventListener('click', async () => {
        const form = document.getElementById(`cta-inline-${n}`);
        const btn  = document.getElementById(`cta-toggle-${n}`);
        const open = form.classList.toggle('hidden') === false;
        btn.textContent = open ? '✕ Close' : '+ Add CTA Bumper';
        btn.classList.toggle('cta-toggle-open', open);

        // Auto-fill from YouTube API on first open (only for YouTube URLs)
        if (open && !ctaAutoFilled) {
          ctaAutoFilled = true;
          const videoUrl = jobMeta?.webpage_url || '';
          if (videoUrl && !videoUrl.startsWith('local://')) {
            try {
              const infoRes = await fetch(`/api/youtube/channel-info?url=${encodeURIComponent(videoUrl)}`);
              if (infoRes.ok) {
                const info = await infoRes.json();
                // Pre-fill subscriber count
                const subsEl = document.getElementById(`cta-subs-${n}`);
                if (subsEl && !subsEl.value) subsEl.value = info.subscriber_count || '';
                // Pre-fill channel name if blank
                const nameEl = document.getElementById(`cta-name-${n}`);
                if (nameEl && !nameEl.value) nameEl.value = info.channel_name || '';
                // Download + display the channel logo
                if (info.logo_url) {
                  const lpEl = document.getElementById(`cta-lp-${n}`);
                  try {
                    const logoRes = await fetch('/api/fetch-logo-from-url', {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ url: info.logo_url }),
                    });
                    if (logoRes.ok) {
                      const logoData = await logoRes.json();
                      logoId = logoData.logo_id;
                      if (lpEl) lpEl.innerHTML =
                        `<img src="${info.logo_url}" style="width:100%;height:100%;object-fit:cover;border-radius:50%" crossorigin="anonymous">`;
                    }
                  } catch (_) {}
                }
              }
            } catch (_) {}
          }
        }
      });

      // Color swatches
      document.querySelectorAll(`#cta-sw-${n} .cta-sw`).forEach(sw => {
        sw.addEventListener('click', () => {
          document.querySelectorAll(`#cta-sw-${n} .cta-sw`).forEach(s => s.classList.remove('active'));
          sw.classList.add('active');
          accentColor = sw.dataset.color;
          document.getElementById(`cta-color-${n}`).value = accentColor;
        });
      });
      document.getElementById(`cta-color-${n}`).addEventListener('input', e => {
        accentColor = e.target.value;
        document.querySelectorAll(`#cta-sw-${n} .cta-sw`).forEach(s => s.classList.remove('active'));
      });

      // Logo upload
      document.getElementById(`cta-logo-btn-${n}`).addEventListener('click', () => {
        document.getElementById(`cta-logo-inp-${n}`).click();
      });
      document.getElementById(`cta-logo-inp-${n}`).addEventListener('change', async () => {
        const file = document.getElementById(`cta-logo-inp-${n}`).files[0];
        if (!file) return;
        const form = new FormData();
        form.append('logo', file);
        try {
          const r = await fetch('/api/upload-logo', { method: 'POST', body: form });
          if (!r.ok) throw new Error('Logo upload failed');
          logoId = (await r.json()).logo_id;
          const reader = new FileReader();
          reader.onload = ev => {
            document.getElementById(`cta-lp-${n}`).innerHTML =
              `<img src="${ev.target.result}" style="width:100%;height:100%;object-fit:cover;border-radius:50%">`;
          };
          reader.readAsDataURL(file);
        } catch (e) { toast(e.message); }
      });

      // Apply bumper
      document.getElementById(`cta-apply-${n}`).addEventListener('click', async () => {
        const channelName = document.getElementById(`cta-name-${n}`).value.trim();
        const subCount    = document.getElementById(`cta-subs-${n}`).value.trim() || '0';
        if (!channelName) { toast('Enter a channel name first.'); return; }

        const applyBtn = document.getElementById(`cta-apply-${n}`);
        const statusEl = document.getElementById(`cta-status-${n}`);
        applyBtn.disabled    = true;
        applyBtn.textContent = 'Rendering…';
        statusEl.textContent = 'Generating bumper, please wait (~15s)…';
        statusEl.className   = 'cta-status cta-status-loading';
        statusEl.classList.remove('hidden');

        try {
          const r = await fetch(`/api/clip/${jobId}/${n}/add-cta`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              enabled: true,
              channel_name: channelName,
              subscriber_count: subCount,
              logo_id: logoId,
              accent_color: accentColor,
            }),
          });
          const data = await r.json().catch(() => null);
          if (!r.ok) throw new Error((data && data.detail) || r.statusText);

          const ctaDuration  = (data && data.new_duration)   || 0;
          const ctaThumbUrl  = (data && data.cta_thumb_url)  || null;
          const ctaStartSec  = Math.max(0, ctaDuration - 3.5);

          // Collapse form, lock toggle button
          document.getElementById(`cta-inline-${n}`).classList.add('hidden');
          const toggleBtn = document.getElementById(`cta-toggle-${n}`);
          toggleBtn.textContent = '✓ CTA Applied';
          toggleBtn.classList.add('cta-applied');
          toggleBtn.disabled = true;

          // Bust browser cache so the updated file (with bumper) is fetched
          const ts = Date.now();
          const newUrl = `/api/clip/${jobId}/${n}?t=${ts}`;

          // Update video preview source + reload
          const vid = document.getElementById(`clip-video-${n}`);
          if (vid) {
            const srcEl = vid.querySelector('source');
            if (srcEl) srcEl.src = newUrl;
            vid.src = newUrl;
            // Show the CTA frame as the poster so the card thumbnail changes
            if (ctaThumbUrl) vid.poster = ctaThumbUrl + `?t=${ts}`;
            vid.load();
          }

          // Update download link
          const dlLink = document.getElementById(`clip-dl-${n}`);
          if (dlLink) {
            dlLink.href = newUrl;
            dlLink.setAttribute('download', `short_${n}_cta.mp4`);
          }

          // Inject a "▶ Preview CTA" button that opens the modal at the CTA start
          const actionsRow = document.querySelector(`#clip-card-${n} .clip-actions-row`);
          if (actionsRow && !actionsRow.querySelector('.cta-preview-btn')) {
            const previewBtn = document.createElement('button');
            previewBtn.className = 'btn-ghost btn-sm cta-preview-btn';
            previewBtn.textContent = '▶ Preview CTA';
            previewBtn.addEventListener('click', () => {
              const modalVideo = $('modal-video');
              modalVideo.src = newUrl;
              modalVideo.load();
              $('video-modal').classList.remove('hidden');
              modalVideo.addEventListener('loadedmetadata', function seekOnce() {
                modalVideo.currentTime = ctaStartSec;
                modalVideo.play().catch(() => {});
                modalVideo.removeEventListener('loadedmetadata', seekOnce);
              });
            });
            actionsRow.insertBefore(previewBtn, actionsRow.firstChild);
          }

          toastSuccess(`CTA bumper appended (${ctaDuration}s total)! Click "▶ Preview CTA" to see it, or ↓ Download to save.`);

        } catch (e) {
          statusEl.textContent = '✗ Failed: ' + e.message;
          statusEl.className   = 'cta-status cta-status-error';
          applyBtn.disabled    = false;
          applyBtn.textContent = 'Apply Bumper ›';
        }
      });
    });

    // ── Play in modal ──────────────────────────────────────────────────────
    grid.querySelectorAll('.clip-play-overlay').forEach(overlay => {

      overlay.addEventListener('click', e => {
        e.stopPropagation();
        const card    = overlay.closest('.clip-card');
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
  function toastSuccess(msg) {
    const t = document.createElement('div');
    t.className   = 'toast toast-success';
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 7000);
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

  window.rereframeActiveJob = async function() {
    if (!viewJobId) return;
    try {
      const res = await fetch(`/api/generate/${viewJobId}/rereframe`, { method: 'POST' });
      if (!res.ok) {
        toast('Error: ' + (await res.text()));
        return;
      }
      toast('Re-reframe started! Re-processing reframe + subtitles...');

      $('rereframe-btn').classList.add('hidden');
      $('regenerate-btn').classList.add('hidden');
      $('detail-progress').classList.remove('hidden');
      $('detail-progress-pct').textContent = '60%';
      $('detail-progress-fill').style.width = '60%';

      localStorage.setItem(LS_KEY, viewJobId);
      clearInterval(pollTimer);
      pollTimer = setInterval(() => pollActive(viewJobId), 2000);
      pollActive(viewJobId);
    } catch (e) {
      toast('Re-reframe failed: ' + e.message);
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

  // ══════════════════════════════════════════════════════════════════
  // PAGE NAVIGATION (Shorts ↔ Hook Studio)
  // ══════════════════════════════════════════════════════════════════
  document.querySelectorAll('.page-nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.page-nav-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const page = btn.dataset.page;
      $('shorts-page').classList.toggle('hidden', page !== 'shorts');
      $('hooks-page').classList.toggle('hidden', page !== 'hooks');
      if (page === 'hooks') hsLoadJobList();
    });
  });


  // ══════════════════════════════════════════════════════════════════
  // HOOK STUDIO
  // ══════════════════════════════════════════════════════════════════

  let hsJobData = null;   // currently loaded job payload

  const VOICES = [
    { id: 'en-US-GuyNeural',    label: 'Guy (US Male)' },
    { id: 'en-US-JennyNeural',  label: 'Jenny (US Female)' },
    { id: 'en-GB-RyanNeural',   label: 'Ryan (UK Male)' },
  ];

  // Threshold matching hook_detector.py
  const HOOK_ENERGY_THRESHOLD = 0.7;
  const HOOK_SEMANTIC_CEILING = 0.4;

  function hsIsCandidate(w) {
    const s2 = w.signal2 || 0, s3 = w.signal3 || 0, s1 = w.signal1 || 0;
    return (s2 + s3) / 2 > HOOK_ENERGY_THRESHOLD && s1 < HOOK_SEMANTIC_CEILING;
  }

  // Populate job selector with done jobs
  async function hsLoadJobList() {
    try {
      const r = await fetch('/api/jobs');
      if (!r.ok) return;
      const jobs = await r.json();
      const sel = $('hs-job-select');
      const prev = sel.value;
      sel.innerHTML = '<option value="">— Select a completed job —</option>';
      jobs.filter(j => j.stage === 'done').forEach(j => {
        const title = j.metadata?.title || j.url;
        const opt = document.createElement('option');
        opt.value = j.job_id;
        opt.textContent = (title.length > 70 ? title.slice(0, 68) + '…' : title);
        sel.appendChild(opt);
      });
      if (prev) sel.value = prev;
    } catch (_) {}
  }

  $('hs-load-btn').addEventListener('click', async () => {
    const jobId = $('hs-job-select').value;
    if (!jobId) { toast('Please select a job first.'); return; }
    try {
      const r = await fetch(`/api/status/${jobId}`);
      if (!r.ok) throw new Error('Job not found');
      hsJobData = await r.json();
      hsRenderWorkspace(jobId, hsJobData);
    } catch (e) { toast(e.message); }
  });

  function hsRenderWorkspace(jobId, job) {
    $('hs-workspace').classList.remove('hidden');

    // Header
    const title = job.metadata?.title || job.url;
    $('hs-job-title').textContent = title;
    $('hs-job-meta').textContent =
      (job.metadata?.duration ? fmtDuration(job.metadata.duration) + '  ·  ' : '') +
      (job.metadata?.uploader || '') +
      (job.classification?.content_type ? '  ·  ' + job.classification.content_type.replace(/_/g, ' ') : '');

    // Video player
    $('hs-video-src').src = `/api/video/${jobId}`;
    $('hs-video').load();

    // Timeline
    hsRenderTimeline(job);

    // Window cards
    hsRenderWindows(jobId, job);
  }

  function hsRenderTimeline(job) {
    const duration = job.metadata?.duration || 0;
    if (!duration) { $('hs-timeline').style.display = 'none'; return; }
    $('hs-timeline').style.display = '';
    $('hs-tl-time-end').textContent = fmtTime(duration);

    const windows = job.windows || [];
    $('hs-tl-segments').innerHTML = windows.map((w, i) => {
      const left  = (w.start / duration * 100).toFixed(2);
      const width = Math.max(0.5, ((w.end - w.start) / duration * 100)).toFixed(2);
      const isHook = hsIsCandidate(w);
      const label = `#${i + 1}  ${fmtTime(w.start)}–${fmtTime(w.end)}`;
      return `<div class="hs-tl-seg ${isHook ? 'hook-cand' : ''}"
                   style="left:${left}%;width:${width}%"
                   data-start="${w.start}"
                   title="${label}${isHook ? '  🔥 Hook candidate' : ''}"></div>`;
    }).join('');

    // Click to seek
    $('hs-tl-segments').querySelectorAll('.hs-tl-seg').forEach(seg => {
      seg.addEventListener('click', () => {
        const video = $('hs-video');
        video.currentTime = parseFloat(seg.dataset.start);
        video.play().catch(() => {});
      });
    });

    // Playhead sync
    $('hs-video').addEventListener('timeupdate', () => {
      if (!duration) return;
      const pct = ($('hs-video').currentTime / duration * 100).toFixed(2);
      $('hs-tl-playhead').style.left = pct + '%';
    });

    // Click anywhere on timeline to seek
    $('hs-timeline').addEventListener('click', e => {
      if (e.target.classList.contains('hs-tl-seg')) return;
      const rect = $('hs-timeline').getBoundingClientRect();
      const pct  = (e.clientX - rect.left) / rect.width;
      $('hs-video').currentTime = pct * duration;
    });
  }

  function hsRenderWindows(jobId, job) {
    const windows  = job.windows || [];
    const hookClips = job.hook_clips || [];
    const content_type = job.classification?.content_type || 'general';
    const container = $('hs-windows');

    if (!windows.length) {
      container.innerHTML = '<div style="color:var(--muted);text-align:center;padding:2rem">No scored windows found in this job.</div>';
      return;
    }

    container.innerHTML = windows.map((w, i) => {
      const isHook  = hsIsCandidate(w);
      const dur     = (w.end - w.start).toFixed(1);
      const pct     = Math.round((w.score || 0) * 100);
      const s1      = Math.round((w.signal1 || 0) * 100);
      const s2      = Math.round((w.signal2 || 0) * 100);
      const s3      = Math.round((w.signal3 || 0) * 100);
      const existHook = hookClips.find(h => Math.abs(h.start_sec - w.start) < 2);

      const voiceOpts = VOICES.map(v =>
        `<option value="${v.id}" ${v.id === 'en-US-GuyNeural' ? 'selected' : ''}>${v.label}</option>`
      ).join('');

      const scoreSignals = `
        <div class="hs-signals">
          <div class="hs-sig-row" title="Semantic quality (Signal 1)">
            <span class="hs-sig-lbl">Semantic</span>
            <div class="hs-sig-track"><div class="hs-sig-fill semantic" style="width:${s1}%"></div></div>
            <span class="hs-sig-num">${s1}%</span>
          </div>
          <div class="hs-sig-row" title="Audio energy (Signal 2)">
            <span class="hs-sig-lbl">Energy</span>
            <div class="hs-sig-track"><div class="hs-sig-fill energy" style="width:${s2}%"></div></div>
            <span class="hs-sig-num">${s2}%</span>
          </div>
          <div class="hs-sig-row" title="Speaker activity (Signal 3)">
            <span class="hs-sig-lbl">Speaker</span>
            <div class="hs-sig-track"><div class="hs-sig-fill speaker" style="width:${s3}%"></div></div>
            <span class="hs-sig-num">${s3}%</span>
          </div>
        </div>`;

      const defaultChannelName = job.metadata?.uploader || '';
      const existClip = existHook ? `
        <div class="hs-clip-result" id="hs-result-${i}">
          <div class="hs-clip-preview-row">
            <video class="hs-clip-thumb" preload="metadata" playsinline loop
              poster="/api/hook/${jobId}/${existHook.clip_number}/thumb"
              src="${existHook.download_url}"></video>
            <div class="hs-clip-info">
              <div class="hs-clip-title">Hook Clip #${existHook.clip_number}</div>
              <div class="hs-clip-sub">${existHook.hook_type} · ${existHook.duration}s · ${existHook.voice}</div>
              <div class="hs-clip-hook-text">"${escHtml(existHook.hook_text)}"</div>
              ${_hookCTAHtml(jobId, existHook.clip_number, defaultChannelName)}
            </div>
          </div>
        </div>` : `<div class="hs-clip-result hidden" id="hs-result-${i}"></div>`;

      return `
        <div class="card hs-window-card ${isHook ? 'hook-candidate-card' : ''}" data-idx="${i}">
          <div class="hs-win-header">
            <div class="hs-win-left">
              <span class="hs-win-rank">#${i + 1}</span>
              <div>
                <div class="hs-win-time">${fmtTime(w.start)} → ${fmtTime(w.end)} · ${dur}s</div>
                ${w.engagement_type ? `<span class="eng-badge">${w.engagement_type}</span>` : ''}
              </div>
            </div>
            <div class="hs-win-right">
              ${isHook ? '<span class="hook-cand-badge">🔥 Hook Candidate</span>' : ''}
              <span class="hs-win-score">${pct}%</span>
            </div>
          </div>

          ${scoreSignals}

          ${w.hook ? `<div class="hs-original-hook"><span class="hs-label">AI Caption:</span> "${escHtml(w.hook)}"</div>` : ''}

          <div class="hs-script-section">
            <div class="hs-label-row">
              <label class="hs-label" for="hs-script-${i}">Hook Script</label>
              <button class="btn-ghost btn-sm hs-suggest-btn" data-idx="${i}" data-content-type="${content_type}">
                ✨ AI Suggest
              </button>
            </div>
            <textarea class="hs-script-area" id="hs-script-${i}"
              placeholder="Write or auto-generate a 30-40 word spoken hook…">${w.hook || ''}</textarea>
            <div class="hs-gen-row">
              <select class="hs-voice-select" id="hs-voice-${i}">${voiceOpts}</select>
              <button class="btn-primary hs-gen-btn" data-idx="${i}" data-job="${jobId}">
                🎤 Generate Clip
              </button>
            </div>
            <div class="hs-gen-status hidden" id="hs-status-${i}"></div>
          </div>

          ${existClip}
        </div>`;
    }).join('');

    // Bind CTA forms for pre-existing hook clips
    const defaultChannelName = job.metadata?.uploader || '';
    hookClips.forEach(hc => _bindHookCTA(jobId, hc.clip_number, defaultChannelName));

    // Suggest buttons
    container.querySelectorAll('.hs-suggest-btn').forEach(btn => {
      btn.addEventListener('click', () => hsSuggestScript(btn.dataset.idx, btn.dataset.contentType, jobId));
    });

    // Generate buttons
    container.querySelectorAll('.hs-gen-btn').forEach(btn => {
      btn.addEventListener('click', () => hsGenerateClip(btn.dataset.idx, btn.dataset.job));
    });

    // Click on clip thumbnails to play in modal
    container.querySelectorAll('.hs-clip-thumb').forEach(vid => {
      vid.addEventListener('click', () => {
        const modalVideo = $('modal-video');
        modalVideo.src = vid.src;
        modalVideo.load();
        $('video-modal').classList.remove('hidden');
        modalVideo.play().catch(() => {});
      });
    });
  }

  async function hsSuggestScript(idx, contentType, jobId) {
    const btn = document.querySelector(`.hs-suggest-btn[data-idx="${idx}"]`);
    const area = $(`hs-script-${idx}`);
    const status = $(`hs-status-${idx}`);
    btn.disabled = true;
    btn.textContent = '⏳ Thinking…';
    status.classList.remove('hidden');
    status.textContent = 'Generating hook script…';
    status.className = 'hs-gen-status info';
    try {
      const r = await fetch(`/api/hook/${jobId}/suggest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ window_index: parseInt(idx), content_type: contentType }),
      });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      const data = await r.json();
      area.value = data.hook;
      status.textContent = `✅ ${data.hook_type} hook suggested (est. ${data.duration_estimate_sec}s)`;
      status.className = 'hs-gen-status success';
    } catch (e) {
      status.textContent = '⚠ Suggestion failed: ' + e.message;
      status.className = 'hs-gen-status error';
    } finally {
      btn.disabled = false;
      btn.textContent = '✨ AI Suggest';
    }
  }

  async function hsGenerateClip(idx, jobId) {
    const btn    = document.querySelector(`.hs-gen-btn[data-idx="${idx}"]`);
    const area   = $(`hs-script-${idx}`);
    const voice  = $(`hs-voice-${idx}`).value;
    const status = $(`hs-status-${idx}`);
    const result = $(`hs-result-${idx}`);

    const hookText = area.value.trim();
    if (!hookText) { toast('Please write or generate a hook script first.'); return; }

    btn.disabled = true;
    btn.textContent = '⏳ Generating…';
    status.classList.remove('hidden');
    status.textContent = '🎤 Synthesising speech…';
    status.className = 'hs-gen-status info';

    try {
      const r = await fetch(`/api/hook/${jobId}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ window_index: parseInt(idx), hook_text: hookText, voice }),
      });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      const clip = await r.json();

      status.textContent = `✅ Hook clip #${clip.clip_number} ready (${clip.duration}s)`;
      status.className = 'hs-gen-status success';

      // Render result card
      const hsDefaultName = document.getElementById('hs-job-title')?.textContent || '';
      result.classList.remove('hidden');
      result.innerHTML = `
        <div class="hs-clip-preview-row">
          <video class="hs-clip-thumb" preload="metadata" playsinline loop
            src="${clip.download_url}" style="cursor:pointer"></video>
          <div class="hs-clip-info">
            <div class="hs-clip-title">Hook Clip #${clip.clip_number}</div>
            <div class="hs-clip-sub">${clip.hook_type} · ${clip.duration}s · ${clip.voice}</div>
            <div class="hs-clip-hook-text">"${escHtml(clip.hook_text)}"</div>
            ${_hookCTAHtml(jobId, clip.clip_number, hsDefaultName)}
          </div>
        </div>`;

      _bindHookCTA(jobId, clip.clip_number, hsDefaultName);

      // Click to play in modal
      result.querySelector('.hs-clip-thumb').addEventListener('click', () => {
        const modalVideo = $('modal-video');
        modalVideo.src = clip.download_url;
        modalVideo.load();
        $('video-modal').classList.remove('hidden');
        modalVideo.play().catch(() => {});
      });

    } catch (e) {
      status.textContent = '⚠ Generation failed: ' + e.message;
      status.className = 'hs-gen-status error';
    } finally {
      btn.disabled = false;
      btn.textContent = '🎤 Generate Clip';
    }
  }

  // ── CTA helpers shared by Hook Studio ─────────────────────────────────

  function _hookCTAHtml(jobId, clipNumber, defaultName) {
    const n  = clipNumber;
    const dn = escHtml(defaultName || '');
    return `
      <div class="cta-inline hidden" id="hcta-inline-${n}">
        <div class="cta-inline-grid">
          <input type="text" class="cta-inp" id="hcta-name-${n}"
            placeholder="Channel name" value="${dn}">
          <input type="text" class="cta-inp" id="hcta-subs-${n}"
            placeholder="Subscribers e.g. 20K">
        </div>
        <div class="cta-inline-row2">
          <div class="cta-swatches" id="hcta-sw-${n}">
            <button class="cta-sw active" data-color="#CC0000" style="background:#CC0000" title="Red"></button>
            <button class="cta-sw" data-color="#000000" style="background:#000000" title="Black"></button>
            <button class="cta-sw" data-color="#1565C0" style="background:#1565C0" title="Blue"></button>
            <button class="cta-sw" data-color="#2E7D32" style="background:#2E7D32" title="Green"></button>
            <label class="cta-custom-lbl" title="Custom color">
              <input type="color" id="hcta-color-${n}" value="#CC0000">
            </label>
          </div>
          <div class="cta-logo-wrap">
            <div class="cta-lp" id="hcta-lp-${n}">?</div>
            <button class="btn-ghost btn-xs" id="hcta-logo-btn-${n}">Logo</button>
            <input type="file" id="hcta-logo-inp-${n}" accept="image/*" hidden>
          </div>
        </div>
        <button class="btn-primary btn-sm cta-apply-btn" id="hcta-apply-${n}">
          Apply Bumper ›
        </button>
        <div class="cta-status hidden" id="hcta-status-${n}"></div>
      </div>
      <div class="clip-actions-row" style="padding:.6rem .2rem .2rem">
        <button class="btn-ghost btn-sm cta-toggle-btn" id="hcta-toggle-${n}">
          + Add CTA Bumper
        </button>
        <a class="btn-outline btn-sm" href="/api/hook/${jobId}/${n}" download
          id="hcta-dl-${n}">↓ Download</a>
      </div>`;
  }

  function _bindHookCTA(jobId, clipNumber) {
    const n = clipNumber;
    // Guard: if this hook clip's window card isn't rendered in the DOM, skip.
    if (!document.getElementById(`hcta-toggle-${n}`)) return;

    let accentColor = '#CC0000';
    let logoId      = null;

    document.getElementById(`hcta-toggle-${n}`).addEventListener('click', () => {
      const form = document.getElementById(`hcta-inline-${n}`);
      const btn  = document.getElementById(`hcta-toggle-${n}`);
      const open = form.classList.toggle('hidden') === false;
      btn.textContent = open ? '✕ Close' : '+ Add CTA Bumper';
      btn.classList.toggle('cta-toggle-open', open);
    });

    document.querySelectorAll(`#hcta-sw-${n} .cta-sw`).forEach(sw => {
      sw.addEventListener('click', () => {
        document.querySelectorAll(`#hcta-sw-${n} .cta-sw`).forEach(s => s.classList.remove('active'));
        sw.classList.add('active');
        accentColor = sw.dataset.color;
        document.getElementById(`hcta-color-${n}`).value = accentColor;
      });
    });
    document.getElementById(`hcta-color-${n}`).addEventListener('input', e => {
      accentColor = e.target.value;
      document.querySelectorAll(`#hcta-sw-${n} .cta-sw`).forEach(s => s.classList.remove('active'));
    });

    document.getElementById(`hcta-logo-btn-${n}`).addEventListener('click', () => {
      document.getElementById(`hcta-logo-inp-${n}`).click();
    });
    document.getElementById(`hcta-logo-inp-${n}`).addEventListener('change', async () => {
      const file = document.getElementById(`hcta-logo-inp-${n}`).files[0];
      if (!file) return;
      const form = new FormData();
      form.append('logo', file);
      try {
        const r = await fetch('/api/upload-logo', { method: 'POST', body: form });
        if (!r.ok) throw new Error('Logo upload failed');
        logoId = (await r.json()).logo_id;
        const reader = new FileReader();
        reader.onload = ev => {
          document.getElementById(`hcta-lp-${n}`).innerHTML =
            `<img src="${ev.target.result}" style="width:100%;height:100%;object-fit:cover;border-radius:50%">`;
        };
        reader.readAsDataURL(file);
      } catch (e) { toast(e.message); }
    });

    document.getElementById(`hcta-apply-${n}`).addEventListener('click', async () => {
      const channelName = document.getElementById(`hcta-name-${n}`).value.trim();
      const subCount    = document.getElementById(`hcta-subs-${n}`).value.trim() || '0';
      if (!channelName) { toast('Enter a channel name first.'); return; }

      const applyBtn = document.getElementById(`hcta-apply-${n}`);
      const statusEl = document.getElementById(`hcta-status-${n}`);
      applyBtn.disabled    = true;
      applyBtn.textContent = 'Rendering…';
      statusEl.textContent = 'Generating bumper, please wait (~15s)…';
      statusEl.className   = 'cta-status cta-status-loading';
      statusEl.classList.remove('hidden');

      try {
        const r = await fetch(`/api/hook/${jobId}/${n}/add-cta`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            enabled: true,
            channel_name: channelName,
            subscriber_count: subCount,
            logo_id: logoId,
            accent_color: accentColor,
          }),
        });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);

        document.getElementById(`hcta-inline-${n}`).classList.add('hidden');
        const toggleBtn = document.getElementById(`hcta-toggle-${n}`);
        toggleBtn.textContent = '✓ CTA Applied';
        toggleBtn.classList.add('cta-applied');
        toggleBtn.disabled = true;

        // Bust cache so the updated hook clip (with bumper) is fetched
        const ts = Date.now();
        const newUrl = `/api/hook/${jobId}/${n}?t=${ts}`;

        // Reload the video thumbnail in Hook Studio
        const vid = document.querySelector(`#hs-result-${n} .hs-clip-thumb, .hs-clip-thumb[src*="/api/hook/${jobId}/${n}"]`);
        if (vid) {
          vid.src = newUrl;
          vid.load();
        }

        // Update the download link
        const dlLink = document.getElementById(`hcta-dl-${n}`);
        if (dlLink) {
          dlLink.href = newUrl;
          dlLink.setAttribute('download', `hook_${n}_cta.mp4`);
        }

      } catch (e) {
        statusEl.textContent = '✗ Failed: ' + e.message;
        statusEl.className   = 'cta-status cta-status-error';
        applyBtn.disabled    = false;
        applyBtn.textContent = 'Apply Bumper ›';
      }
    });
  }

})();
