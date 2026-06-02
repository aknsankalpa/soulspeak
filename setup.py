"""
setup.py  —  SoulSpeak HTML Setup
Injects the Flask API bridge into the original HTML and writes
templates/index.html that the Flask app serves.

Run once before starting the server:
    python setup.py
"""

import os

_ROOT        = os.path.dirname(os.path.abspath(__file__))
SOURCE_HTML  = os.path.join(_ROOT, "frontend", "index_source.html")
TEMPLATE_DIR = os.path.join(_ROOT, "frontend", "templates")
OUTPUT_HTML  = os.path.join(TEMPLATE_DIR, "index.html")

# ─── JS Bridge — injected just before </body> ──────────────────────────────────
API_BRIDGE = r"""
<script>
/* ═══════════════════════════════════════════════════════════════════════
   SoulSpeak — Flask / Ollama API Bridge  (injected by setup.py)
   Overrides static demo functions with real fetch() + SSE calls.
   ═══════════════════════════════════════════════════════════════════════ */

// ── Shared state ─────────────────────────────────────────────────────────────
const API = {
  profession:    '',      // set from registered profession on login
  personality:   {},     // Big Five scores {Openness:74, …}
  chatHistory:   [],     // [{role,content}] multi-turn chat log
  chatSessionId: null,   // UUID of the current chat conversation
  ollamaModel:   '',     // populated from /api/status
};

// ── On load: restore profession from saved session, fetch status & history ───
document.addEventListener('DOMContentLoaded', async () => {
  // Restore profession from localStorage if user is already logged in
  try {
    const savedUser = JSON.parse(localStorage.getItem('soulspeak_user') || 'null');
    if (savedUser && savedUser.profession) {
      API.profession = savedUser.profession;
    }
  } catch (_) {}

  try {
    const r   = await fetch('/api/status');
    const stat = await r.json();
    API.ollamaModel = stat.ollama && stat.ollama.model ? stat.ollama.model : '';
    console.log('[SoulSpeak] Status:', stat);
  } catch (_) {}

  loadSessionHistory();
});

// ── Auth ──────────────────────────────────────────────────────────────────────
async function apiLogin(email, password) {
  const r = await fetch('/api/auth/login', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({email, password})
  });
  return r.json();
}

async function apiRegister(payload) {
  const r = await fetch('/api/auth/register', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(payload)
  });
  return r.json();
}

// ── Analysis ──────────────────────────────────────────────────────────────────
// Replaces the static sendForAnalysis() with a real multipart POST + loading overlay.
async function sendForAnalysis() {
  const tick = (typeof _showAnalysisOverlay === 'function') ? _showAnalysisOverlay() : null;

  const fd = new FormData();
  fd.append('duration',    window._recordSeconds || window.recordSeconds || 30);
  fd.append('profession',  API.profession);
  fd.append('prompt_type', 'passage');

  const hasRecording = window._recordedBlob && window._recordedBlob.size > 0;
  const fileInput = document.getElementById('file-input');
  const hasUpload = fileInput && fileInput.files && fileInput.files[0];

  if (hasRecording) {
    const ext = (window._mediaRecorder && window._mediaRecorder.mimeType || '').includes('ogg') ? '.ogg' : '.webm';
    fd.append('audio', window._recordedBlob, 'recording' + ext);
  } else if (hasUpload) {
    fd.append('audio', fileInput.files[0]);
  } else {
    if (typeof _hideAnalysisOverlay === 'function') _hideAnalysisOverlay(tick);
    showToast('⚠️ No audio to analyse. Record or upload a file first.', 'warning');
    return;
  }

  const token = window.authToken || localStorage.getItem('soulspeak_token') || '';
  const headers = token ? { 'Authorization': 'Bearer ' + token } : {};

  try {
    const r   = await fetch('/api/analyze', { method: 'POST', body: fd, headers });
    if (!r.ok) {
      const errBody = await r.json().catch(() => ({}));
      if (typeof _hideAnalysisOverlay === 'function') _hideAnalysisOverlay(tick);
      showToast('❌ Analysis error: ' + (errBody.error || r.statusText), 'danger');
      return;
    }
    const res = await r.json();

    if (typeof _hideAnalysisOverlay === 'function') _hideAnalysisOverlay(tick);

    if (res.error) { showToast('❌ ' + res.error, 'danger'); return; }

    API.personality = res.scores || {};
    // Sync profession used in analysis back to API state
    if (res.profession) API.profession = res.profession;

    _populateResultsView(res);

    const src = res.source || 'completed';
    showToast('✅ Analysis complete! Your personality profile is ready.', 'success');
    console.log('[SoulSpeak] Analysis src:', src, '| scores:', res.scores);

    if (typeof switchView === 'function') switchView('results');
  } catch (err) {
    if (typeof _hideAnalysisOverlay === 'function') _hideAnalysisOverlay(tick);
    showToast('❌ Analysis failed: ' + err.message + ' — Is the Flask server running?', 'danger');
  }
}

// Populate results page with real data
function _populateResultsView(session) {
  const traitShorts = {
    Openness:'o', Conscientiousness:'c',
    Extraversion:'e', Agreeableness:'a', Neuroticism:'n'
  };

  // Score values and bars (common ID patterns)
  Object.entries(session.scores || {}).forEach(([trait, score]) => {
    const sh = traitShorts[trait];
    // Update text elements
    ['score-'+sh,'val-'+sh,'pct-'+sh,'score-'+trait.toLowerCase()].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = score + '%';
    });
    // Update bar widths — also set --w CSS variable for elements using it
    ['bar-'+sh,'fill-'+sh,'prog-'+sh,'bar-'+trait.toLowerCase()].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.style.width = score + '%';
        el.style.setProperty('--w', score + '%');
      }
    });
    // Data attribute selectors
    document.querySelectorAll('[data-trait="'+trait.toLowerCase()+'"]').forEach(el => {
      const fill = el.querySelector('.bar-fill,.fill,.progress,.trait-fill,.trait-fill-full');
      if (fill) { fill.style.width = score + '%'; fill.style.setProperty('--w', score + '%'); }
      const label = el.querySelector('.score,.pct,.value,.trait-pct');
      if (label) label.textContent = score + '%';
    });
  });

  // Meta
  const overall = session.overall || 0;
  const metaMap = {
    'result-date':     session.date || '',
    'result-duration': session.duration_fmt || '',
    'result-overall':  overall + '%',
  };
  Object.entries(metaMap).forEach(([id, val]) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  });

  // SVG ring — dashoffset = 314 * (1 - overall/100)
  const ring = document.getElementById('ring-circle');
  if (ring) ring.setAttribute('stroke-dashoffset', (314 * (1 - overall / 100)).toFixed(1));

  // Personality type label
  const ptEl = document.getElementById('result-personality-type');
  if (ptEl && session.personality_type) ptEl.textContent = session.personality_type;

  // Transcript panel (if present)
  const tPanel = document.getElementById('transcript-text') ||
                 document.querySelector('.transcript-content');
  if (tPanel && session.transcript) {
    tPanel.textContent = session.transcript;
    const wrap = document.getElementById('transcript-panel') ||
                 document.querySelector('.transcript-panel');
    if (wrap) wrap.style.display = 'block';
  }

  // Source badge
  const srcEl = document.getElementById('analysis-source') ||
                document.querySelector('.analysis-source');
  if (srcEl) srcEl.textContent = session.source || '';

  // Radar chart
  _updateRadarChart(session.scores || {});

  // Insights
  const container = document.getElementById('insight-list') ||
                    document.querySelector('.insight-list,.insights-list');
  if (container && session.insights && session.insights.length) {
    container.innerHTML = session.insights.map(ins => `
      <div class="insight-card" style="
        padding:14px 18px; border-radius:10px; margin-bottom:10px;
        background:${ins.type==='strength'?'rgba(16,185,129,0.1)':'rgba(245,158,11,0.1)'};
        border-left:3px solid ${ins.type==='strength'?'var(--accent4)':'var(--accent3)'};
        font-size:0.85rem; line-height:1.6; color:var(--text);">
        <span style="margin-right:8px">${ins.icon||''}</span>${_escHtml(ins.text)}
      </div>`).join('');
  }

  // Profession card — use session profession falling back to logged-in user's profession
  const _prof = session.profession || (window.appUser && window.appUser.profession) || '';
  if (_prof) _updateProfessionCard(_prof, session.scores || {});
}

function _updateRadarChart(scores) {
  const poly = document.getElementById('radar-polygon') ||
               document.querySelector('.radar-polygon, polygon.data');
  if (!poly) return;
  const r      = 100;
  const order  = ['Openness','Conscientiousness','Extraversion','Agreeableness','Neuroticism'];
  const angles = [-90,-18,54,126,198].map(a => a * Math.PI / 180);
  const pts = order.map((t, i) => {
    const pct = (scores[t] || 0) / 100;
    return (Math.cos(angles[i]) * r * pct).toFixed(1) + ',' +
           (Math.sin(angles[i]) * r * pct).toFixed(1);
  });
  poly.setAttribute('points', pts.join(' '));
}

// ── VOXMIND Streaming Chat (Ollama SSE) ──────────────────────────────────────
// Fully replaces the static sendChat() with a real streaming fetch.
async function sendChat() {
  const inp = document.getElementById('chat-input');
  const msg = inp ? inp.value.trim() : '';
  if (!msg) return;

  const area = document.getElementById('chat-messages');
  if (!area) return;

  // User bubble
  area.innerHTML += `
    <div class="chat-msg user">
      <div class="avatar" style="width:32px;height:32px;font-size:0.75rem">SC</div>
      <div class="chat-bubble">${_escHtml(msg)}</div>
    </div>`;
  inp.value = '';
  area.scrollTop = area.scrollHeight;

  // Bot bubble (streaming target)
  const botId = 'bot-' + Date.now();
  area.innerHTML += `
    <div class="chat-msg bot" id="${botId}">
      <div class="chat-avatar"
           style="background:linear-gradient(135deg,var(--accent),var(--accent2))">SS</div>
      <div class="chat-bubble" id="bubble-${botId}">
        <div class="typing-indicator">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    </div>`;
  area.scrollTop = area.scrollHeight;

  API.chatHistory.push({ role: 'user', content: msg });

  // Ensure a chat session ID exists for this conversation
  if (!API.chatSessionId) {
    API.chatSessionId = 'chat_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
  }

  let fullReply = '';
  const bubble  = document.getElementById('bubble-' + botId);

  const _token  = window.authToken || localStorage.getItem('soulspeak_token') || '';
  const _hdrs   = { 'Content-Type': 'application/json' };
  if (_token) _hdrs['Authorization'] = 'Bearer ' + _token;

  try {
    const resp = await fetch('/api/chat', {
      method:  'POST',
      headers: _hdrs,
      body: JSON.stringify({
        message:         msg,
        history:         API.chatHistory.slice(-10),
        personality:     API.personality,
        profession:      API.profession,
        conversation_id: API.chatSessionId,
      }),
    });

    if (!resp.ok) {
      let errMsg = `Server error (${resp.status})`;
      try {
        const errBody = await resp.json();
        errMsg = errBody.error || errMsg;
      } catch (_) {
        try { errMsg = (await resp.text()).slice(0, 120) || errMsg; } catch (_2) {}
      }
      if (bubble) bubble.innerHTML = `<span style="color:var(--danger)">❌ ${_escHtml(errMsg)}</span>`;
      return;
    }

    // Read SSE stream token-by-token
    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();   // keep incomplete line for next iteration

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw) continue;
        try {
          const chunk = JSON.parse(raw);
          if (chunk.token) {
            fullReply += chunk.token;
            // Render incrementally (simple HTML-safe newline handling)
            if (bubble) bubble.innerHTML = _escHtml(fullReply).replace(/\n/g, '<br>');
            area.scrollTop = area.scrollHeight;
          }
          if (chunk.done) {
            loadChatSessions();   // refresh sidebar with new/updated conversation
            break;
          }
        } catch (_) {}
      }
    }

    API.chatHistory.push({ role: 'assistant', content: fullReply });

  } catch (err) {
    if (bubble) bubble.innerHTML =
      `<span style="color:var(--danger)">❌ ${_escHtml(err.message)}</span>`;
  }
}

// Allow Enter key in chat input
document.addEventListener('DOMContentLoaded', () => {
  const inp = document.getElementById('chat-input');
  if (inp) {
    inp.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
    });
  }
});

// ── Session history ───────────────────────────────────────────────────────────
async function loadSessionHistory() {
  try {
    const token = window.authToken || localStorage.getItem('soulspeak_token') || '';
    const headers = token ? { 'Authorization': 'Bearer ' + token } : {};
    const r    = await fetch('/api/sessions', { headers });
    const list = await r.json();
    const container = document.getElementById('session-history-list') ||
                      document.querySelector('.history-list');
    if (!container || !list.length) return;

    container.innerHTML = list.map(s => `
      <div class="history-item" onclick="loadSession('${s.id}')">
        <div class="history-title">${s.date} · ${s.duration_fmt}</div>
        <div class="history-date">${s.profession} · ${s.source||''}</div>
        <div class="history-score">Overall ${s.overall}%</div>
        <span class="history-del" onclick="deleteSessionApi(event,'${s.id}')">🗑️</span>
      </div>`).join('');
  } catch (_) {}
}

async function deleteSessionApi(e, id) {
  e.stopPropagation();
  await fetch('/api/sessions/' + id, { method: 'DELETE' });
  loadSessionHistory();
  showToast('🗑️ Session deleted.', 'danger');
}

// Hook switchView to auto-load real data per view
const _origSV = window.switchView;
if (typeof _origSV === 'function') {
  window.switchView = function(view, el) {
    _origSV(view, el);
    if (view === 'history') loadSessionHistory();
    if (view === 'chat') {
      loadLatestPersonalityForChat();
      loadChatSessions();
    }
  };
}

// Hook showPage to load admin data when the admin portal is opened
const _origSP = window.showPage;
if (typeof _origSP === 'function') {
  window.showPage = function(id) {
    _origSP(id);
    if (id === 'admin') {
      loadAdminStats();
      loadAdminUsers();
    }
  };
}

// Hook switchAdminView — pass clickedEl through and reload relevant data
const _origSAV = window.switchAdminView;
window.switchAdminView = function(view, clickedEl) {
  if (typeof _origSAV === 'function') _origSAV(view, clickedEl);
  if (view === 'overview') loadAdminStats();
  if (view === 'users')    loadAdminUsers();
  if (view === 'model')    loadAdminStats();
  if (view === 'usage')    loadAdminStats();
};

// ── Admin panel data loaders ─────────────────────────────────────────────────

async function loadAdminStats() {
  try {
    const token = window.authToken || localStorage.getItem('soulspeak_token') || '';
    const headers = token ? { 'Authorization': 'Bearer ' + token } : {};
    const r = await fetch('/api/admin/stats', { headers });
    if (!r.ok) { console.warn('[admin] stats fetch failed:', r.status); return; }
    const d = await r.json();

    // Overview stat cards
    const _s = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    _s('admin-stat-users',    d.users);
    _s('admin-stat-sessions', d.sessions);
    _s('admin-stat-chats',    d.chat_messages);
    _s('admin-stat-ollama',   d.ollama_online ? '🟢 Online' : '🔴 Offline');

    // Analysis source bars
    const barR = document.getElementById('admin-bar-recording');
    const barU = document.getElementById('admin-bar-upload');
    if (barR) { barR.style.width = d.recording_pct + '%'; barR.textContent = d.recording_pct + '%'; }
    if (barU) { barU.style.width = d.upload_pct    + '%'; barU.textContent = d.upload_pct    + '%'; }

    // Top professions in overview
    const profBars = document.getElementById('admin-prof-bars');
    if (profBars && d.top_profs && d.top_profs.length) {
      const maxC = d.top_profs[0].count || 1;
      const GRAD = ['#8b5cf6,#6d28d9','#06b6d4,#0891b2','#f59e0b,#d97706','#10b981,#059669','#ef4444,#dc2626'];
      profBars.innerHTML = d.top_profs.map((p, i) => {
        const pct = Math.round(p.count / maxC * 100);
        return `<div class="chart-bar-h">
          <div class="chart-bar-label">${_escHtml(p.name)}</div>
          <div class="chart-bar-track">
            <div class="chart-bar-fill" style="width:${pct}%;background:linear-gradient(90deg,${GRAD[i]||GRAD[0]})">${p.count}</div>
          </div></div>`;
      }).join('');
    }

    // Model cards
    _s('admin-model-wav2vec2', d.wav2vec2);
    _s('admin-model-bert',     d.bert);
    _s('admin-model-whisper',  d.whisper);
    _s('admin-model-ollama',   d.ollama_model || (d.ollama_online ? 'Online' : 'Offline'));

    // Usage analytics
    _s('admin-usage-sessions-today', d.sessions_today);
    _s('admin-usage-chats-total',    d.chat_messages);
    _s('admin-usage-avg-score',      d.avg_score ? d.avg_score + '%' : '—');
    _s('admin-usage-users-total',    d.users);

    // Usage prof bars
    const usageProfBars = document.getElementById('admin-usage-prof-bars');
    if (usageProfBars && d.top_profs && d.top_profs.length) {
      const maxC = d.top_profs[0].count || 1;
      const GRAD = ['#8b5cf6,#6d28d9','#06b6d4,#0891b2','#f59e0b,#d97706','#10b981,#059669','#ef4444,#dc2626'];
      usageProfBars.innerHTML = d.top_profs.map((p, i) => {
        const pct = Math.round(p.count / maxC * 100);
        return `<div class="chart-bar-h">
          <div class="chart-bar-label">${_escHtml(p.name)}</div>
          <div class="chart-bar-track">
            <div class="chart-bar-fill" style="width:${pct}%;background:linear-gradient(90deg,${GRAD[i]||GRAD[0]})">${p.count}</div>
          </div></div>`;
      }).join('');
    } else if (usageProfBars) {
      usageProfBars.innerHTML = '<div style="text-align:center;padding:20px;color:var(--text3);font-size:0.85rem">No session data yet</div>';
    }

  } catch (e) { console.warn('[admin] loadAdminStats error:', e); }
}

async function loadAdminUsers() {
  const tbody = document.getElementById('admin-users-tbody');
  if (!tbody) return;
  try {
    const token = window.authToken || localStorage.getItem('soulspeak_token') || '';
    const headers = token ? { 'Authorization': 'Bearer ' + token } : {};
    const r = await fetch('/api/admin/users', { headers });
    if (!r.ok) { console.warn('[admin] users fetch failed:', r.status); tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:20px;color:var(--danger)">Failed to load users</td></tr>'; return; }
    const d = await r.json();
    if (!d.users || !d.users.length) {
      tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:20px;color:var(--text3)">No users yet</td></tr>';
      return;
    }
    tbody.innerHTML = d.users.map(u => `
      <tr>
        <td><input type="checkbox"></td>
        <td>${_escHtml(u.name)}</td>
        <td>${_escHtml(u.email)}</td>
        <td>${_escHtml(u.profession)}</td>
        <td>${u.session_count}</td>
        <td>${u.last_active || u.created_at || '—'}</td>
        <td><span class="status-badge ${u.session_count > 0 ? 'status-active' : 'status-inactive'}">${u.session_count > 0 ? '● Active' : '○ Inactive'}</span></td>
        <td><button class="btn btn-ghost btn-sm" onclick="showToast('User: ${_escHtml(u.name)}','info')">View</button></td>
      </tr>`).join('');
  } catch (_) {
    tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:20px;color:var(--danger)">Error loading users</td></tr>';
  }
}

// Load the latest session's personality scores into API state before chat
async function loadLatestPersonalityForChat() {
  try {
    const token = window.authToken || localStorage.getItem('soulspeak_token') || '';
    const headers = token ? { 'Authorization': 'Bearer ' + token } : {};
    const r = await fetch('/api/sessions', { headers });
    const sessions = await r.json();
    if (Array.isArray(sessions) && sessions.length > 0) {
      const latest = sessions[0]; // already sorted newest-first
      if (latest.scores && Object.keys(latest.scores).length > 0) {
        API.personality = latest.scores;
      }
      if (latest.profession) API.profession = latest.profession;
      // Update the initial greeting in the chat area if it hasn't been replied to yet
      const area = document.getElementById('chat-messages');
      if (area) {
        const firstBubble = area.querySelector('.chat-msg.bot .chat-bubble');
        if (firstBubble && !API.chatHistory.length) {
          const savedUser = JSON.parse(localStorage.getItem('soulspeak_user') || 'null');
          const firstName = savedUser && savedUser.name ? savedUser.name.split(' ')[0] : 'there';
          firstBubble.innerHTML = `Hello, <strong>${firstName}</strong>! 👋 I&rsquo;m your <strong>SoulSpeak AI Coach</strong>. I can see your latest session &mdash; Overall <strong>${latest.overall || '?'}%</strong> on ${latest.date || ''}. Ask me anything about your personality profile, your progress over time, or get profession-specific coaching tailored to your role as a <em>${API.profession}</em>.`;
        }
      }
    }
  } catch (_) {}
}

// ── Chat conversation management ─────────────────────────────────────────────

async function loadChatSessions() {
  const container = document.getElementById('chat-sessions-list');
  if (!container) return;
  try {
    const token = window.authToken || localStorage.getItem('soulspeak_token') || '';
    const headers = token ? { 'Authorization': 'Bearer ' + token } : {};
    const r = await fetch('/api/chat/sessions', { headers });
    const sessions = await r.json();
    if (!Array.isArray(sessions) || sessions.length === 0) {
      container.innerHTML = '<div style="text-align:center;padding:24px 12px;color:var(--text3);font-size:0.82rem">No previous chats yet</div>';
      return;
    }
    container.innerHTML = sessions.map(s => {
      const isActive = API.chatSessionId === s.session_id;
      const date = s.last_msg_at ? new Date(s.last_msg_at).toLocaleDateString() : '';
      const bg = isActive ? 'rgba(139,92,246,0.12)' : 'transparent';
      const border = isActive ? '1px solid var(--accent)' : '1px solid transparent';
      return `<div class="chat-session-item"
        onclick="loadChatSession('${s.session_id}')"
        style="padding:10px 12px;border-radius:8px;margin-bottom:4px;cursor:pointer;background:${bg};border:${border};transition:background 0.2s;position:relative"
        onmouseover="this.style.background='rgba(139,92,246,0.08)'"
        onmouseout="this.style.background='${bg}'">
        <div style="font-size:0.75rem;color:var(--text3);margin-bottom:3px">${date} · ${s.msg_count || 0} msg</div>
        <div style="font-size:0.83rem;color:var(--text);line-height:1.4;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding-right:20px">${_escHtml(s.preview || 'New conversation')}</div>
        <button onclick="event.stopPropagation();deleteChatSession('${s.session_id}')"
          style="position:absolute;top:8px;right:8px;background:none;border:none;cursor:pointer;color:var(--text3);font-size:0.8rem;padding:2px 4px;border-radius:4px"
          title="Delete conversation">🗑️</button>
      </div>`;
    }).join('');
  } catch (_) {}
}

function startNewChat() {
  API.chatSessionId = 'chat_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
  API.chatHistory   = [];
  const area = document.getElementById('chat-messages');
  if (area) {
    const savedUser = JSON.parse(localStorage.getItem('soulspeak_user') || 'null');
    const firstName = savedUser && savedUser.name ? savedUser.name.split(' ')[0] : 'there';
    area.innerHTML = `
      <div class="chat-msg bot">
        <div class="chat-avatar" style="background:linear-gradient(135deg,var(--accent),var(--accent2))">SS</div>
        <div class="chat-bubble">Hello, <strong>${_escHtml(firstName)}</strong>! 👋 What would you like to explore today?</div>
      </div>`;
  }
  loadChatSessions();
}

async function loadChatSession(sid) {
  if (!sid) return;
  API.chatSessionId = sid;
  API.chatHistory   = [];
  const area = document.getElementById('chat-messages');
  if (!area) return;
  area.innerHTML = '<div style="text-align:center;padding:32px;color:var(--text3)">Loading conversation…</div>';
  try {
    const token = window.authToken || localStorage.getItem('soulspeak_token') || '';
    const headers = token ? { 'Authorization': 'Bearer ' + token } : {};
    const r = await fetch('/api/chat/history?conversation_id=' + encodeURIComponent(sid) + '&limit=100', { headers });
    const msgs = await r.json();
    if (!Array.isArray(msgs) || msgs.length === 0) {
      area.innerHTML = '<div style="text-align:center;padding:32px;color:var(--text3)">Empty conversation.</div>';
      return;
    }
    area.innerHTML = msgs.map(m => {
      if (m.role === 'user') {
        return `<div class="chat-msg user"><div class="avatar" style="width:32px;height:32px;font-size:0.75rem">SC</div><div class="chat-bubble">${_escHtml(m.content)}</div></div>`;
      }
      return `<div class="chat-msg bot"><div class="chat-avatar" style="background:linear-gradient(135deg,var(--accent),var(--accent2))">SS</div><div class="chat-bubble">${_escHtml(m.content).replace(/\n/g,'<br>')}</div></div>`;
    }).join('');
    API.chatHistory = msgs.map(m => ({ role: m.role, content: m.content }));
    area.scrollTop = area.scrollHeight;
  } catch (_) {
    area.innerHTML = '<div style="text-align:center;padding:32px;color:var(--danger)">Failed to load conversation.</div>';
  }
  loadChatSessions();
}

async function deleteChatSession(sid) {
  if (!confirm('Delete this conversation? This cannot be undone.')) return;
  try {
    const token = window.authToken || localStorage.getItem('soulspeak_token') || '';
    await fetch('/api/chat/sessions/' + encodeURIComponent(sid), {
      method: 'DELETE',
      headers: token ? { 'Authorization': 'Bearer ' + token } : {},
    });
    if (API.chatSessionId === sid) startNewChat();
    else loadChatSessions();
    if (typeof showToast === 'function') showToast('Conversation deleted.', 'danger');
  } catch (_) {}
}

// ── Profession card updater ───────────────────────────────────────────────────

function _updateProfessionCard(profession, scores) {
  const tagEl   = document.getElementById('prof-tag');
  const titleEl = document.getElementById('prof-title');
  const bodyEl  = document.getElementById('prof-body');
  if (!tagEl && !titleEl && !bodyEl) return;

  const p = (profession || '').toLowerCase();
  let cat = 'general';
  if (p.match(/doctor|health|nurs|medic|therap|physician|clinical/))         cat = 'healthcare';
  else if (p.match(/teach|educat|professor|lectur|instruc/))                  cat = 'teacher';
  else if (p.match(/engineer|develop|software|tech|program|devop/))           cat = 'engineer';
  else if (p.match(/business|manag|execut|director|ceo|coo|cfo/))            cat = 'business';
  else if (p.match(/legal|lawyer|attorney|solicitor|barrister|law/))          cat = 'legal';
  else if (p.match(/sales|market/))                                           cat = 'sales';
  else if (p.match(/student|undergrad|postgrad|academic|researcher/))         cat = 'student';
  else if (p.match(/speaker|present|host|anchor|broadcast/))                  cat = 'speaker';
  else if (p.match(/counsel|social work|psychol|coach|mentor/))               cat = 'counsellor';
  else if (p.match(/entrepreneur|founder|startup|venture/))                   cat = 'entrepreneur';

  const META = {
    healthcare:   { icon:'🏥', title:'How your profile maps to clinical communication',        demands:'Healthcare professionals must convey empathy under pressure, maintain calm authority, and simplify complex information for patients — requiring high Agreeableness, emotional stability, and measured Conscientiousness.' },
    teacher:      { icon:'📚', title:'How your profile maps to classroom presence',            demands:'Educators need engaging delivery, patience, and the ability to motivate diverse learners through clear explanations — drawing on Openness, Agreeableness, and confident Extraversion.' },
    engineer:     { icon:'💻', title:'How your profile maps to technical communication',       demands:'Technical professionals must explain complex ideas to non-technical audiences, lead reviews, and collaborate across teams — requiring clarity, Conscientiousness, and measured confidence.' },
    business:     { icon:'💼', title:'How your profile maps to leadership communication',      demands:'Business leaders must inspire teams, communicate decisions concisely, navigate conflict, and project authority — calling on Extraversion, low Neuroticism, and calibrated Agreeableness.' },
    legal:        { icon:'⚖️', title:'How your profile maps to legal communication',           demands:'Legal professionals must structure arguments precisely, project authority, build client trust, and remain composed under cross-examination — requiring Conscientiousness and controlled Extraversion.' },
    sales:        { icon:'📈', title:'How your profile maps to persuasive communication',      demands:'Sales professionals must build rapport instantly, pitch confidently, read emotional signals, and close — drawing on Extraversion, Agreeableness, and resilient low Neuroticism.' },
    student:      { icon:'🎓', title:'How your profile maps to academic communication',        demands:'Students benefit most from strong presentation skills, interview confidence, and professional communication habits — where Openness and Conscientiousness are especially valuable.' },
    speaker:      { icon:'🎤', title:'How your profile maps to stage presence',                demands:'Public speakers need commanding vocal variety, authentic audience connection, confident storytelling, and the ability to hold attention at scale — requiring strong Extraversion and low Neuroticism.' },
    counsellor:   { icon:'🤝', title:'How your profile maps to supportive communication',      demands:'Counsellors must practise deep active listening, respond with empathy, hold space for difficult conversations, and build trust over time — anchored in high Agreeableness and emotional regulation.' },
    entrepreneur: { icon:'🚀', title:'How your profile maps to entrepreneurial communication', demands:'Founders must pitch to investors, inspire early teams, network authentically, and communicate vision with conviction — calling on Extraversion, Openness, and resilient low Neuroticism.' },
    general:      { icon:'📋', title:'How your profile maps to professional communication',    demands:'Strong professional communicators balance clarity, empathy, and confidence — drawing on all five Big Five dimensions to adapt effectively to any context.' },
  };
  const m = META[cat] || META.general;

  const TCOLS = { Openness:'#8b5cf6', Conscientiousness:'#06b6d4', Extraversion:'#f59e0b', Agreeableness:'#10b981', Neuroticism:'#ef4444' };
  const highlights = Object.entries(scores || {})
    .filter(([,v]) => v >= 65 || v <= 35)
    .map(([t,v]) => {
      const lbl = v >= 80 ? 'Exceptional' : v >= 65 ? 'High' : v <= 20 ? 'Very Low' : 'Low';
      return `<strong style="color:${TCOLS[t]||'var(--accent)'}">${t} (${v}%) — ${lbl}</strong>`;
    }).join(', ');

  const top = Object.entries(scores || {}).sort((a,b)=>b[1]-a[1])[0];
  const encourage = top
    ? `Your strongest trait — <strong>${top[0]} at ${top[1]}%</strong> — is a real professional asset. Channel it deliberately and use the SoulSpeak AI Coach to build a targeted plan around your full profile.`
    : 'Complete more voice sessions to unlock deeper personalised profession insights from your SoulSpeak AI Coach.';

  if (tagEl)   tagEl.textContent  = `${m.icon} Profession Insight · ${profession}`;
  if (titleEl) titleEl.textContent = m.title;
  if (bodyEl)  bodyEl.innerHTML   =
    `<strong>What your profession demands:</strong> ${m.demands}<br><br>` +
    (highlights ? `<strong>What you have:</strong> ${highlights}.<br><br>` : '') +
    `<strong>Encouragement:</strong> ${encourage}`;
}

// loadSession — load a single session into results view
async function loadSession(id) {
  if (!id) return;
  try {
    const r   = await fetch('/api/sessions/' + id, { headers: { 'Authorization': 'Bearer ' + (window.authToken || '') } });
    const res = await r.json();
    if (!res.error) {
      API.personality = res.scores;
      _populateResultsView(res);
      if (typeof switchView === 'function') switchView('results');
    }
  } catch (_) {}
}

// ── MediaRecorder integration ─────────────────────────────────────────────────
// Captures the recorded blob so sendForAnalysis can attach it.
window._recordedBlob    = null;
window._recordSeconds   = 0;
window._mediaRecorder   = null;
window._recordedChunks  = [];

function startMediaCapture() {
  if (!navigator.mediaDevices) return;
  navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
    window._recordedChunks = [];
    window._mediaRecorder  = new MediaRecorder(stream);
    window._mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0) window._recordedChunks.push(e.data);
    };
    window._mediaRecorder.onstop = () => {
      window._recordedBlob = new Blob(window._recordedChunks, { type: 'audio/webm' });
      console.log('[recorder] blob ready:', window._recordedBlob.size, 'bytes');
      stream.getTracks().forEach(t => t.stop());
    };
    window._mediaRecorder.start();
  }).catch(err => console.warn('[recorder] mic access denied:', err));
}

function stopMediaCapture() {
  if (window._mediaRecorder && window._mediaRecorder.state !== 'inactive') {
    window._mediaRecorder.stop();
  }
}

// Wrap the original toggleRecording to also start/stop MediaRecorder
const _origToggle = window.toggleRecording;
window.toggleRecording = function(isRecording) {
  if (isRecording) {
    window._recordedBlob = null;
    startMediaCapture();
  } else {
    window._recordSeconds = window.recordSeconds || 30;
    stopMediaCapture();
  }
  if (typeof _origToggle === 'function') _origToggle(isRecording);
};

// ── Utilities ─────────────────────────────────────────────────────────────────
function _escHtml(t) {
  return String(t)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}
</script>
"""


def inject_bridge(html: str) -> str:
    idx = html.rfind("</body>")
    if idx == -1:
        return html + API_BRIDGE
    return html[:idx] + API_BRIDGE + "\n</body>" + html[idx + 7:]


def main():
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    os.makedirs("static",      exist_ok=True)

    if not os.path.exists(SOURCE_HTML):
        print(f"\n❌  Source HTML not found: {SOURCE_HTML}")
        print("    Place your SoulSpeak HTML file at that path and rerun.\n")
        return

    with open(SOURCE_HTML, encoding="utf-8") as f:
        html = f.read()

    patched = inject_bridge(html)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(patched)

    print(f"\n✅  Template generated → {OUTPUT_HTML}")
    print("✅  static/ folder ready  (place 1.png and 3.png inside)\n")
    print("  Quick start:")
    print("  ─────────────────────────────────────────────────────")
    print("  1. pip install -r requirements.txt")
    print("  2. Place model files in the project root:")
    print("       wav2vec2_personality_best.pt")
    print("       bert_personality_best.pt")
    print("  3. Ensure Ollama is running:  ollama serve")
    print("  4. Pull a model if needed:    ollama pull llama3")
    print("  5. python app.py")
    print("  6. Open http://localhost:5000")
    print("  ─────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
