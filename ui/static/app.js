/* ================================================================
   SpotifyCares Support Agent — Frontend JavaScript
================================================================ */

const API = '';  // same-origin

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadExamples();
  loadStats();
  checkHealth();
  setupCharCounter();
  setupKeyboardShortcut();
});

// ── Health check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  const badge = document.getElementById('status-badge');
  const dot = badge.querySelector('.badge-dot');
  const text = badge.querySelector('.badge-text');
  try {
    const r = await fetch(`${API}/api/health`);
    if (r.ok) {
      dot.className = 'badge-dot online';
      text.textContent = 'Agent Ready';
    } else { throw new Error(); }
  } catch {
    dot.className = 'badge-dot offline';
    text.textContent = 'Agent Offline';
  }
}

// ── Load example messages ─────────────────────────────────────────────────────
async function loadExamples() {
  try {
    const r = await fetch(`${API}/api/examples`);
    const examples = await r.json();
    const grid = document.getElementById('examples-grid');
    grid.innerHTML = examples.map(ex => `
      <button class="example-chip" onclick="useExample(${JSON.stringify(ex.message)})">
        ${ex.label}
      </button>
    `).join('');
  } catch {
    document.getElementById('examples-grid').innerHTML =
      '<span style="font-size:12px;color:var(--text-muted)">Could not load examples</span>';
  }
}

function useExample(message) {
  const input = document.getElementById('message-input');
  input.value = message;
  updateCharCount();
  input.focus();
}

// ── Load dataset stats ────────────────────────────────────────────────────────
async function loadStats() {
  const grid = document.getElementById('stats-grid');
  try {
    const r = await fetch(`${API}/api/stats`);
    const stats = await r.json();
    if (stats.error) { grid.innerHTML = `<p style="font-size:12px;color:var(--text-muted)">${stats.error}</p>`; return; }
    const items = [
      { value: fmt(stats.brand_tweets), label: 'Brand Tweets' },
      { value: fmt(stats.customer_tweets), label: 'Customer Tweets' },
      { value: fmt(stats.reconstructed_pairs), label: 'Reconstructed Pairs' },
      { value: fmt(stats.interactions_for_retrieval), label: 'Retrieval-Worthy' },
    ];
    grid.innerHTML = items.map(i => `
      <div class="stat-item">
        <div class="stat-value">${i.value}</div>
        <div class="stat-label">${i.label}</div>
      </div>
    `).join('');
  } catch {
    grid.innerHTML = '<p style="font-size:12px;color:var(--text-muted)">Run prepare_data.py to see stats</p>';
  }
}

function fmt(n) {
  if (n === undefined || n === null) return '—';
  return Number(n).toLocaleString();
}

// ── Char counter ──────────────────────────────────────────────────────────────
function setupCharCounter() {
  const input = document.getElementById('message-input');
  input.addEventListener('input', updateCharCount);
}

function updateCharCount() {
  const input = document.getElementById('message-input');
  document.getElementById('char-count').textContent = input.value.length;
}

// ── Keyboard shortcut (Ctrl/Cmd + Enter) ─────────────────────────────────────
function setupKeyboardShortcut() {
  document.getElementById('message-input').addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') runAgent();
  });
}

// ── Main: Run agent ───────────────────────────────────────────────────────────
async function runAgent() {
  const message = document.getElementById('message-input').value.trim();
  if (!message) { showToast('Please enter a customer message', 'error'); return; }

  showLoading();

  // Simulate step progression
  const steps = ['step-intent', 'step-retrieve', 'step-generate', 'step-escalate'];
  let stepIdx = 0;
  const stepInterval = setInterval(() => {
    if (stepIdx > 0) document.getElementById(steps[stepIdx-1])?.classList.replace('active', 'done');
    if (stepIdx < steps.length) document.getElementById(steps[stepIdx])?.classList.add('active');
    stepIdx++;
    if (stepIdx > steps.length) clearInterval(stepInterval);
  }, 600);

  try {
    const resp = await fetch(`${API}/api/agent`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });

    clearInterval(stepInterval);
    steps.forEach(s => {
      const el = document.getElementById(s);
      el?.classList.remove('active');
      el?.classList.add('done');
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }

    const data = await resp.json();
    renderResult(data);

  } catch (err) {
    clearInterval(stepInterval);
    showError(err.message);
  }
}

// ── Render result ─────────────────────────────────────────────────────────────
function renderResult(data) {
  document.getElementById('loading-state').style.display = 'none';
  document.getElementById('empty-state').style.display = 'none';
  document.getElementById('result-content').style.display = 'flex';

  // Intent
  document.getElementById('res-intent').textContent = data.intent;
  const confPct = Math.round((data.intent_confidence || 0) * 100);
  document.getElementById('res-confidence').textContent = confPct;
  document.getElementById('confidence-fill').style.width = `${confPct}%`;

  // Decision
  const isEscalate = data.decision === 'ESCALATE';
  const badge = document.getElementById('decision-badge');
  badge.className = `decision-badge ${isEscalate ? 'escalate' : 'auto'}`;
  document.getElementById('decision-icon').textContent = isEscalate ? '🚨' : '✅';
  document.getElementById('decision-text').textContent = isEscalate ? 'ESCALATE' : 'AUTO HANDLE';
  document.getElementById('decision-reason').textContent = data.decision_reason;

  // Risk flags
  const flagsCard = document.getElementById('risk-flags-card');
  const flagsList = document.getElementById('risk-flags-list');
  if (data.risk_flags && data.risk_flags.length > 0) {
    flagsCard.style.display = 'block';
    flagsList.innerHTML = data.risk_flags.map(f =>
      `<span class="risk-flag">${escHtml(f)}</span>`
    ).join('');
  } else {
    flagsCard.style.display = 'none';
  }

  // Draft reply
  document.getElementById('res-reply').textContent = data.draft_reply || '(No reply generated)';

  // Grounding score
  const gs = (data.grounding_score || 0);
  const gsColor = gs >= 0.6 ? 'var(--green)' : gs >= 0.4 ? 'var(--warn)' : 'var(--escalate)';
  const gsBadge = document.getElementById('grounding-badge');
  gsBadge.textContent = `Grounding: ${Math.round(gs * 100)}%`;
  gsBadge.style.color = gsColor;

  // Latency
  document.getElementById('latency-badge').textContent = `${data.latency_ms}ms`;

  // Evidence
  const count = (data.retrieved_evidence || []).length;
  document.getElementById('evidence-count').textContent = count;
  renderEvidence(data.retrieved_evidence || []);
}

function renderEvidence(evidence) {
  const list = document.getElementById('evidence-list');
  if (!evidence.length) {
    list.innerHTML = `<p class="evidence-no-results">No historical evidence retrieved for this intent.</p>`;
    return;
  }
  list.innerHTML = evidence.map((ev, i) => {
    const sim = Math.round((ev.similarity || 0) * 100);
    const simColor = sim >= 60 ? 'var(--green)' : sim >= 40 ? 'var(--warn)' : 'var(--escalate)';
    return `
      <div class="evidence-item">
        <div class="evidence-header">
          <div class="evidence-rank">${i + 1}</div>
          <span class="evidence-sim" style="color:${simColor}">Sim: ${sim}%</span>
          ${ev.resolution_type ? `<span class="evidence-type">${escHtml(ev.resolution_type)}</span>` : ''}
          ${ev.intent ? `<span class="evidence-type" style="background:rgba(255,255,255,0.05);color:var(--text-muted)">${escHtml(ev.intent.split('/')[0].trim())}</span>` : ''}
        </div>
        <p class="evidence-msg">💬 ${escHtml((ev.customer_message || '').substring(0, 120))}${(ev.customer_message || '').length > 120 ? '…' : ''}</p>
        <p class="evidence-reply">🎵 ${escHtml((ev.brand_response || '').substring(0, 180))}${(ev.brand_response || '').length > 180 ? '…' : ''}</p>
      </div>
    `;
  }).join('');
}

// ── UI states ─────────────────────────────────────────────────────────────────
function showLoading() {
  document.getElementById('empty-state').style.display = 'none';
  document.getElementById('result-content').style.display = 'none';
  document.getElementById('loading-state').style.display = 'flex';
  document.getElementById('submit-btn').disabled = true;

  // Reset steps
  ['step-intent','step-retrieve','step-generate','step-escalate'].forEach(s => {
    const el = document.getElementById(s);
    el.className = 'step';
  });
}

function showError(msg) {
  document.getElementById('loading-state').style.display = 'none';
  document.getElementById('empty-state').style.display = 'flex';
  document.getElementById('submit-btn').disabled = false;
  showToast(`Error: ${msg}`, 'error');
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function showToast(msg, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// Re-enable submit after result renders
const observer = new MutationObserver(() => {
  const rc = document.getElementById('result-content');
  if (rc && rc.style.display !== 'none') {
    document.getElementById('submit-btn').disabled = false;
  }
});
observer.observe(document.getElementById('result-content'), { attributes: true });
