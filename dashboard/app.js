/**
 * app.js — introspect dashboard application.
 *
 * Handles API communication, chart rendering (custom Canvas 2D),
 * WebSocket live updates, and UI state management.
 *
 * All charts are rendered with pure Canvas 2D — no external charting
 * libraries. This demonstrates low-level graphics programming capability.
 */

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════════

const API_BASE = window.location.origin;
const WS_URL = `ws://${window.location.host}/ws/live`;

// Color palette (matching CSS design tokens).
const COLORS = {
    blue:     'hsl(217, 91%, 60%)',
    cyan:     'hsl(187, 80%, 55%)',
    green:    'hsl(152, 69%, 53%)',
    red:      'hsl(0, 72%, 60%)',
    yellow:   'hsl(43, 96%, 58%)',
    purple:   'hsl(265, 70%, 65%)',
    text:     'hsl(215, 15%, 60%)',
    muted:    'hsl(215, 10%, 40%)',
    grid:     'hsla(225, 12%, 18%, 0.6)',
    surface:  'hsl(225, 16%, 13%)',
};


// ═══════════════════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════════════════

let ws = null;
let runs = [];
let consistencyData = [];
let driftData = [];
let currentStepData = [];
let isEvaluating = false;


// ═══════════════════════════════════════════════════════════════════════════════
// API Client
// ═══════════════════════════════════════════════════════════════════════════════

async function apiGet(path) {
    const res = await fetch(`${API_BASE}${path}`);
    if (!res.ok) throw new Error(`API ${path}: ${res.status}`);
    return res.json();
}

async function apiPost(path, params = {}) {
    const query = new URLSearchParams(params).toString();
    const url = query ? `${API_BASE}${path}?${query}` : `${API_BASE}${path}`;
    const res = await fetch(url, { method: 'POST' });
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `API ${path}: ${res.status}`);
    }
    return res.json();
}


// ═══════════════════════════════════════════════════════════════════════════════
// WebSocket
// ═══════════════════════════════════════════════════════════════════════════════

function connectWebSocket() {
    const indicator = document.getElementById('ws-indicator');
    const label = document.getElementById('ws-indicator')?.querySelector('.ws-label');

    try {
        ws = new WebSocket(WS_URL);
    } catch {
        return;
    }

    ws.onopen = () => {
        indicator.classList.remove('ws-disconnected');
        indicator.classList.add('ws-connected');
        if (label) label.textContent = 'Connected';
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'step') {
                handleLiveStep(data);
            }
        } catch { /* ignore malformed messages */ }
    };

    ws.onclose = () => {
        indicator.classList.remove('ws-connected');
        indicator.classList.add('ws-disconnected');
        if (label) label.textContent = 'Disconnected';
        // Reconnect after 3 seconds.
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = () => {
        ws.close();
    };
}

function handleLiveStep(data) {
    const overlay = document.getElementById('live-overlay');
    overlay.classList.remove('hidden');

    // Update progress bar.
    const progress = ((data.step_index + 1) / data.total_steps) * 100;
    document.getElementById('live-bar-fill').style.width = `${progress}%`;
    document.getElementById('live-step-label').textContent =
        `Step ${data.step_index + 1}/${data.total_steps}`;

    // Update stats.
    document.getElementById('live-tokens').textContent = data.tokens_unmasked;
    document.getElementById('live-confidence').textContent =
        data.confidence_mean?.toFixed(3) ?? '—';
    document.getElementById('live-latency').textContent =
        `${data.elapsed_ms?.toFixed(1) ?? '—'}ms`;

    // Collect step data for waterfall chart.
    currentStepData.push(data);
}


// ═══════════════════════════════════════════════════════════════════════════════
// Data Loading
// ═══════════════════════════════════════════════════════════════════════════════

async function loadDashboardData() {
    try {
        [runs, consistencyData, driftData] = await Promise.all([
            apiGet('/api/runs?limit=20'),
            apiGet('/api/consistency/trend?limit=30'),
            apiGet('/api/drift/history?limit=30'),
        ]);

        updateVitals();
        renderRunsTable();
        renderConsistencyChart();
        renderDriftChart();

        // Load waterfall for the latest run.
        if (runs.length > 0 && runs[0].run_id) {
            const steps = await apiGet(`/api/runs/${runs[0].run_id}/steps`);
            renderWaterfallChart(steps);
            renderConfidenceChart(steps);
        }
    } catch (err) {
        console.warn('Dashboard data load failed:', err.message);
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// Vitals Cards
// ═══════════════════════════════════════════════════════════════════════════════

function updateVitals() {
    // Total runs.
    document.getElementById('val-total-runs').textContent = runs.length;

    // Average ICS.
    const icsScores = runs.filter(r => r.ics_score != null).map(r => r.ics_score);
    const avgIcs = icsScores.length > 0
        ? (icsScores.reduce((a, b) => a + b, 0) / icsScores.length)
        : null;
    document.getElementById('val-avg-ics').textContent =
        avgIcs != null ? avgIcs.toFixed(3) : '—';
    setTrendColor('trend-avg-ics', avgIcs, 0.85);

    // Pass rate.
    const completed = runs.filter(r => r.status === 'completed');
    const passed = completed.filter(r => r.passed);
    const passRate = completed.length > 0 ? (passed.length / completed.length) : null;
    document.getElementById('val-pass-rate').textContent =
        passRate != null ? `${(passRate * 100).toFixed(0)}%` : '—';
    setTrendColor('trend-pass-rate', passRate, 0.8);

    // Average latency.
    const latencies = runs.filter(r => r.total_elapsed_ms != null).map(r => r.total_elapsed_ms);
    const avgLat = latencies.length > 0
        ? (latencies.reduce((a, b) => a + b, 0) / latencies.length)
        : null;
    document.getElementById('val-avg-latency').textContent =
        avgLat != null ? `${avgLat.toFixed(0)}ms` : '—';
}

function setTrendColor(elementId, value, threshold) {
    const el = document.getElementById(elementId);
    if (!el || value == null) return;
    if (value >= threshold) {
        el.textContent = '● Healthy';
        el.className = 'vital-trend trend-up';
    } else {
        el.textContent = '● Below threshold';
        el.className = 'vital-trend trend-down';
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// Runs Table
// ═══════════════════════════════════════════════════════════════════════════════

function renderRunsTable() {
    const tbody = document.getElementById('runs-tbody');
    if (runs.length === 0) {
        tbody.innerHTML = `<tr class="empty-row"><td colspan="8">No evaluation runs yet. Click "New Evaluation" to start.</td></tr>`;
        return;
    }

    tbody.innerHTML = runs.map(run => {
        const statusBadge = getStatusBadge(run.status);
        const resultBadge = run.passed != null
            ? (run.passed ? '<span class="badge badge-pass">PASS</span>' : '<span class="badge badge-fail">FAIL</span>')
            : '<span class="badge badge-pending">—</span>';
        const time = run.started_at ? new Date(run.started_at * 1000).toLocaleTimeString() : '—';

        return `<tr data-run-id="${run.run_id}">
            <td style="font-family: var(--font-mono); font-size: 0.8rem;">${run.run_id}</td>
            <td>${statusBadge}</td>
            <td>${run.ics_score != null ? run.ics_score.toFixed(4) : '—'}</td>
            <td>${run.drift_score != null ? run.drift_score.toFixed(6) : '—'}</td>
            <td>${run.total_steps ?? '—'}</td>
            <td>${run.total_elapsed_ms != null ? run.total_elapsed_ms.toFixed(1) + 'ms' : '—'}</td>
            <td>${resultBadge}</td>
            <td style="color: var(--text-muted);">${time}</td>
        </tr>`;
    }).join('');

    // Click handler for run details.
    tbody.querySelectorAll('tr[data-run-id]').forEach(row => {
        row.addEventListener('click', async () => {
            const runId = row.dataset.runId;
            const steps = await apiGet(`/api/runs/${runId}/steps`);
            renderWaterfallChart(steps);
            renderConfidenceChart(steps);
        });
    });
}

function getStatusBadge(status) {
    const classMap = {
        completed: 'badge-completed',
        running: 'badge-running',
        failed: 'badge-fail',
        pending: 'badge-pending',
    };
    const cls = classMap[status] || 'badge-pending';
    return `<span class="badge ${cls}">${status}</span>`;
}


// ═══════════════════════════════════════════════════════════════════════════════
// Chart Rendering (Pure Canvas 2D)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Setup a canvas for high-DPI rendering.
 * @param {HTMLCanvasElement} canvas
 * @returns {{ ctx: CanvasRenderingContext2D, w: number, h: number }}
 */
function setupCanvas(canvas) {
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = rect.width;
    const h = 220;

    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    return { ctx, w, h };
}

/**
 * Draw axis grid lines and labels.
 */
function drawGrid(ctx, w, h, opts = {}) {
    const { yMin = 0, yMax = 1, ySteps = 5, pad = { top: 20, right: 20, bottom: 30, left: 50 } } = opts;
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 0.5;
    ctx.fillStyle = COLORS.muted;
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (let i = 0; i <= ySteps; i++) {
        const y = pad.top + (plotH * i / ySteps);
        const val = yMax - (yMax - yMin) * (i / ySteps);

        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(w - pad.right, y);
        ctx.stroke();

        ctx.fillText(
            typeof opts.yFormat === 'function' ? opts.yFormat(val) : val.toFixed(2),
            pad.left - 8,
            y
        );
    }

    return { plotW, plotH, pad };
}


// ── Consistency Timeline ─────────────────────────────────────────────────────

function renderConsistencyChart() {
    const canvas = document.getElementById('chart-consistency');
    if (!canvas) return;

    const { ctx, w, h } = setupCanvas(canvas);
    ctx.clearRect(0, 0, w, h);

    const data = [...consistencyData].reverse();
    if (data.length === 0) {
        drawEmptyState(ctx, w, h, 'No consistency data');
        return;
    }

    const scores = data.map(d => d.ics_score);
    const yMin = Math.max(0, Math.min(...scores) - 0.05);
    const yMax = Math.min(1, Math.max(...scores) + 0.05);

    const { plotW, plotH, pad } = drawGrid(ctx, w, h, { yMin, yMax, ySteps: 5 });

    // Threshold line.
    const threshold = data[0]?.threshold ?? 0.85;
    const threshY = pad.top + plotH * (1 - (threshold - yMin) / (yMax - yMin));
    ctx.strokeStyle = COLORS.red;
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.left, threshY);
    ctx.lineTo(w - pad.right, threshY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = COLORS.red;
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('threshold', w - pad.right - 55, threshY - 6);

    // Gradient fill under the line.
    const gradient = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
    gradient.addColorStop(0, 'hsla(187, 80%, 55%, 0.2)');
    gradient.addColorStop(1, 'hsla(187, 80%, 55%, 0.02)');

    ctx.beginPath();
    for (let i = 0; i < scores.length; i++) {
        const x = pad.left + (i / Math.max(scores.length - 1, 1)) * plotW;
        const y = pad.top + plotH * (1 - (scores[i] - yMin) / (yMax - yMin));
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    // Close path for fill.
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Line.
    ctx.beginPath();
    for (let i = 0; i < scores.length; i++) {
        const x = pad.left + (i / Math.max(scores.length - 1, 1)) * plotW;
        const y = pad.top + plotH * (1 - (scores[i] - yMin) / (yMax - yMin));
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = COLORS.cyan;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Data points.
    for (let i = 0; i < scores.length; i++) {
        const x = pad.left + (i / Math.max(scores.length - 1, 1)) * plotW;
        const y = pad.top + plotH * (1 - (scores[i] - yMin) / (yMax - yMin));

        ctx.beginPath();
        ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = scores[i] >= threshold ? COLORS.cyan : COLORS.red;
        ctx.fill();
        ctx.strokeStyle = COLORS.surface;
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }
}


// ── Step Waterfall ───────────────────────────────────────────────────────────

function renderWaterfallChart(steps = []) {
    const canvas = document.getElementById('chart-waterfall');
    if (!canvas) return;

    const { ctx, w, h } = setupCanvas(canvas);
    ctx.clearRect(0, 0, w, h);

    if (steps.length === 0) {
        drawEmptyState(ctx, w, h, 'Select a run to view steps');
        return;
    }

    const latencies = steps.map(s => s.elapsed_ms);
    const maxLat = Math.max(...latencies, 0.1);

    const pad = { top: 20, right: 20, bottom: 30, left: 50 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    // Y-axis grid.
    const ySteps = 4;
    ctx.fillStyle = COLORS.muted;
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'right';

    for (let i = 0; i <= ySteps; i++) {
        const y = pad.top + (plotH * i / ySteps);
        const val = maxLat * (1 - i / ySteps);

        ctx.strokeStyle = COLORS.grid;
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(w - pad.right, y);
        ctx.stroke();

        ctx.fillText(`${val.toFixed(1)}ms`, pad.left - 8, y + 3);
    }

    // Bars.
    const barCount = steps.length;
    const barGap = 3;
    const barWidth = Math.max(4, (plotW - barGap * barCount) / barCount);

    for (let i = 0; i < barCount; i++) {
        const x = pad.left + i * (barWidth + barGap);
        const barH = (latencies[i] / maxLat) * plotH;
        const y = pad.top + plotH - barH;

        // Gradient bar.
        const gradient = ctx.createLinearGradient(x, y, x, pad.top + plotH);
        gradient.addColorStop(0, COLORS.blue);
        gradient.addColorStop(1, 'hsla(217, 91%, 60%, 0.3)');

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.roundRect(x, y, barWidth, barH, [3, 3, 0, 0]);
        ctx.fill();

        // Step label.
        if (barCount <= 20 || i % 2 === 0) {
            ctx.fillStyle = COLORS.muted;
            ctx.font = '9px JetBrains Mono, monospace';
            ctx.textAlign = 'center';
            ctx.fillText(`${steps[i].step_index}`, x + barWidth / 2, pad.top + plotH + 14);
        }
    }
}


// ── Drift Monitor ────────────────────────────────────────────────────────────

function renderDriftChart() {
    const canvas = document.getElementById('chart-drift');
    if (!canvas) return;

    const { ctx, w, h } = setupCanvas(canvas);
    ctx.clearRect(0, 0, w, h);

    const data = [...driftData].reverse();
    if (data.length === 0) {
        drawEmptyState(ctx, w, h, 'No drift data');
        return;
    }

    const scores = data.map(d => d.aggregate_drift);
    const yMax = Math.max(...scores, 0.15) * 1.2;

    const { plotW, plotH, pad } = drawGrid(ctx, w, h, {
        yMin: 0,
        yMax,
        ySteps: 4,
        yFormat: v => v.toFixed(3),
    });

    // Gradient fill.
    const gradient = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
    gradient.addColorStop(0, 'hsla(265, 70%, 65%, 0.2)');
    gradient.addColorStop(1, 'hsla(265, 70%, 65%, 0.02)');

    ctx.beginPath();
    for (let i = 0; i < scores.length; i++) {
        const x = pad.left + (i / Math.max(scores.length - 1, 1)) * plotW;
        const y = pad.top + plotH * (1 - scores[i] / yMax);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Line.
    ctx.beginPath();
    for (let i = 0; i < scores.length; i++) {
        const x = pad.left + (i / Math.max(scores.length - 1, 1)) * plotW;
        const y = pad.top + plotH * (1 - scores[i] / yMax);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = COLORS.purple;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Points.
    for (let i = 0; i < scores.length; i++) {
        const x = pad.left + (i / Math.max(scores.length - 1, 1)) * plotW;
        const y = pad.top + plotH * (1 - scores[i] / yMax);

        ctx.beginPath();
        ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = data[i].passed ? COLORS.purple : COLORS.red;
        ctx.fill();
        ctx.strokeStyle = COLORS.surface;
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }
}


// ── Confidence Distribution ──────────────────────────────────────────────────

function renderConfidenceChart(steps = []) {
    const canvas = document.getElementById('chart-confidence');
    if (!canvas) return;

    const { ctx, w, h } = setupCanvas(canvas);
    ctx.clearRect(0, 0, w, h);

    if (steps.length === 0) {
        drawEmptyState(ctx, w, h, 'Select a run to view confidence');
        return;
    }

    const means = steps.map(s => s.confidence_mean);
    const mins = steps.map(s => s.confidence_min);
    const maxs = steps.map(s => s.confidence_max);

    const { plotW, plotH, pad } = drawGrid(ctx, w, h, {
        yMin: 0,
        yMax: 1,
        ySteps: 5,
    });

    const n = steps.length;

    // Confidence range (min-max band).
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
        const x = pad.left + (i / Math.max(n - 1, 1)) * plotW;
        const y = pad.top + plotH * (1 - maxs[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    for (let i = n - 1; i >= 0; i--) {
        const x = pad.left + (i / Math.max(n - 1, 1)) * plotW;
        const y = pad.top + plotH * (1 - mins[i]);
        ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = 'hsla(43, 96%, 58%, 0.1)';
    ctx.fill();

    // Mean line.
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
        const x = pad.left + (i / Math.max(n - 1, 1)) * plotW;
        const y = pad.top + plotH * (1 - means[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = COLORS.yellow;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Points.
    for (let i = 0; i < n; i++) {
        const x = pad.left + (i / Math.max(n - 1, 1)) * plotW;
        const y = pad.top + plotH * (1 - means[i]);
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = COLORS.yellow;
        ctx.fill();
    }

    // X-axis labels.
    ctx.fillStyle = COLORS.muted;
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    for (let i = 0; i < n; i++) {
        if (n <= 20 || i % 2 === 0) {
            const x = pad.left + (i / Math.max(n - 1, 1)) * plotW;
            ctx.fillText(`T${steps[i].step_index}`, x, h - 8);
        }
    }
}


// ── Empty State ──────────────────────────────────────────────────────────────

function drawEmptyState(ctx, w, h, message) {
    ctx.fillStyle = COLORS.muted;
    ctx.font = '13px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(message, w / 2, h / 2);
}


// ═══════════════════════════════════════════════════════════════════════════════
// Evaluation Trigger
// ═══════════════════════════════════════════════════════════════════════════════

async function runEvaluation(config) {
    if (isEvaluating) return;
    isEvaluating = true;
    currentStepData = [];

    const overlay = document.getElementById('live-overlay');
    overlay.classList.remove('hidden');
    document.getElementById('live-bar-fill').style.width = '0%';
    document.getElementById('live-step-label').textContent = 'Starting...';

    try {
        const result = await apiPost('/api/evaluate', config);

        // Hide live overlay after a brief pause.
        setTimeout(() => {
            overlay.classList.add('hidden');
        }, 1500);

        // Refresh all data.
        await loadDashboardData();

        return result;
    } catch (err) {
        console.error('Evaluation failed:', err);
        overlay.classList.add('hidden');
        alert(`Evaluation failed: ${err.message}`);
    } finally {
        isEvaluating = false;
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// Event Handlers
// ═══════════════════════════════════════════════════════════════════════════════

function setupEventHandlers() {
    const btnNewEval = document.getElementById('btn-new-eval');
    const modal = document.getElementById('eval-modal');
    const form = document.getElementById('eval-form');
    const btnCancel = document.getElementById('btn-cancel-eval');

    btnNewEval?.addEventListener('click', () => {
        modal?.showModal();
    });

    btnCancel?.addEventListener('click', () => {
        modal?.close();
    });

    modal?.addEventListener('click', (e) => {
        if (e.target === modal) modal.close();
    });

    form?.addEventListener('submit', async (e) => {
        e.preventDefault();
        modal?.close();

        const config = {
            seq_len: document.getElementById('input-seq-len')?.value || 128,
            num_steps: document.getElementById('input-num-steps')?.value || 16,
            inconsistency_rate: document.getElementById('input-inconsistency')?.value || 0.1,
            vocab_size: document.getElementById('input-vocab-size')?.value || 32000,
        };

        await runEvaluation(config);
    });
}


// ═══════════════════════════════════════════════════════════════════════════════
// Resize handling
// ═══════════════════════════════════════════════════════════════════════════════

let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        renderConsistencyChart();
        renderDriftChart();
        // Re-render waterfall and confidence if we have data.
        if (runs.length > 0 && runs[0].run_id) {
            apiGet(`/api/runs/${runs[0].run_id}/steps`).then(steps => {
                renderWaterfallChart(steps);
                renderConfidenceChart(steps);
            }).catch(() => {});
        }
    }, 200);
});


// ═══════════════════════════════════════════════════════════════════════════════
// Initialization
// ═══════════════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    setupEventHandlers();
    connectWebSocket();
    loadDashboardData();

    // Auto-refresh every 15 seconds.
    setInterval(loadDashboardData, 15000);
});
