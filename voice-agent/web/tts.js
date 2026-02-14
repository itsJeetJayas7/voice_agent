/**
 * TTS Playground — Chatterbox Turbo Test Client
 *
 * Chat-style interface for testing text-to-speech generation.
 * Sends text to the Chatterbox TTS server and plays back audio.
 */

// ── Configuration ─────────────────────────────────────────

const TTS_BASE_URL = 'http://localhost:8001';
const TTS_ENDPOINT = `${TTS_BASE_URL}/v1/audio/speech`;
const HEALTH_ENDPOINT = `${TTS_BASE_URL}/health`;
const REFERENCE_ENDPOINT = `${TTS_BASE_URL}/v1/voice/reference`;
const STORAGE_KEY = 'tts-playground';
const HEALTH_INTERVAL_MS = 15_000;
const REQUEST_TIMEOUT_MS = 60_000;
const MAX_REFERENCE_SIZE = 10 * 1024 * 1024; // 10 MB

const DEFAULT_SETTINGS = {
    voice: 'default',
    response_format: 'wav',
    exaggeration: 0.5,
    cfg_weight: 0.5,
    temperature: 0.8,
    cloneEnabled: false,
};

// ── State ─────────────────────────────────────────────────

let messages = [];          // { id, text, status, audioUrl, duration, error }
let settings = { ...DEFAULT_SETTINGS };
let activeController = null; // AbortController for in-flight request
let activeMessageId = null;  // ID of message currently generating
let serverOnline = false;

// Voice clone state (not persisted across page loads — reference is server-side in memory)
let cloneReferenceId = null;
let cloneFilename = null;
let cloneDuration = null;

// ── DOM Helpers ───────────────────────────────────────────

function $(id) {
    return document.getElementById(id);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ── Persistence ───────────────────────────────────────────

function saveState() {
    const data = {
        messages: messages.map(m => ({
            id: m.id,
            text: m.text,
            status: m.status === 'generating' ? 'error' : m.status,
            duration: m.duration,
            error: m.status === 'generating' ? 'Interrupted — page reloaded' : m.error,
            // Audio blobs can't be serialized; mark for no replay
        })),
        settings,
        clone: cloneReferenceId ? {
            referenceId: cloneReferenceId,
            filename: cloneFilename,
            duration: cloneDuration,
        } : null,
    };
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } catch {
        // Storage full — silently drop
    }
}

function loadState() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return;
        const data = JSON.parse(raw);
        if (data.settings) {
            settings = { ...DEFAULT_SETTINGS, ...data.settings };
        }
        if (Array.isArray(data.messages)) {
            messages = data.messages.slice(-50); // cap at 50
        }
        if (data.clone) {
            cloneReferenceId = data.clone.referenceId;
            cloneFilename = data.clone.filename;
            cloneDuration = data.clone.duration;
        }
    } catch {
        // Corrupt data — start fresh
    }
}

// ── Health Check ──────────────────────────────────────────

async function checkHealth() {
    const badge = $('serverStatus');
    const text = badge.querySelector('.status-text');
    try {
        const res = await fetch(HEALTH_ENDPOINT, { signal: AbortSignal.timeout(5000) });
        if (res.ok) {
            serverOnline = true;
            badge.className = 'status-badge connected';
            text.textContent = 'Server Online';
        } else {
            throw new Error(`HTTP ${res.status}`);
        }
    } catch {
        serverOnline = false;
        badge.className = 'status-badge error';
        text.textContent = 'Server Offline';
    }
}

// ── Settings UI ───────────────────────────────────────────

function applySettingsToUI() {
    $('voiceSelect').value = settings.voice;
    $('formatSelect').value = settings.response_format;
    $('exaggerationSlider').value = settings.exaggeration;
    $('exaggerationValue').textContent = settings.exaggeration;
    $('cfgWeightSlider').value = settings.cfg_weight;
    $('cfgWeightValue').textContent = settings.cfg_weight;
    $('temperatureSlider').value = settings.temperature;
    $('temperatureValue').textContent = settings.temperature;

    // Clone UI
    $('cloneToggle').checked = settings.cloneEnabled;
    $('cloneBody').classList.toggle('hidden', !settings.cloneEnabled);
    updateCloneFileUI();
}

function readSettingsFromUI() {
    settings.voice = $('voiceSelect').value.trim() || 'default';
    settings.response_format = $('formatSelect').value;
    settings.exaggeration = parseFloat($('exaggerationSlider').value);
    settings.cfg_weight = parseFloat($('cfgWeightSlider').value);
    settings.temperature = parseFloat($('temperatureSlider').value);
    settings.cloneEnabled = $('cloneToggle').checked;
    saveState();
}

function bindSettingsEvents() {
    // Sliders: update displayed value in real-time
    for (const [sliderId, valueId] of [
        ['exaggerationSlider', 'exaggerationValue'],
        ['cfgWeightSlider', 'cfgWeightValue'],
        ['temperatureSlider', 'temperatureValue'],
    ]) {
        $(sliderId).addEventListener('input', () => {
            $(valueId).textContent = $(sliderId).value;
        });
        $(sliderId).addEventListener('change', readSettingsFromUI);
    }

    $('voiceSelect').addEventListener('change', readSettingsFromUI);
    $('formatSelect').addEventListener('change', readSettingsFromUI);

    $('resetSettingsBtn').addEventListener('click', () => {
        settings = { ...DEFAULT_SETTINGS };
        removeCloneReference();
        applySettingsToUI();
        saveState();
    });
}

// ── Chat Rendering ────────────────────────────────────────

function renderMessages() {
    const area = $('chatArea');
    const placeholder = $('chatPlaceholder');

    if (messages.length === 0) {
        area.innerHTML = '';
        area.appendChild(createPlaceholder());
        return;
    }

    // Remove placeholder if present
    if (placeholder) placeholder.remove();

    // Build message elements
    area.innerHTML = '';
    for (const msg of messages) {
        area.appendChild(createMessageElement(msg));
    }

    // Auto-scroll
    requestAnimationFrame(() => {
        area.scrollTop = area.scrollHeight;
    });
}

function createPlaceholder() {
    const div = document.createElement('div');
    div.className = 'chat-placeholder';
    div.id = 'chatPlaceholder';
    div.innerHTML = `
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.3">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
            <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
            <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
        </svg>
        <p>Type a message to hear it spoken</p>
        <p class="hint">Ctrl+Enter to send</p>
    `;
    return div;
}

function createMessageElement(msg) {
    const el = document.createElement('div');
    el.className = 'chat-message';
    el.dataset.id = msg.id;

    let controlsHtml = '';

    if (msg.status === 'generating') {
        controlsHtml = `
            <div class="msg-controls">
                <span class="msg-status generating">
                    <span class="msg-spinner"></span>
                    Generating...
                </span>
            </div>
        `;
    } else if (msg.status === 'success') {
        const durationStr = msg.duration ? ` (${msg.duration.toFixed(1)}s)` : '';
        controlsHtml = `
            <div class="msg-controls">
                <button class="btn-play" data-id="${msg.id}" aria-label="Play audio">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" stroke="none">
                        <polygon points="5 3 19 12 5 21 5 3"/>
                    </svg>
                    Play
                </button>
                <span class="msg-duration">${durationStr}</span>
                <span class="msg-status success">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
                        <polyline points="20 6 9 17 4 12"/>
                    </svg>
                </span>
            </div>
        `;
    } else if (msg.status === 'error') {
        controlsHtml = `
            <div class="msg-controls">
                <span class="msg-status error">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </span>
                <span class="msg-error-text">${escapeHtml(msg.error || 'Unknown error')}</span>
                <button class="btn-retry" data-id="${msg.id}" aria-label="Retry generation">Retry</button>
            </div>
        `;
    }

    el.innerHTML = `
        <div class="msg-text">${escapeHtml(msg.text)}</div>
        ${controlsHtml}
    `;

    return el;
}

function updateSingleMessage(msg) {
    const area = $('chatArea');
    const existing = area.querySelector(`.chat-message[data-id="${msg.id}"]`);
    const newEl = createMessageElement(msg);

    if (existing) {
        existing.replaceWith(newEl);
    } else {
        // Remove placeholder if present
        const ph = $('chatPlaceholder');
        if (ph) ph.remove();
        area.appendChild(newEl);
    }

    requestAnimationFrame(() => {
        area.scrollTop = area.scrollHeight;
    });
}

// ── TTS Generation ────────────────────────────────────────

async function generateSpeech(msg) {
    readSettingsFromUI();

    msg.status = 'generating';
    msg.error = null;
    msg.audioUrl = null;
    updateSingleMessage(msg);
    setGeneratingUI(true);

    activeController = new AbortController();
    activeMessageId = msg.id;

    // Timeout wrapper
    const timeoutId = setTimeout(() => {
        if (activeController) activeController.abort();
    }, REQUEST_TIMEOUT_MS);

    try {
        const payload = {
            input: msg.text,
            model: 'chatterbox-turbo',
            voice: settings.voice,
            response_format: settings.response_format,
            exaggeration: settings.exaggeration,
            cfg_weight: settings.cfg_weight,
            temperature: settings.temperature,
        };

        // Voice clone: attach reference_id if enabled and available
        if (settings.cloneEnabled && cloneReferenceId) {
            payload.reference_id = cloneReferenceId;
        }

        const res = await fetch(TTS_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: activeController.signal,
        });

        if (!res.ok) {
            let detail = `Server returned ${res.status}`;
            try {
                const errBody = await res.json();
                if (errBody.detail) detail = errBody.detail;
                else if (errBody.message) detail = errBody.message;
                else if (errBody.error) detail = typeof errBody.error === 'string' ? errBody.error : errBody.error.message || detail;
            } catch {
                // Could not parse error body
            }
            throw new Error(detail);
        }

        const blob = await res.blob();
        if (blob.size === 0) throw new Error('Empty audio response');

        const audioUrl = URL.createObjectURL(blob);
        msg.audioUrl = audioUrl;
        msg.status = 'success';

        // Get audio duration
        msg.duration = await getAudioDuration(audioUrl);

        updateSingleMessage(msg);
        saveState();

        // Auto-play
        playAudio(msg);

    } catch (err) {
        if (err.name === 'AbortError') {
            msg.status = 'error';
            msg.error = 'Cancelled';
        } else if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
            msg.status = 'error';
            msg.error = 'Cannot reach TTS server — is it running on localhost:8001?';
        } else {
            msg.status = 'error';
            msg.error = err.message;
        }
        updateSingleMessage(msg);
        saveState();
    } finally {
        clearTimeout(timeoutId);
        activeController = null;
        activeMessageId = null;
        setGeneratingUI(false);
    }
}

function cancelGeneration() {
    if (activeController) {
        activeController.abort();
    }
}

function getAudioDuration(url) {
    return new Promise((resolve) => {
        const audio = new Audio();
        audio.addEventListener('loadedmetadata', () => {
            resolve(audio.duration);
        });
        audio.addEventListener('error', () => resolve(0));
        audio.src = url;
    });
}

// ── Audio Playback ────────────────────────────────────────

let currentAudio = null;
let currentPlayBtn = null;

function playAudio(msg) {
    if (!msg.audioUrl) return;

    // Stop any playing audio
    stopCurrentAudio();

    const audio = new Audio(msg.audioUrl);
    currentAudio = audio;

    // Find play button for this message
    const btn = document.querySelector(`.btn-play[data-id="${msg.id}"]`);
    if (btn) {
        btn.classList.add('playing');
        btn.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" stroke="none">
                <rect x="6" y="4" width="4" height="16"/>
                <rect x="14" y="4" width="4" height="16"/>
            </svg>
            Playing
        `;
        currentPlayBtn = btn;
    }

    audio.addEventListener('ended', () => {
        resetPlayButton(btn);
        currentAudio = null;
        currentPlayBtn = null;
    });

    audio.addEventListener('error', () => {
        resetPlayButton(btn);
        currentAudio = null;
        currentPlayBtn = null;
    });

    audio.play().catch(() => {
        resetPlayButton(btn);
        currentAudio = null;
        currentPlayBtn = null;
    });
}

function stopCurrentAudio() {
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
        resetPlayButton(currentPlayBtn);
        currentAudio = null;
        currentPlayBtn = null;
    }
}

function resetPlayButton(btn) {
    if (!btn) return;
    btn.classList.remove('playing');
    btn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" stroke="none">
            <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        Play
    `;
}

// ── UI State ──────────────────────────────────────────────

function setGeneratingUI(generating) {
    $('sendBtn').classList.toggle('hidden', generating);
    $('cancelBtn').classList.toggle('hidden', !generating);
    $('messageInput').disabled = generating;

    if (!generating) {
        $('messageInput').focus();
    }
}

// ── Event Handlers ────────────────────────────────────────

function handleSend() {
    const input = $('messageInput');
    const text = input.value.trim();
    if (!text) return;
    if (activeController) return; // Already generating

    const msg = {
        id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
        text,
        status: 'idle',
        audioUrl: null,
        duration: null,
        error: null,
    };

    messages.push(msg);
    input.value = '';
    autoResizeTextarea(input);
    saveState();

    generateSpeech(msg);
}

function handleRetry(msgId) {
    const msg = messages.find(m => m.id === msgId);
    if (!msg) return;
    if (activeController) return;

    // Revoke old audio if any
    if (msg.audioUrl) {
        URL.revokeObjectURL(msg.audioUrl);
        msg.audioUrl = null;
    }

    generateSpeech(msg);
}

function handlePlay(msgId) {
    const msg = messages.find(m => m.id === msgId);
    if (!msg) return;

    // Toggle: if this message is currently playing, stop it
    if (currentAudio && currentPlayBtn && currentPlayBtn.dataset.id === msgId) {
        stopCurrentAudio();
        return;
    }

    if (msg.audioUrl) {
        playAudio(msg);
    }
}

function clearChat() {
    // Revoke all audio URLs
    for (const msg of messages) {
        if (msg.audioUrl) URL.revokeObjectURL(msg.audioUrl);
    }
    messages = [];
    stopCurrentAudio();
    renderMessages();
    saveState();
}

function autoResizeTextarea(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

// ── Gesture Tag Insertion ─────────────────────────────────

function insertGestureTag(tag) {
    const textarea = $('messageInput');
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const text = textarea.value;

    // Insert tag at cursor, with smart spacing
    let insert = tag;
    if (start > 0 && text[start - 1] !== ' ' && text[start - 1] !== '\n') {
        insert = ' ' + insert;
    }
    if (end < text.length && text[end] !== ' ' && text[end] !== '\n') {
        insert = insert + ' ';
    }

    textarea.value = text.slice(0, start) + insert + text.slice(end);
    const newPos = start + insert.length;
    textarea.setSelectionRange(newPos, newPos);
    textarea.focus();
    autoResizeTextarea(textarea);
}

// ── Voice Clone ──────────────────────────────────────────

function updateCloneFileUI() {
    const hasRef = !!cloneReferenceId;
    $('cloneDropzone').classList.toggle('hidden', hasRef);
    $('cloneFileInfo').classList.toggle('hidden', !hasRef);

    if (hasRef) {
        $('cloneFilename').textContent = cloneFilename || 'reference';
        $('cloneDuration').textContent = cloneDuration ? `${cloneDuration}s` : '';
    }
}

function showCloneError(msg) {
    const el = $('cloneError');
    el.textContent = msg;
    el.classList.remove('hidden');
    setTimeout(() => el.classList.add('hidden'), 6000);
}

async function uploadCloneReference(file) {
    // Client-side validation
    const allowedExts = ['.wav', '.mp3', '.m4a', '.flac'];
    const ext = file.name.toLowerCase().match(/\.[^.]+$/)?.[0];
    if (ext && !allowedExts.includes(ext)) {
        showCloneError(`Unsupported file type. Use WAV, MP3, M4A, or FLAC.`);
        return;
    }
    if (file.size > MAX_REFERENCE_SIZE) {
        showCloneError(`File too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Max 10MB.`);
        return;
    }

    // Upload to backend
    const formData = new FormData();
    formData.append('file', file);

    $('cloneDropzone').classList.add('hidden');
    $('cloneError').classList.add('hidden');

    try {
        const res = await fetch(REFERENCE_ENDPOINT, {
            method: 'POST',
            body: formData,
            signal: AbortSignal.timeout(30000),
        });

        if (!res.ok) {
            let detail = `Upload failed (${res.status})`;
            try {
                const body = await res.json();
                if (body.detail) detail = body.detail;
            } catch {}
            throw new Error(detail);
        }

        const data = await res.json();
        cloneReferenceId = data.id;
        cloneFilename = data.filename || file.name;
        cloneDuration = data.duration_seconds;

        updateCloneFileUI();
        saveState();
    } catch (err) {
        showCloneError(err.message);
        $('cloneDropzone').classList.remove('hidden');
    }
}

async function removeCloneReference() {
    if (cloneReferenceId) {
        // Fire-and-forget cleanup on server
        fetch(`${REFERENCE_ENDPOINT}/${cloneReferenceId}`, { method: 'DELETE' }).catch(() => {});
    }
    cloneReferenceId = null;
    cloneFilename = null;
    cloneDuration = null;
    updateCloneFileUI();
    saveState();
}

async function validateCloneReference() {
    // After page reload, check if the server still has our reference.
    // If the server restarted, the in-memory reference is gone.
    // We do a lightweight check by attempting a no-op speech and catching 404,
    // or simpler: just mark it as "needs re-upload" and let the user know.
    // For simplicity, we keep the UI state but the error will surface
    // naturally when generation is attempted with a stale reference_id.
    // The user can remove and re-upload.
}

function bindCloneEvents() {
    // Toggle
    $('cloneToggle').addEventListener('change', () => {
        const enabled = $('cloneToggle').checked;
        settings.cloneEnabled = enabled;
        $('cloneBody').classList.toggle('hidden', !enabled);
        saveState();
    });

    // File input
    $('cloneFileInput').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) uploadCloneReference(file);
        e.target.value = ''; // Reset so same file can be re-selected
    });

    // Drag-and-drop
    const dropzone = $('cloneDropzone');
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-over');
    });
    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('drag-over');
    });
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file) uploadCloneReference(file);
    });

    // Remove
    $('cloneRemoveBtn').addEventListener('click', removeCloneReference);
}

// ── Init ──────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    loadState();
    applySettingsToUI();
    bindSettingsEvents();
    bindCloneEvents();
    renderMessages();

    // Validate persisted clone reference still exists on server
    if (cloneReferenceId) {
        validateCloneReference();
    }

    // Health check
    checkHealth();
    setInterval(checkHealth, HEALTH_INTERVAL_MS);

    // Send button
    $('sendBtn').addEventListener('click', handleSend);

    // Cancel button
    $('cancelBtn').addEventListener('click', cancelGeneration);

    // Clear chat
    $('clearChatBtn').addEventListener('click', clearChat);

    // Textarea: Ctrl/Cmd+Enter to send, auto-resize
    const textarea = $('messageInput');
    textarea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            handleSend();
        }
    });
    textarea.addEventListener('input', () => autoResizeTextarea(textarea));

    // Gesture tag chips
    $('gestureBar').addEventListener('click', (e) => {
        const chip = e.target.closest('.gesture-chip');
        if (chip) {
            insertGestureTag(chip.dataset.tag);
        }
    });

    // Delegate clicks on play/retry buttons
    $('chatArea').addEventListener('click', (e) => {
        const playBtn = e.target.closest('.btn-play');
        if (playBtn) {
            handlePlay(playBtn.dataset.id);
            return;
        }

        const retryBtn = e.target.closest('.btn-retry');
        if (retryBtn) {
            handleRetry(retryBtn.dataset.id);
            return;
        }
    });
});
