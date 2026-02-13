/**
 * Voice Agent — Web Client
 *
 * Connects to LiveKit room, publishes microphone audio,
 * and plays back agent audio with transcript display.
 */

// Configuration
const TOKEN_SERVER_URL = 'http://localhost:8081';

// State
let room = null;
let localTrack = null;
let isMuted = false;
let isConnected = false;

// ── UI Helpers ──────────────────────────────────────────

function $(id) {
    return document.getElementById(id);
}

function setConnectionStatus(status, connected) {
    const badge = $('connectionStatus');
    const text = badge.querySelector('.status-text');
    text.textContent = status;
    badge.classList.toggle('connected', connected);
}

function showPanel(panelId) {
    $('connectionPanel').classList.toggle('hidden', panelId !== 'connectionPanel');
    $('callPanel').classList.toggle('hidden', panelId !== 'callPanel');
}

function setCallStatus(text) {
    $('callStatus').textContent = text;
}

function setVisualizerSpeaking(speaking) {
    $('visualizer').classList.toggle('speaking', speaking);
}

function addTranscript(speaker, text, isPartial = false) {
    const area = $('transcriptArea');

    // Remove placeholder
    const placeholder = area.querySelector('.transcript-placeholder');
    if (placeholder) placeholder.remove();

    // Update existing partial or create new entry
    const existingPartial = area.querySelector('.transcript-entry.partial');
    if (isPartial && existingPartial && existingPartial.classList.contains(speaker)) {
        existingPartial.querySelector('.text').textContent = text;
    } else {
        // Finalise any existing partial
        if (existingPartial) {
            existingPartial.classList.remove('partial');
        }

        const entry = document.createElement('div');
        entry.className = `transcript-entry ${speaker}${isPartial ? ' partial' : ''}`;
        entry.innerHTML = `
            <div class="speaker">${speaker === 'user' ? 'You' : 'Agent'}</div>
            <div class="text">${escapeHtml(text)}</div>
        `;
        area.appendChild(entry);
    }

    // Auto-scroll
    area.scrollTop = area.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function clearTranscripts() {
    $('transcriptArea').innerHTML = `
        <div class="transcript-placeholder">
            <p>Conversation will appear here...</p>
        </div>
    `;
}

// ── Connection ──────────────────────────────────────────

async function toggleConnection() {
    if (isConnected) {
        await disconnect();
    } else {
        await connect();
    }
}

async function connect() {
    const roomName = $('roomInput').value.trim();
    const identity = $('identityInput').value.trim() || `user-${Date.now().toString(36)}`;

    if (!roomName) {
        alert('Please enter a room name');
        return;
    }

    const btn = $('connectBtn');
    btn.disabled = true;
    btn.querySelector('span').textContent = 'Connecting...';

    try {
        // 1. Get token from token server
        const tokenRes = await fetch(`${TOKEN_SERVER_URL}/token`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                room: roomName,
                identity: identity,
                participant_type: 'user',
            }),
        });

        if (!tokenRes.ok) {
            const err = await tokenRes.json();
            throw new Error(err.detail || 'Failed to get token');
        }

        const tokenData = await tokenRes.json();

        // 2. Create LiveKit room
        room = new LivekitClient.Room({
            adaptiveStream: true,
            dynacast: true,
            audioCaptureDefaults: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
        });

        // 3. Set up event handlers
        setupRoomEvents(room);

        // 4. Connect to room
        await room.connect(tokenData.ws_url, tokenData.token);

        // 5. Publish microphone
        await room.localParticipant.setMicrophoneEnabled(true);

        isConnected = true;
        setConnectionStatus('Connected', true);
        showPanel('callPanel');
        setCallStatus('Waiting for agent...');
        clearTranscripts();

    } catch (err) {
        console.error('Connection failed:', err);
        alert(`Connection failed: ${err.message}`);
        setConnectionStatus('Disconnected', false);
    } finally {
        btn.disabled = false;
        btn.querySelector('span').textContent = 'Connect';
    }
}

async function disconnect() {
    if (room) {
        await room.disconnect();
        room = null;
    }
    isConnected = false;
    isMuted = false;
    setConnectionStatus('Disconnected', false);
    showPanel('connectionPanel');
    setVisualizerSpeaking(false);
    $('muteBtn').classList.remove('muted');
}

// ── Room Events ─────────────────────────────────────────

function setupRoomEvents(room) {
    room.on(LivekitClient.RoomEvent.ParticipantConnected, (participant) => {
        console.log('Participant connected:', participant.identity);
        setCallStatus('Agent joined — start speaking!');
    });

    room.on(LivekitClient.RoomEvent.ParticipantDisconnected, (participant) => {
        console.log('Participant disconnected:', participant.identity);
        setCallStatus('Agent disconnected');
        setVisualizerSpeaking(false);
    });

    room.on(LivekitClient.RoomEvent.TrackSubscribed, (track, publication, participant) => {
        console.log('Track subscribed:', track.kind, 'from', participant.identity);
        if (track.kind === LivekitClient.Track.Kind.Audio) {
            // Attach remote audio for playback
            const audioEl = track.attach();
            audioEl.id = `audio-${participant.identity}`;
            document.body.appendChild(audioEl);
            setCallStatus('Agent is connected');
        }
    });

    room.on(LivekitClient.RoomEvent.TrackUnsubscribed, (track, publication, participant) => {
        // Clean up audio elements
        const elements = track.detach();
        elements.forEach(el => el.remove());
    });

    room.on(LivekitClient.RoomEvent.ActiveSpeakersChanged, (speakers) => {
        const agentSpeaking = speakers.some(
            s => s.identity !== room.localParticipant.identity
        );
        const userSpeaking = speakers.some(
            s => s.identity === room.localParticipant.identity
        );

        setVisualizerSpeaking(userSpeaking || agentSpeaking);

        if (agentSpeaking) {
            setCallStatus('Agent is speaking...');
        } else if (userSpeaking) {
            setCallStatus('Listening...');
        } else {
            setCallStatus('Ready');
        }
    });

    room.on(LivekitClient.RoomEvent.DataReceived, (payload, participant, kind) => {
        try {
            const data = JSON.parse(new TextDecoder().decode(payload));
            if (data.type === 'transcript') {
                addTranscript(
                    data.speaker || 'agent',
                    data.text,
                    data.is_partial || false
                );
            }
        } catch (e) {
            // Non-JSON data, ignore
        }
    });

    room.on(LivekitClient.RoomEvent.Disconnected, (reason) => {
        console.log('Disconnected:', reason);
        isConnected = false;
        setConnectionStatus('Disconnected', false);
        showPanel('connectionPanel');
        setVisualizerSpeaking(false);
    });

    room.on(LivekitClient.RoomEvent.Reconnecting, () => {
        setConnectionStatus('Reconnecting...', false);
    });

    room.on(LivekitClient.RoomEvent.Reconnected, () => {
        setConnectionStatus('Connected', true);
    });
}

// ── Mute ────────────────────────────────────────────────

async function toggleMute() {
    if (!room) return;

    isMuted = !isMuted;
    await room.localParticipant.setMicrophoneEnabled(!isMuted);

    const btn = $('muteBtn');
    btn.classList.toggle('muted', isMuted);
    btn.querySelector('.mic-on').classList.toggle('hidden', isMuted);
    btn.querySelector('.mic-off').classList.toggle('hidden', !isMuted);
}

// ── Init ────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    // Generate a random identity
    const adjectives = ['Happy', 'Bright', 'Swift', 'Calm', 'Bold'];
    const nouns = ['Falcon', 'River', 'Peak', 'Star', 'Wave'];
    const adj = adjectives[Math.floor(Math.random() * adjectives.length)];
    const noun = nouns[Math.floor(Math.random() * nouns.length)];
    $('identityInput').value = `${adj}${noun}`;
});
