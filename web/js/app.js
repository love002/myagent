/**
 * Real-Time Voice Assistant - WebSocket Client
 *
 * Connects to the FastAPI WebSocket backend and drives the UI.
 */

// ============================================================================
// State
// ============================================================================

const state = {
    status: 'stopped',    // stopped | listening | thinking | speaking
    ws: null,
    reconnectAttempts: 0,
    reconnectTimer: null,
    maxReconnectDelay: 30000,
};

// ============================================================================
// DOM References
// ============================================================================

const $ = (sel) => document.querySelector(sel);
const chatArea = $('#chatArea');
const welcome = $('#welcome');
const statusDot = $('#statusDot');
const statusText = $('#statusText');
const asrText = $('#asrText');
const asrWave = $('#asrWave');
const textInput = $('#textInput');
const sendBtn = $('#sendBtn');
const toggleBtn = $('#toggleBtn');
const emotionBar = $('#emotionBar');
const emotionEmoji = $('#emotionEmoji');

// ============================================================================
// Utility
// ============================================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/** Parse leading emoji(s) from text. */
function extractEmoji(text) {
    const match = text.match(
        /^(\p{Emoji_Presentation}|\p{Emoji}️)+/u,
    );
    return match ? match[0] : '';
}

// ============================================================================
// UI Helpers
// ============================================================================

function setStatus(newStatus) {
    state.status = newStatus;

    // Dot
    statusDot.className = 'status-dot';
    if (newStatus === 'listening') statusDot.classList.add('active');
    else if (newStatus === 'thinking') statusDot.classList.add('thinking');
    else if (newStatus === 'speaking') statusDot.classList.add('speaking');
    else if (newStatus === 'error') statusDot.classList.add('error');

    // Text
    const labels = {
        stopped: '已停止',
        listening: '聆听中',
        thinking: '思考中',
        speaking: '播放中',
        error: '错误',
    };
    statusText.textContent = labels[newStatus] || newStatus;

    // ASR bar
    if (newStatus === 'listening') {
        asrText.textContent = '正在聆听...';
        asrText.classList.add('active');
        asrWave.classList.add('active');
    } else if (newStatus === 'stopped') {
        asrText.textContent = '已停止';
        asrText.classList.remove('active');
        asrWave.classList.remove('active');
    } else if (newStatus === 'thinking' || newStatus === 'speaking') {
        asrText.textContent = '处理中...';
        asrText.classList.remove('active');
        asrWave.classList.remove('active');
    }

    // Toggle button
    if (newStatus === 'stopped') {
        toggleBtn.textContent = '▶ 启动';
        toggleBtn.className = 'btn btn-start';
        textInput.disabled = true;
        sendBtn.disabled = true;
    } else {
        toggleBtn.textContent = '⏹ 停止';
        toggleBtn.className = 'btn btn-start running';
        textInput.disabled = false;
        sendBtn.disabled = false;
    }

    // Hide emotion on stop
    if (newStatus === 'stopped') {
        emotionBar.style.display = 'none';
    }
}

function addMessage(role, text) {
    welcome.style.display = 'none';

    const template = document.getElementById(
        role === 'user' ? 'msgTemplateUser' : 'msgTemplateAI',
    );
    const clone = template.content.cloneNode(true);
    const bubble = clone.querySelector('.msg-bubble');

    // If text starts with emoji, show it prominently
    const emoji = extractEmoji(text);
    let displayText = text;

    if (role === 'user' && emoji) {
        // For user messages with emotion emoji, add a small emotion badge
        displayText = text.replace(emoji, '').trim();
        if (displayText) {
            bubble.textContent = displayText;
            // Could add a small badge for the emoji
        } else {
            bubble.textContent = text;
        }
    } else {
        bubble.textContent = text;
    }

    chatArea.appendChild(clone);
    chatArea.scrollTop = chatArea.scrollHeight;
}

function showTyping() {
    const existing = document.getElementById('typingIndicator');
    if (existing) return;

    welcome.style.display = 'none';

    const template = document.getElementById('typingTemplate');
    const clone = template.content.cloneNode(true);
    chatArea.appendChild(clone);
    chatArea.scrollTop = chatArea.scrollHeight;
}

function hideTyping() {
    const el = document.getElementById('typingIndicator');
    if (el) el.remove();
}

function setEmotion(emoji) {
    emotionEmoji.textContent = emoji;
    emotionBar.style.display = 'flex';
}

function showError(msg) {
    // Remove existing toast
    const old = document.querySelector('.error-toast');
    if (old) old.remove();

    const toast = document.createElement('div');
    toast.className = 'error-toast';
    toast.textContent = msg;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ============================================================================
// WebSocket
// ============================================================================

function getWebSocketUrl() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${location.host}/ws`;
}

function connectWebSocket() {
    if (state.ws &&
        (state.ws.readyState === WebSocket.OPEN ||
         state.ws.readyState === WebSocket.CONNECTING)
    ) return;

    const url = getWebSocketUrl();
    state.ws = new WebSocket(url);

    state.ws.onopen = () => {
        console.log('[WS] Connected');
        state.reconnectAttempts = 0;
        if (state.reconnectTimer) {
            clearTimeout(state.reconnectTimer);
            state.reconnectTimer = null;
        }
    };

    state.ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            handleMessage(msg);
        } catch (err) {
            console.error('[WS] Parse error:', err);
        }
    };

    state.ws.onclose = () => {
        console.log('[WS] Disconnected');
        scheduleReconnect();
    };

    state.ws.onerror = (err) => {
        console.error('[WS] Error:', err);
    };
}

function scheduleReconnect() {
    if (state.reconnectTimer) return;
    const delay = Math.min(
        1000 * Math.pow(2, state.reconnectAttempts),
        state.maxReconnectDelay,
    );
    console.log(`[WS] Reconnecting in ${delay}ms (attempt ${state.reconnectAttempts + 1})`);
    state.reconnectTimer = setTimeout(() => {
        state.reconnectAttempts++;
        state.reconnectTimer = null;
        connectWebSocket();
    }, delay);
}

function wsSend(data) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify(data));
    } else {
        showError('未连接到服务器');
    }
}

// ============================================================================
// Message Handler
// ============================================================================

function handleMessage(msg) {
    switch (msg.type) {

        case 'status':
            setStatus(msg.data);
            if (msg.data === 'thinking') showTyping();
            else hideTyping();
            break;

        case 'user_message':
            hideTyping();
            addMessage('user', msg.data);
            break;

        case 'ai_message':
            hideTyping();
            addMessage('ai', msg.data);
            break;

        case 'asr_done':
            // Clear ASR text; the user_message event carries the actual text
            asrText.textContent = '识别完成';
            break;

        case 'emotion':
            setEmotion(msg.data);
            break;

        case 'tts':
            if (msg.data === 'start') {
                asrText.textContent = '🔊 语音播放中...';
            } else if (msg.data === 'end') {
                asrText.textContent = '播放完成';
                setTimeout(() => {
                    if (state.status === 'listening') {
                        asrText.textContent = '正在聆听...';
                    }
                }, 1000);
            }
            break;

        case 'error':
            showError(msg.data);
            hideTyping();
            break;

        default:
            console.log('[WS] Unknown message:', msg);
    }
}

// ============================================================================
// User Actions
// ============================================================================

function sendChatMessage() {
    const text = textInput.value.trim();
    if (!text) return;

    textInput.value = '';
    wsSend({ action: 'chat', message: text });
}

function togglePipeline() {
    if (state.status === 'stopped') {
        wsSend({ action: 'start' });
    } else {
        wsSend({ action: 'stop' });
    }
}

// ============================================================================
// Event Bindings
// ============================================================================

// Send button
sendBtn.addEventListener('click', sendChatMessage);

// Enter key in input
textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
    }
});

// Start/Stop toggle
toggleBtn.addEventListener('click', togglePipeline);

// ============================================================================
// Init
// ============================================================================

connectWebSocket();
