/* Chat Page Logic */

const API_BASE_URL = '/api';
let ocrContext = '';

// DOM Elements
const chatDocPreview = document.getElementById('chatDocPreview');
const ocrDataDisplay = document.getElementById('ocrDataDisplay');
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendChatBtn = document.getElementById('sendChatBtn');
const toggleOcr = document.getElementById('toggleOcr');

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    const imageData = sessionStorage.getItem('dc_chat_image');
    if (!imageData) {
        alert('No document data found. Returning to dashboard.');
        window.location.href = 'index.html';
        return;
    }

    chatDocPreview.src = imageData;
    initChat(imageData);

    // Event Listeners
    if (sendChatBtn) sendChatBtn.addEventListener('click', sendMessage);
    if (toggleOcr) toggleOcr.addEventListener('click', toggleOcrView);
    
    // Auto-resize textarea
    if (chatInput) {
        chatInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });

        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }
});

async function initChat(imageData) {
    try {
        const response = await fetch(`${API_BASE_URL}/chat/ocr`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('dc_token')}`
            },
            body: JSON.stringify({ image_base64: imageData })
        });

        if (!response.ok) throw new Error('OCR Extraction failed');
        
        const data = await response.json();
        ocrContext = data.text || 'No text detected in this document.';
        ocrDataDisplay.textContent = ocrContext;
        
        setTimeout(() => {
            appendBotMessage("I've finished the OCR analysis. You can now ask me questions about the content of this document!");
        }, 500);
    } catch (error) {
        console.error('OCR failed:', error);
        ocrDataDisplay.textContent = 'Error extracting text.';
        appendBotMessage("I had trouble reading the text from this file, but I can still try to help based on general knowledge!");
    }
}

function toggleOcrView() {
    const ocrSection = document.querySelector('.ocr-data-section');
    const icon = toggleOcr.querySelector('i');
    if (ocrSection.style.maxHeight === '40px') {
        ocrSection.style.maxHeight = '25%';
        icon.classList.replace('fa-chevron-down', 'fa-chevron-up');
    } else {
        ocrSection.style.maxHeight = '40px';
        icon.classList.replace('fa-chevron-up', 'fa-chevron-down');
    }
}

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    appendUserMessage(text);
    chatInput.value = '';
    chatInput.style.height = 'auto';

    // Show typing indicator
    const typingId = 'typing-' + Date.now();
    const typingMsg = document.createElement('div');
    typingMsg.id = typingId;
    typingMsg.className = 'message bot-message';
    typingMsg.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> &nbsp; Llama 3 is thinking...';
    chatMessages.appendChild(typingMsg);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch(`${API_BASE_URL}/chat/query`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('dc_token')}`
            },
            body: JSON.stringify({ 
                message: text,
                context: ocrContext,
                model: 'llama3'
            })
        });

        const typingElem = document.getElementById(typingId);
        if (typingElem) typingElem.remove();

        if (response.status === 503) {
            appendBotMessage("Error: Ollama service is not detected. Please ensure you have Llama 3 running locally.");
            return;
        }

        if (!response.ok) throw new Error('Chat failed');
        
        const data = await response.json();
        appendBotMessage(data.response);
    } catch (error) {
        const typingElem = document.getElementById(typingId);
        if (typingElem) typingElem.remove();
        appendBotMessage("Error: " + error.message);
    }
}

function appendUserMessage(text) {
    const msg = document.createElement('div');
    msg.className = 'message user-message';
    msg.textContent = text;
    chatMessages.appendChild(msg);
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
}

function appendBotMessage(text) {
    const msg = document.createElement('div');
    msg.className = 'message bot-message';
    msg.textContent = text;
    chatMessages.appendChild(msg);
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
}
