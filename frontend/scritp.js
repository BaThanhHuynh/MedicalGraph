const chatBody = document.getElementById('chat-body');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

function appendMessage(role, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `msg ${role}`;
    msgDiv.innerHTML = `<div class="bubble">${text}</div>`;
    chatBody.appendChild(msgDiv);
    chatBody.scrollTop = chatBody.scrollHeight;
}

async function handleChat() {
    const message = userInput.value.trim();
    if (!message) return;

    // Gửi tin nhắn của người dùng
    appendMessage('user', message);
    userInput.value = '';

    // Hiệu ứng "đang trả lời"
    const typingDiv = document.createElement('div');
    typingDiv.className = 'msg bot';
    typingDiv.innerHTML = `<div class="bubble" id="typing">...</div>`;
    chatBody.appendChild(typingDiv);

    // Giả lập thời gian xử lý (Sau này bạn kết nối với API MedicalGraph tại đây)
    setTimeout(() => {
        chatBody.removeChild(typingDiv);
        appendMessage('bot', "Hệ thống đã nhận thông tin. Bạn cần tra cứu chi tiết về mã bệnh hay triệu chứng này?");
    }, 1500);
}

sendBtn.addEventListener('click', handleChat);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleChat();
});