document.addEventListener('DOMContentLoaded', () => {
    const chatInput = document.querySelector('input[type="text"]');
    const chatButton = document.querySelector('button');
    const historyContainer = document.getElementById('history');

    function sendChat() {
        if (chatInput.value.trim() !== '') {
            const userText = chatInput.value.trim();

            // Append user's message to chat history
            const userEntry = document.createElement('li');
            userEntry.classList.add('user-message');
            userEntry.textContent = `You: ${userText}`;
            historyContainer.appendChild(userEntry);

            // Simulate AI response and append to chat history
            const botResponseEntry = document.createElement('li');
            botResponseEntry.classList.add('ai-message');
            botResponseEntry.textContent = `AI: Thank you for your message!`;
            historyContainer.appendChild(botResponseEntry);

            // Clear input after sending
            chatInput.value = '';

            // Auto-scroll to the latest message
            userEntry.scrollIntoView({ behavior: 'smooth', block: 'end' });
        } else {
            alert('Please enter a message to chat.');
        }
    }

    chatButton.addEventListener('click', sendChat);
    chatInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendChat();
        }
    });
});
