function sendMessage() {
    let msg = document.getElementById("user-input").value;

    if (msg.trim() === "") return;

    let chatBox = document.getElementById("chat-box");
    chatBox.innerHTML += `<p class="user-message">${msg}</p>`;
    document.getElementById("user-input").value = "";

    fetch("/get_response", {
        method: "POST",
        body: JSON.stringify({ message: msg }),
        headers: { "Content-Type": "application/json" }
    })
    .then(res => res.json())
    .then(data => {
        chatBox.innerHTML += `<p class="bot-message">${data.reply}</p>`;
        chatBox.scrollTop = chatBox.scrollHeight;
    });
}
