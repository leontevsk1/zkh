const chat = document.getElementById('chat');
const textInput = document.getElementById('text-input');
const sendBtn = document.getElementById('send-btn');
const voiceBtn = document.getElementById('voice-btn');

let userMessage = null;
let userAddress = null;
let isRecording = false;
let mediaRecorder;
let audioChunks = [];

// --- Настройка API ---
const CONTROLLER_URL = "http://localhost:8000/ingest_audio";

// Функция добавления сообщения в чат
function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(sender);
    messageDiv.textContent = text;
    chat.appendChild(messageDiv);
    chat.scrollTop = chat.scrollHeight;
}

// Функция добавления HTML-элемента (кнопки, инпута и т.д.)
function addElementToChat(element) {
    const wrapper = document.createElement('div');
    wrapper.classList.add('message', 'server');
    wrapper.appendChild(element);
    chat.appendChild(wrapper);
    chat.scrollTop = chat.scrollHeight;
}

// Отправка текстового сообщения
sendBtn.addEventListener('click', () => {
    const text = textInput.value.trim();
    if (text) {
        userMessage = text;
        addMessage(text, 'user');
        requestAddress(text);
        textInput.value = '';
    }
});

textInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendBtn.click();
    }
});

// Голосовой ввод
voiceBtn.addEventListener('click', async () => {
    if (isRecording) {
        // Остановить запись
        mediaRecorder.stop();
        isRecording = false;
        voiceBtn.classList.remove('recording');
        voiceBtn.textContent = '🎤';
    } else {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);

            audioChunks = [];
            mediaRecorder.ondataavailable = (e) => {
                audioChunks.push(e.data);
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });

                // Конвертация в WAV (16 кГц, моно)
                const wavBlob = await convertToWav(audioBlob);

                // Показываем аудио в чате
                const voiceMsg = document.createElement('div');
                voiceMsg.classList.add('message', 'user');
                const audioEl = document.createElement('audio');
                audioEl.controls = true;
                audioEl.src = URL.createObjectURL(wavBlob);
                voiceMsg.appendChild(audioEl);
                chat.appendChild(voiceMsg);
                chat.scrollTop = chat.scrollHeight;

                // Кнопка отправки на сервер
                const sendToServerBtn = document.createElement('button');
                sendToServerBtn.textContent = '📤 Отправить на обработку';
                sendToServerBtn.style.marginTop = '5px';

                sendToServerBtn.onclick = () => {
                    sendToServerBtn.disabled = true;
                    sendToServerBtn.textContent = 'Отправляю...';
                    sendAudioToController(wavBlob);
                };

                const wrapper = document.createElement('div');
                wrapper.classList.add('message', 'server');
                wrapper.appendChild(sendToServerBtn);
                chat.appendChild(wrapper);
            };

            mediaRecorder.start();
            isRecording = true;
            voiceBtn.classList.add('recording');
            voiceBtn.textContent = '●';
        } catch (err) {
            alert('Ошибка доступа к микрофону: ' + err.message);
            console.error(err);
        }
    }
});

// Конвертация в WAV (16kHz, mono)
async function convertToWav(audioBlob) {
    const arrayBuffer = await audioBlob.arrayBuffer();
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await audioContext.decodeAudioData(arrayBuffer);

    // Ресемплинг до 16 кГц
    const offlineContext = new OfflineAudioContext(1, decoded.length * (16000 / decoded.sampleRate), 16000);
    const source = offlineContext.createBufferSource();
    source.buffer = decoded;
    source.connect(offlineContext.destination);
    source.start();

    const renderedBuffer = await offlineContext.startRendering();

    // Кодируем в WAV
    const wavBuffer = encodeWAV(renderedBuffer.getChannelData(0));
    return new Blob([wavBuffer], { type: 'audio/wav' });
}

// Кодирование в WAV (PCM, 16-bit)
function encodeWAV(samples) {
    const sampleRate = 16000;
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeString = (offset, str) => {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset + i, str.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        const val = s < 0 ? s * 0x8000 : s * 0x7FFF;
        view.setInt16(offset, val, true);
        offset += 2;
    }

    return buffer;
}

// Отправка аудио на управляющий сервис
async function sendAudioToController(wavBlob) {
    const formData = new FormData();
    formData.append('file', wavBlob, 'recording.wav');

    try {
        const response = await fetch(CONTROLLER_URL, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            const requestId = result.request_id || "неизвестен";

            addMessage(`Ваше голосовое сообщение отправлено на обработку.\nID: ${requestId}`, 'server');

            // Теперь запрашиваем адрес
            requestAddress("Голосовое сообщение");
        } else {
            const errorText = await response.text();
            addMessage(`Ошибка при отправке: ${response.status}\n${errorText}`, 'server');
        }
    } catch (err) {
        addMessage(`Не удалось подключиться к сервису: ${err.message}`, 'server');
    }
}

// Запрос адреса
function requestAddress(text) {
    addMessage("Пожалуйста, уточните адрес дома:", 'server');
    const addressInput = document.createElement('input');
    addressInput.type = 'text';
    addressInput.placeholder = 'Например: ул. Ленина, д. 10, кв. 5';
    addressInput.classList.add('address-input');

    const submitBtn = document.createElement('button');
    submitBtn.textContent = 'Отправить';
    submitBtn.style.marginLeft = '5px';

    submitBtn.onclick = () => {
        const addr = addressInput.value.trim();
        if (addr) {
            confirmAddress(text, addr);
            addressInput.disabled = true;
            submitBtn.disabled = true;
        } else {
            addMessage("Адрес не может быть пустым.", 'server');
        }
    };

    addressInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            submitBtn.click();
        }
    });

    const wrapper = document.createElement('div');
    wrapper.classList.add('message', 'server');
    wrapper.appendChild(addressInput);
    wrapper.appendChild(submitBtn);
    chat.appendChild(wrapper);
    chat.scrollTop = chat.scrollHeight;

    addressInput.focus();
}

// Подтверждение адреса
function confirmAddress(text, address) {
    const confirmationText = `Вы отправили заявку:\n"${text}"\nАдрес: ${address}\n\nВсё верно?`;

    addMessage(confirmationText, 'server');

    const buttonContainer = document.createElement('div');
    buttonContainer.style.display = 'flex';
    buttonContainer.style.gap = '10px';
    buttonContainer.style.marginTop = '5px';

    const correctBtn = document.createElement('button');
    correctBtn.textContent = '✅ Верно';
    correctBtn.style.backgroundColor = '#28a745';
    correctBtn.style.color = 'white';
    correctBtn.style.border = 'none';
    correctBtn.style.padding = '8px 12px';
    correctBtn.style.borderRadius = '4px';
    correctBtn.style.cursor = 'pointer';

    const editBtn = document.createElement('button');
    editBtn.textContent = '✏️ Исправить адрес';
    editBtn.style.backgroundColor = '#ffc107';
    editBtn.style.border = 'none';
    editBtn.style.padding = '8px 12px';
    editBtn.style.borderRadius = '4px';
    editBtn.style.cursor = 'pointer';

    correctBtn.onclick = () => {
        buttonContainer.remove();
        const lastMsg = chat.lastElementChild;
        if (lastMsg && lastMsg.textContent.includes('Всё верно?')) {
            lastMsg.remove();
        }
        finalizeRequest(text, address);
    };

    editBtn.onclick = () => {
        buttonContainer.remove();
        requestAddress(text);
    };

    buttonContainer.appendChild(correctBtn);
    buttonContainer.appendChild(editBtn);
    addElementToChat(buttonContainer);
}

// Финальное подтверждение
function finalizeRequest(text, address) {
    const finalMessage = `Заявка зарегистрирована!\n\nТекст: "${text}"\nАдрес: ${address}\n\nСпасибо за обращение.`;
    addMessage(finalMessage, 'server');
}
