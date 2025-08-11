const chat = document.getElementById('chat');
const textInput = document.getElementById('text-input');
const sendBtn = document.getElementById('send-btn');
const voiceBtn = document.getElementById('voice-btn');

let userMessage = null;
let userAddress = null;
let awaitingAddress = false;

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
        handleUserMessage(text);
        textInput.value = '';
    }
});

textInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendBtn.click();
    }
});

// Обработка голосового ввода
async function transcribeAudio(audioBlob) {
    try {
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const audioData = await audioContext.decodeAudioData(arrayBuffer);

        const sampleRate = 16000;
        const offlineContext = new OfflineAudioContext(1, audioData.duration * sampleRate, sampleRate);
        const source = offlineContext.createBufferSource();
        source.buffer = offlineContext.createBuffer(1, audioData.length, sampleRate);
        source.buffer.copyFromChannel(audioData.getChannelData(0), 0);
        source.connect(offlineContext.destination);
        source.start();

        const renderedBuffer = await offlineContext.startRendering();

        const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.lang = 'ru-RU';
        recognition.interimResults = false;

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript.trim();
            if (transcript) {
                addMessage(transcript, 'user');
                handleUserMessage(transcript);
            }
        };

        recognition.onerror = () => {
            addMessage("Не удалось распознать речь. Попробуйте ещё раз.", 'server');
        };

        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audio.onloadedmetadata = () => {
            recognition.start();
            audio.play();
        };

    } catch (err) {
        addMessage("Ошибка обработки аудио: " + err.message, 'server');
    }
}

// Обработка сообщения пользователя
function handleUserMessage(text) {
    userMessage = text;
    addMessage(text, 'user');

    // Спрашиваем адрес
    addMessage("Пожалуйста, уточните адрес дома:", 'server');
    const addressInput = document.createElement('input');
    addressInput.type = 'text';
    addressInput.placeholder = 'Например: ул. Ленина, д. 10, кв. 5';
    addressInput.classList.add('address-input');

    const submitBtn = document.createElement('button');
    submitBtn.textContent = 'Отправить адрес';
    submitBtn.style.marginLeft = '5px';

    submitBtn.onclick = () => {
        const addr = addressInput.value.trim();
        if (addr) {
            // Убираем инпут и кнопку
            addressInput.disabled = true;
            submitBtn.disabled = true;
            submitBtn.textContent = "Отправлено";
            submitBtn.onclick = null;
            addressInput.style.opacity = "0.6";

            confirmAddress(addr);
        } else {
            addMessage("Адрес не может быть пустым.", 'server');
        }
    };

    addressInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            submitBtn.click();
        }
    });

    const inputWrapper = document.createElement('div');
    inputWrapper.classList.add('message', 'server');
    inputWrapper.appendChild(addressInput);
    inputWrapper.appendChild(submitBtn);

    chat.appendChild(inputWrapper);
    chat.scrollTop = chat.scrollHeight;

    addressInput.focus();
}

// Подтверждение адреса
function confirmAddress(address) {
    userAddress = address;
    const confirmationText = `Вы отправили заявку:\n"${userMessage}"\nАдрес: ${userAddress}\n\nВсё верно?`;

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

    // Кнопка "Верно" — удаляет ВСЁ и показывает финальное сообщение
    correctBtn.onclick = () => {
        // Удаляем контейнер с кнопками
        buttonContainer.remove();

        // Удаляем предыдущее сообщение с вопросом "Всё верно?"
        const lastMessage = chat.lastElementChild;
        if (lastMessage && lastMessage.textContent.includes('Всё верно?')) {
            lastMessage.remove();
        }

        // Отправляем финальное сообщение
        finalizeRequest();
    };

    editBtn.onclick = () => {
        buttonContainer.remove();
        askForAddressAgain();
    };

    buttonContainer.appendChild(correctBtn);
    buttonContainer.appendChild(editBtn);
    addElementToChat(buttonContainer);
}

// Повторный ввод адреса
function askForAddressAgain() {
    addMessage("Введите адрес заново:", 'server');
    const addressInput = document.createElement('input');
    addressInput.type = 'text';
    addressInput.placeholder = 'ул. Пушкина, д. 5';
    addressInput.classList.add('address-input');

    const submitBtn = document.createElement('button');
    submitBtn.textContent = 'Отправить';
    submitBtn.style.marginLeft = '5px';

    submitBtn.onclick = () => {
        const addr = addressInput.value.trim();
        if (addr) {
            addressInput.disabled = true;
            submitBtn.disabled = true;
            submitBtn.textContent = "Отправлено";

            // Удаляем сообщение "Введите адрес заново"
            const lastMsg = chat.lastElementChild;
            if (lastMsg && lastMsg.textContent.includes('адрес заново')) {
                lastMsg.remove();
            }

            // Удаляем поле ввода
            const wrapper = submitBtn.parentElement;
            if (wrapper) wrapper.remove();

            confirmAddress(addr);
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

// Финальная отправка заявки
function finalizeRequest() {
    const finalMessage = `✅ Заявка зарегистрирована!\n\nТекст: "${userMessage}"\nАдрес: ${userAddress}\n\nСпасибо за обращение.`;
    addMessage(finalMessage, 'server');

    // Можно отправить на сервер:
    // fetch('/api/submit', {
    //     method: 'POST',
    //     headers: { 'Content-Type': 'application/json' },
    //     body: JSON.stringify({ message: userMessage, address: userAddress })
    // });

    // Сброс (на случай новой заявки)
    userMessage = null;
    userAddress = null;
}

// Голосовой ввод
let isRecording = false;
let mediaRecorder;
let audioChunks = [];

voiceBtn.addEventListener('click', async () => {
    if (isRecording) {
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

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                const audioEl = document.createElement('audio');
                audioEl.controls = true;
                audioEl.src = audioUrl;

                const voiceMsg = document.createElement('div');
                voiceMsg.classList.add('message', 'user');
                voiceMsg.appendChild(audioEl);
                chat.appendChild(voiceMsg);
                chat.scrollTop = chat.scrollHeight;

                transcribeAudio(audioBlob);
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