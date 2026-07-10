let selectedDevice = null;
let activeStreamBubble = null;

async function loadAgentDevice() {
  const serialInput = document.getElementById("serial");
  const status = document.getElementById("agentStatus");
  const deviceStatus = document.getElementById("deviceStatus");
  const deviceMeta = document.getElementById("deviceMeta");

  const serial = serialInput.value.trim();

  if (!serial) {
    status.textContent = "Enter a serial number first.";
    status.className = "status agent-side-status error";
    return;
  }

  status.textContent = "Loading device...";
  status.className = "status agent-side-status loading";

  try {
    const response = await fetch(
      `${API_BASE_URL}/device/${encodeURIComponent(serial)}`
    );

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Device not found");
    }

    selectedDevice = data;

    deviceStatus.textContent =
      `${data.item_name || "Device"} • ${data.serial_number || serial}`;

    if (deviceMeta) {
      deviceMeta.textContent =
        `${data.item_category || "Unknown category"} • ` +
        `${data.status || "Unknown status"}`;
    }

    status.textContent = "Device loaded successfully.";
    status.className = "status agent-side-status success";
  } catch (error) {
    selectedDevice = null;

    deviceStatus.textContent = "No device selected";

    if (deviceMeta) {
      deviceMeta.textContent = "General advice mode";
    }

    status.textContent =
      error.message || "Could not load device.";

    status.className = "status agent-side-status error";
  }
}

function usePrompt(text) {
  const questionInput = document.getElementById("question");

  questionInput.value = text;
  questionInput.focus();

  autoResizeComposer();
}

function handleComposerKey(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    askAgent();
  }

  requestAnimationFrame(autoResizeComposer);
}

function autoResizeComposer() {
  const questionInput = document.getElementById("question");

  if (!questionInput) {
    return;
  }

  questionInput.style.height = "auto";
  questionInput.style.height =
    `${Math.min(questionInput.scrollHeight, 110)}px`;
}

function clearChat() {
  const chat = document.getElementById("chat");
  const status = document.getElementById("agentStatus");

  chat.innerHTML = `
    <div class="chat-message assistant">
      <div class="message-avatar">AI</div>

      <div class="message-bubble welcome-bubble">
        <strong>Welcome to the Smart E-Waste Advisor.</strong>

        <span>
          Ask about a device's warranty, value, repair decision,
          reusable parts, resale potential, or recyclable materials.
        </span>
      </div>
    </div>
  `;

  status.textContent = "";
  status.className = "status agent-side-status";
}

async function askAgent() {
  const questionInput = document.getElementById("question");
  const status = document.getElementById("agentStatus");
  const serialInput = document.getElementById("serial");

  const question = questionInput.value.trim();

  if (!question) {
    return;
  }

  appendMessage("user", question);

  questionInput.value = "";
  autoResizeComposer();

  const assistantBubble = appendMessage("assistant", "");

  activeStreamBubble = assistantBubble;

  assistantBubble.classList.add("typing");
  assistantBubble.textContent = "Thinking...";

  status.textContent = "Generating response...";
  status.className = "status agent-side-status loading";

  try {
    const response = await fetch(
      `${API_BASE_URL}/advisor/stream`,
      {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
          "Accept": "text/event-stream"
        },

        body: JSON.stringify({
          question: question,

          serial_number:
            serialInput.value.trim() || null,

          language:
            getLang() === "ar" ? "ar" : "en",

          device_context:
            selectedDevice || null
        })
      }
    );

    if (!response.ok) {
      let errorMessage = `Advisor error ${response.status}`;

      try {
        const errorData = await response.json();

        if (errorData.error) {
          errorMessage = errorData.error;
        }
      } catch (error) {
        // The response was not JSON.
      }

      throw new Error(errorMessage);
    }

    if (!response.body) {
      throw new Error("Streaming is not supported by this browser.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let buffer = "";
    let completeText = "";

    assistantBubble.textContent = "";
    assistantBubble.classList.remove("typing");

    while (true) {
      const result = await reader.read();

      if (result.done) {
        break;
      }

      buffer += decoder.decode(
        result.value,
        {
          stream: true
        }
      );

      /*
       * Each Server-Sent Event ends with two newline characters.
       * This was the broken part in the previous agent.js file.
       */
      const events = buffer.split("\n\n");

      buffer = events.pop() || "";

      for (const event of events) {
        let eventType = "";
        let eventData = "";

        /*
         * Each line inside an SSE event is separated by one newline.
         */
        for (const line of event.split("\n")) {
          if (line.startsWith("event:")) {
            eventType = line.slice(6).trim();
          }

          if (line.startsWith("data:")) {
            eventData += line.slice(5).trim();
          }
        }

        if (eventType === "delta" && eventData) {
          try {
            const parsedData = JSON.parse(eventData);
            const newText = parsedData.text || "";

            completeText += newText;
            assistantBubble.textContent = completeText;

            scrollChat();
          } catch (error) {
            console.warn(
              "Could not parse advisor stream event:",
              eventData
            );
          }
        }

        if (eventType === "error" && eventData) {
          try {
            const parsedError = JSON.parse(eventData);

            throw new Error(
              parsedError.error ||
              parsedError.message ||
              "The advisor returned an error."
            );
          } catch (error) {
            if (error instanceof SyntaxError) {
              throw new Error(eventData);
            }

            throw error;
          }
        }
      }
    }

    /*
     * Process any final text remaining in the buffer.
     */
    if (buffer.trim()) {
      let eventType = "";
      let eventData = "";

      for (const line of buffer.split("\n")) {
        if (line.startsWith("event:")) {
          eventType = line.slice(6).trim();
        }

        if (line.startsWith("data:")) {
          eventData += line.slice(5).trim();
        }
      }

      if (eventType === "delta" && eventData) {
        try {
          const parsedData = JSON.parse(eventData);

          completeText += parsedData.text || "";
          assistantBubble.textContent = completeText;
        } catch (error) {
          console.warn(
            "Could not parse final advisor event:",
            eventData
          );
        }
      }
    }

    if (!completeText.trim()) {
      assistantBubble.textContent =
        "No response was returned by the advisor.";
    }

    status.textContent = "";
    status.className = "status agent-side-status";
  } catch (error) {
    console.error("AI advisor error:", error);

    assistantBubble.classList.remove("typing");

    assistantBubble.textContent =
      "Could not connect to the advisor right now.";

    status.textContent =
      error.message || "Advisor connection failed.";

    status.className =
      "status agent-side-status error";
  } finally {
    activeStreamBubble = null;
    scrollChat();
  }
}

function appendMessage(role, text) {
  const chat = document.getElementById("chat");

  const wrapper = document.createElement("div");
  wrapper.className = `chat-message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent =
    role === "assistant" ? "AI" : "You";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;

  if (role === "user") {
    wrapper.appendChild(bubble);
    wrapper.appendChild(avatar);
  } else {
    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
  }

  chat.appendChild(wrapper);

  scrollChat();

  return bubble;
}

function scrollChat() {
  const chat = document.getElementById("chat");

  if (!chat) {
    return;
  }

  chat.scrollTop = chat.scrollHeight;
}

document.addEventListener(
  "DOMContentLoaded",
  () => {
    const questionInput =
      document.getElementById("question");

    if (questionInput) {
      questionInput.addEventListener(
        "input",
        autoResizeComposer
      );
    }
  }
);
