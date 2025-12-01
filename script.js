// =======================================================
// COMPLETE script.js - WITH DEFINITIVE STATE FIX
// =======================================================

const hero = document.querySelector(".hero");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const addFileBtn = document.getElementById("addFileBtn");

// File Upload variables
const fileInput = document.getElementById("fileInput");
const fileNotationContainer = document.getElementById("fileNotationContainer");
const fileNameDisplay = document.getElementById("fileNameDisplay");
const clearFileBtn = document.getElementById("clearFileBtn");
let stagedFile = null; // Stores the file object currently staged for upload

const leftArrow = document.getElementById("leftArrow");
const rightArrow = document.getElementById("rightArrow");
const centerContent = document.querySelector(".center-content");
const chatContainer = document.getElementById("chatContainer");

// Image Generation Panel variables
const generationPanel = document.getElementById("generationPanel");
const imageDisplayArea = document.getElementById("imageDisplayArea");
const generationInstructions = document.getElementById("generationInstructions");
const imageHistoryContainer = document.getElementById("imageHistoryContainer");

// Secure Detect Panel variables
const secureResultsBox = document.getElementById("secureResultsBox");
const secureInstructions = document.getElementById("secureInstructions");
const secureHistoryContainer = document.getElementById("secureHistoryContainer");

let currentMode = "home";
let chatStarted = false;

// --- FASTAPI CONFIGURATION ---
const API_BASE_URL = "http://127.0.0.1:8000";

/* ---------- helpers ---------- */

function setPlaceholder(text) {
    if (queryInput) queryInput.placeholder = text;
}

function hideArrows() {
    leftArrow?.classList.add("hidden");
    rightArrow?.classList.add("hidden");
}

function showArrows() {
    leftArrow?.classList.remove("hidden");
    rightArrow?.classList.remove("hidden");
}

function updateTitleFade() {
    if (!centerContent || !queryInput) return;
    if (chatStarted || currentMode !== "home") return;

    const hasText = queryInput.value.trim().length > 0;
    const focused = document.activeElement === queryInput;
    if (hasText || focused) centerContent.classList.add("fade-out");
    else centerContent.classList.remove("fade-out");
}

queryInput?.addEventListener("input", updateTitleFade);
queryInput?.addEventListener("focus", updateTitleFade);
queryInput?.addEventListener("blur", updateTitleFade);

/* ---------- chat utilities ---------- */

function scrollChatToBottom() {
    if (!chatContainer) return;
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addMessageBubble(text, sender) {
    if (!chatContainer) return;
    const row = document.createElement("div");
    row.classList.add("message-row", sender === "user" ? "user-row" : "bot-row");

    const bubble = document.createElement("div");
    bubble.classList.add(
        "message-bubble",
        sender === "user" ? "user-message" : "bot-message"
    );
    bubble.innerHTML = text; // Use innerHTML for Markdown/HTML content

    row.appendChild(bubble);
    chatContainer.appendChild(row);
    scrollChatToBottom();
    return row;
}

async function sendCyberChat(userText) {
    const formData = new FormData();
    formData.append("query", userText);

    try {
        const response = await fetch(`${API_BASE_URL}/api/chat`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            return `❌ Chat Error: ${errorData.detail || response.statusText}. Please check your API keys.`;
        }

        const data = await response.json();
        return data.reply;
    } catch (error) {
        console.error("Fetch error:", error);
        return "❌ Network error or backend server is down. Check if uvicorn is running.";
    }
}

function showTypingIndicator() {
    if (!chatContainer) return null;
    const row = document.createElement("div");
    row.classList.add("message-row", "bot-row");

    const bubble = document.createElement("div");
    bubble.classList.add("message-bubble", "bot-message", "typing-bubble");
    bubble.innerHTML =
        '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';

    row.appendChild(bubble);
    chatContainer.appendChild(row);
    scrollChatToBottom();
    return row;
}

/* ---------- File Notation Functions ---------- */

function updateFileNotation(file) {
    if (file) {
        stagedFile = file;
        // Limit file name display length
        fileNameDisplay.textContent =
            file.name.length > 20
                ? file.name.substring(0, 10) + "..." + file.name.slice(-7)
                : file.name;
        fileNotationContainer.classList.remove("hidden");
    } else {
        stagedFile = null;
        fileInput.value = ""; // Clear file input element
        fileNotationContainer.classList.add("hidden");
    }
}

clearFileBtn?.addEventListener("click", () => updateFileNotation(null));

/* ---------- Secure Detect Function ---------- */

async function runSecureDetect(prompt, file) {
    openEvaluationMode();
    addMessageBubble(
        `Running #evaluation on ${file.name} with prompt: ${prompt}`,
        "user"
    );

    // 1. Setup analysis state
    secureInstructions.textContent = `Analyzing file: ${file.name}...`;
    secureInstructions.classList.add("generating");

    // Create local object URL for preview (revoked later)
    const fileUrl = URL.createObjectURL(file);
    const fileName = file.name;

    const analyzingCard = document.createElement("div");
    analyzingCard.classList.add("secure-card", "analyzing-card");
    analyzingCard.innerHTML = `
        <div class="card-loading-placeholder">
            <div class="loader-dot"></div><div class="loader-dot"></div><div class="loader-dot"></div>
        </div>
        <div class="card-footer">
            <span class="card-prompt" title="${fileName}">Analyzing: ${fileName}...</span>
        </div>
    `;
    secureHistoryContainer.prepend(analyzingCard);

    // 2. Prepare Form Data for FastAPI
    const formData = new FormData();
    formData.append("file", file, fileName);
    formData.append("prompt", prompt);

    let llmReply = "Analysis failed due to an unknown error.";
    let result = null;
    let success = false;

    try {
        const response = await fetch(`${API_BASE_URL}/api/evaluate`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            result = await response.json();
            llmReply = `❌ Deepfake Evaluation failed. Error: ${
                result.detail || response.statusText
            }`;
        } else {
            result = await response.json();
            llmReply = result.llm_reply;
            success = true;
        }
    } catch (error) {
        console.error("Fetch error:", error);
        llmReply =
            "❌ Network error: Could not connect to the FastAPI backend. Check server console.";
    }

    // 3. Update the card with results
    analyzingCard.classList.remove("analyzing-card");

    if (success) {
        const { risk_level, fake_prob, math_notation, media_type } = result;
        const color =
            risk_level === "HIGH"
                ? "red"
                : risk_level === "MEDIUM"
                ? "orange"
                : "green";

        // Prepare File Preview HTML
        let filePreviewHTML = "";
        if (media_type === "image") {
            filePreviewHTML = `<p><strong>File Preview:</strong></p><img src="${fileUrl}" class="secure-file-preview" alt="File Preview">`;
        } else if (media_type === "video") {
            filePreviewHTML = `<p><strong>File Preview (Video):</strong></p><video src="${fileUrl}" class="secure-file-preview" controls></video>`;
        }

        analyzingCard.innerHTML = `
            <div class="secure-result-content">
                <p class="file-name-label">File: ${fileName}</p>
                <div class="secure-metrics">
                    <p><strong>Risk:</strong> <span style="color:${color};">${risk_level}</span></p>
                    <p><strong>Prob (real):</strong> ${fake_prob.toFixed(4)}</p>
                    <p class="math-notation">$$${math_notation}$$</p>
                </div>
                ${filePreviewHTML}
            </div>
            <div class="card-footer">
                <button class="icon-btn copy-btn" data-prob="${fake_prob.toFixed(
                    4
                )}" data-risk="${risk_level}" data-name="${fileName}">&#x1F4CB;</button>
            </div>
        `;
        secureInstructions.textContent =
            "Analysis complete. Results added to history.";
        addMessageBubble(
            `🔍 Deepfake Evaluation Result for ${fileName}: ${llmReply}`,
            "bot"
        );
    } else {
        // Error Card
        analyzingCard.classList.add("error-card");
        analyzingCard.innerHTML = `
            <div class="error-content">
                <h3>❌ Evaluation Failed</h3>
                <p>Error processing file: ${result.detail || "Check server logs."}</p>
            </div>
            <div class="card-footer">
                <span class="card-prompt" title="${fileName}">Error for: ${fileName}</span>
            </div>
        `;
        secureInstructions.textContent = "Analysis failed. See error in chat.";
        addMessageBubble(llmReply, "bot");
    }

    secureInstructions.classList.remove("generating");
    openEvaluationMode(); // Keep the panel open after evaluation
    updateFileNotation(null); // Clear staged file after use
    URL.revokeObjectURL(fileUrl); // Clean up memory
}

/* ---------- mode handlers (FIXED: Added chatStarted = true) ---------- */

function enterChatMode() {
    if (!hero) return;
    chatStarted = true;
    currentMode = "chat";
    hero.classList.add("chat-open");
    hideArrows();
    setPlaceholder("Type something..");
}

function openEvaluationMode() {
    if (!hero) return;
    chatStarted = true; // FIX: Lock the session state
    currentMode = "evaluation";
    hero.classList.add("evaluation-open");
    hero.classList.remove("generation-open");
    hero.classList.add("chat-open"); // keeps title hidden
    hideArrows();
    setPlaceholder("#evaluation.....");
    // AUTO-FILL COMMAND
    queryInput.value = "#evaluation ";
    queryInput.focus();
}

function openGenerationMode() {
    if (!hero) return;
    chatStarted = true; // FIX: Lock the session state
    currentMode = "generation";
    hero.classList.add("generation-open");
    hero.classList.remove("evaluation-open");
    hero.classList.add("chat-open");
    hideArrows();
    setPlaceholder("#generate image............");
    // AUTO-FILL COMMAND
    queryInput.value = "#generate image ";
    queryInput.focus();
}

// Fix for closing panels and returning to home/chat state
function closePanels() {
    if (!hero) return;
    hero.classList.remove("evaluation-open", "generation-open");

    if (chatStarted) {
        currentMode = "chat";
        hero.classList.add("chat-open");
        hideArrows();
    } else {
        currentMode = "home";
        hero.classList.remove("chat-open");
        showArrows();
        setPlaceholder("Type something..");
        updateTitleFade();
    }
}

/* ---------- Image Generation Function (FIXED) ---------- */

async function generateAndDisplayImage(prompt) {
    generationInstructions.textContent = "Generating image...";
    generationInstructions.classList.add("generating");

    openGenerationMode();
    addMessageBubble(`#generate image ${prompt}`, "user");

    // Display initial loading card
    const generatingCard = document.createElement("div");
    generatingCard.classList.add("image-card", "generating-card");
    generatingCard.innerHTML = `
        <div class="card-loading-placeholder">
            <div class="loader-dot"></div><div class="loader-dot"></div><div class="loader-dot"></div>
        </div>
        <div class="card-footer">
            <span class="card-prompt" title="${prompt}">Generating: ${prompt.substring(
        0,
        30
    )}...</span>
        </div>
    `;
    imageHistoryContainer.prepend(generatingCard);

    // Prepare Form Data for FastAPI
    const formData = new FormData();
    formData.append("prompt", prompt);

    let success = false;
    let image_urls = [];
    let botMessage = "";
    let result = null;

    try {
        const response = await fetch(`${API_BASE_URL}/api/generate_image`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            result = await response.json();
            botMessage = `❌ Image generation failed. Error: ${
                result.detail || response.statusText
            }`;
        } else {
            result = await response.json();
            image_urls = result.image_urls; // Expecting an array of paths
            botMessage = `Your new image${
                image_urls.length > 1 ? "s" : ""
            } has been generated and added to the panel history.`;
            success = true;
        }
    } catch (error) {
        console.error("Fetch error:", error);
        botMessage =
            "❌ Network error: Could not connect to the FastAPI backend.";
    }

    // 1. Remove the loading card
    if (generatingCard && generatingCard.parentNode) {
        generatingCard.parentNode.removeChild(generatingCard);
    }

    // 2. Display results (or failure card)
    if (success && image_urls && image_urls.length > 0) {
        image_urls.forEach((imageUrlPath) => {
            const finalCard = document.createElement("div");
            finalCard.classList.add("image-card");

            // The image URL must include the API_BASE_URL to access the file
            const fullImageUrl = `${API_BASE_URL}${imageUrlPath}`;

            finalCard.innerHTML = `
                <img src="${fullImageUrl}" alt="AI Generated Image: ${prompt}" class="history-image">
                <div class="card-footer">
                    <span class="card-prompt" title="${prompt}">${prompt.substring(
                0,
                30
            )}...</span>
                    <div class="action-buttons">
                        <button class="icon-btn view-btn" data-url="${fullImageUrl}">&#x27A2;</button>
                        <button class="icon-btn download-btn" data-url="${fullImageUrl}">&#x2b07;</button>
                        <button class="icon-btn share-btn" data-url="${fullImageUrl}">&#x2935;</button>
                    </div>
                </div>
            `;
            imageHistoryContainer.prepend(finalCard);
        });

        generationInstructions.textContent =
            "Click icons for options, or type a new prompt.";
    } else {
        // Display a single failure card
        const failureCard = document.createElement("div");
        failureCard.classList.add("image-card", "error-card");
        failureCard.innerHTML = `
            <div class="error-content">
                <h3>❌ Generation Failed</h3>
                <p>Error: ${
                    botMessage.includes("Error:")
                        ? botMessage.split("Error:")[1]
                        : "See chat for details."
                }</p>
            </div>
            <div class="card-footer">
                <span class="card-prompt" title="${prompt}">Error for: ${prompt.substring(
            0,
            30
        )}...</span>
            </div>
        `;
        imageHistoryContainer.prepend(failureCard);
        generationInstructions.textContent =
            "Image generation failed. Try a different prompt.";
    }

    // 3. Final State Update
    addMessageBubble(botMessage, "bot");
    generationInstructions.classList.remove("generating");
    // Ensure the panel is open after the process finishes.
    openGenerationMode();
}

/* ---------- send handler (DEFINITIVE FIX APPLIED HERE) ---------- */

async function handleSend() {
    if (!queryInput) return;
    const text = queryInput.value.trim();
    if (!text) return;

    const lower = text.toLowerCase();
    let typingRow = null;

    // Force entering chat mode immediately for ANY interaction.
    enterChatMode();

    // 1. Secure Detect Logic
    if (lower.includes("#evaluation")) {
        const evaluationIndex =
            lower.indexOf("#evaluation") + "#evaluation".length;
        const prompt = text.substring(evaluationIndex).trim();

        if (stagedFile) {
            queryInput.value = "";
            await runSecureDetect(prompt, stagedFile);
            return;
        } else {
            openEvaluationMode();
            addMessageBubble(text, "user");
            addMessageBubble(
                "Please upload a file (image/video) using the '+' button before running #evaluation.",
                "bot"
            );
            queryInput.value = "";
            return;
        }
    }

    // 2. Image Generation Logic
    if (lower.includes("#generate image") || lower.includes("#generate")) {
        const generationIndex =
            lower.indexOf("#generate image") + "#generate image".length;
        const prompt = text.substring(generationIndex).trim();

        queryInput.value = "";

        if (prompt.length > 0) {
            await generateAndDisplayImage(prompt);
        } else {
            openGenerationMode();
            generationInstructions.textContent =
                "Please type a detailed prompt in the search bar (e.g., #generate image a space cat).";
        }
        return;
    }

    // 3. Normal Chat Reply
    addMessageBubble(text, "user");
    queryInput.value = "";
    queryInput.focus();

    typingRow = showTypingIndicator();

    // Call the real API
    const botResponseText = await sendCyberChat(text);

    if (typingRow && typingRow.parentNode) {
        typingRow.parentNode.removeChild(typingRow);
    }
    addMessageBubble(botResponseText, "bot");
}

/* ---------- events (No Change) ---------- */

sendBtn?.addEventListener("click", handleSend);

// File upload activation
addFileBtn?.addEventListener("click", () => {
    if (fileInput) {
        fileInput.click();
    }
});

// Listener to handle selected files and update notation
fileInput?.addEventListener("change", (e) => {
    const files = e.target.files;
    if (files.length > 0) {
        updateFileNotation(files[0]);
    }
    fileInput.value = "";
});

// Functionality for View, Download, Share actions (Image Generation Panel)
imageDisplayArea?.addEventListener("click", (e) => {
    const target = e.target;
    const button = target.closest(".icon-btn");

    if (button) {
        const imageUrl = button.dataset.url;

        if (button.classList.contains("view-btn")) {
            window.open(imageUrl, "_blank");
        } else if (button.classList.contains("download-btn")) {
            fetch(imageUrl)
                .then((response) => response.blob())
                .then((blob) => {
                    const url = URL.createObjectURL(blob);
                    const link = document.createElement("a");
                    link.href = url;
                    link.download = `ai_image_${new Date().getTime()}.png`;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    URL.revokeObjectURL(url);
                })
                .catch((err) => console.error("Download failed:", err));
        } else if (button.classList.contains("share-btn")) {
            // Replaced alert with custom console error
            navigator.clipboard
                .writeText(imageUrl)
                .then(() => console.log("Image URL copied to clipboard!"))
                .catch((err) =>
                    console.error("Could not copy text: ", err)
                );
        }
    }
});

// Functionality for Copy button (Secure Detect Panel)
secureResultsBox?.addEventListener("click", (e) => {
    const target = e.target;
    const button = target.closest(".copy-btn");

    if (button) {
        const name = button.dataset.name;
        const prob = button.dataset.prob;
        const risk = button.dataset.risk;

        const textToCopy = `Secure Detect Results for ${name}:\nRisk: ${risk}\nProbability: ${prob}`;
        // Replaced alert with custom console error
        navigator.clipboard
            .writeText(textToCopy)
            .then(() =>
                console.log(
                    `ML Results for ${name} copied to clipboard!`
                )
            )
            .catch((err) => console.error("Copy failed:", err));
    }
});

rightArrow?.addEventListener("click", openEvaluationMode);
leftArrow?.addEventListener("click", openGenerationMode);

// keyboard shortcuts
document.addEventListener("keydown", (e) => {
    const focusedInInput = document.activeElement === queryInput;

    // Enter → send
    if (e.key === "Enter" && focusedInInput) {
        e.preventDefault();
        handleSend();
        return;
    }

    // Arrows for panels (only when input not focused)
    if (!focusedInInput) {
        if (e.key === "ArrowRight") {
            openEvaluationMode();
        } else if (e.key === "ArrowLeft") {
            openGenerationMode();
        } else if (e.key === "ArrowDown") {
            closePanels();
        }
    }

    // W / S scroll chat (only when not typing)
    if (!focusedInInput && chatContainer) {
        const amount = 80; // pixels per press
        if (e.key === "w" || e.key === "W") {
            chatContainer.scrollTop -= amount;
        } else if (e.key === "s" || e.key === "S") {
            chatContainer.scrollTop += amount;
        }
    }
});

// FINAL Initialization: Set full instructions text after load
if (generationInstructions) {
    generationInstructions.textContent =
        "Type #generate image and a prompt in the chat bar to begin.";
}
