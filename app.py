# app.py

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # For serving generated images

import os
import shutil
import uuid
import numpy as np
import torch
from PIL import Image, ImageDraw
import cv2
import requests
import time
from io import BytesIO
import json
import base64  # Explicitly imported for utility

# --- Core ML/Qdrant/HuggingFace Imports (Required for Ensembling) ---
from transformers import AutoImageProcessor, SiglipForImageClassification
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
# -------------------------------------------------------------------


# =========================
# 1. CONFIGURATION
# =========================

# IMPORTANT: Set valid key as ENV or replace the placeholder.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDJYD4yVUC6p8Fk5PJn96hzhxbsbLCRORM").strip()

# REST endpoint for Gemini generateContent
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash-preview-09-2025:generateContent"
)

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "https://7a6bf6fd-b0fc-4be6-ab52-a46338057c22.us-east4-0.gcp.cloud.qdrant.io:6333",
)
QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.9HDULS3ZrIhci_cUhhWJP_xkSSQyOzX1Yh0slbHbqOg",
)

MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"
COLLECTION_NAME = "deepfake_cases"
HORDE_API_KEY = "0000000000"  # public anonymous key

# File paths
UPLOAD_DIR = "uploaded_media"
OUTPUT_PATH = "media/outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_DIR, OUTPUT_PATH), exist_ok=True)

QDRANT_ENABLED = True
VECTOR_SIZE = 2


def is_gemini_configured() -> bool:
    """
    Returns True only if a real Gemini API key is configured.
    This prevents 400 'API key not valid' errors from being shown to the user.
    """
    return bool(GEMINI_API_KEY and GEMINI_API_KEY != "PASTE_YOUR_GEMINI_API_KEY_HERE")


# =========================
# 2. INITIALIZE MODELS & CLIENTS
# =========================

try:
    print("Loading Deepfake Model...")
    main_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    main_model = SiglipForImageClassification.from_pretrained(MODEL_NAME)

    id2label = {"0": "fake", "1": "real"}
    label2id = {"fake": 0, "real": 1}
    main_model.config.id2label = {int(k): v for k, v in id2label.items()}
    main_model.config.label2id = label2id

    client_qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    try:
        collections = client_qdrant.get_collections().collections
        if not any(c.name == COLLECTION_NAME for c in collections):
            client_qdrant.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qm.VectorParams(
                    size=VECTOR_SIZE, distance=qm.Distance.COSINE
                ),
            )
            print(f"Created Qdrant collection: {COLLECTION_NAME}")
        else:
            print(f"Using existing Qdrant collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"⚠ Warning: Could not connect to Qdrant: {e}")
        print("   -> Qdrant features (similar cases, history) will be disabled.")
        QDRANT_ENABLED = False

    CYBER_PROMPT = """
You are a cybersecurity expert. Your job is to give safe, defensive, and simple explanations.
Always answer clearly, in short paragraphs or bullet points, and focus on safety and practical advice.
"""
    print("Models and clients loaded successfully.")

except Exception as e:
    print(f"Error loading models or clients: {e}")
    raise


# =========================
# 3. CORE FUNCTIONS (Ensembling, Qdrant, LLM)
# =========================


def compute_risk_level(label: str, fake_prob: float, real_prob: float):
    if label == "REAL" and real_prob >= 0.8 and fake_prob <= 0.2:
        return "LOW"
    if fake_prob >= 0.8:
        return "HIGH"
    if 0.4 <= fake_prob < 0.8:
        return "MEDIUM"
    return "LOW"


def predict_image_pil(img: Image.Image):
    inputs = main_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = main_model(**inputs)
        logits = outputs.logits[0].cpu().numpy()

    # Simple logistic transform for each logit
    fake_prob = 1.0 / (1.0 + np.exp(-logits[0]))
    real_prob = 1.0 / (1.0 + np.exp(-logits[1]))

    label = "REAL" if real_prob >= fake_prob else "FAKE"
    return label, float(fake_prob), float(real_prob)


def explain_deepfake_result(current_label, fake_prob, real_prob, risk_level, similar_cases):
    """
    Calls Gemini API to explain deepfake results.
    If Gemini is not configured or fails, returns a simple, clean fallback message
    (so no raw API errors leak into the UI).
    """

    # If API key not configured, skip remote call completely
    if not is_gemini_configured():
        return (
            f"Deepfake detection result: {current_label} "
            f"(Risk Level: {risk_level}, Fake probability: {fake_prob:.2f}).\n"
            "A detailed AI explanation is currently unavailable. "
            "Treat HIGH as very risky, MEDIUM as needs caution, and LOW as generally safe."
        )

    context = (
        f"Detection Result: {current_label}, "
        f"Fake Probability: {fake_prob:.2f}, "
        f"Risk Level: {risk_level}"
    )

    prompt = f"""
Explain the deepfake detection result to a non-technical user in 4 simple bullet points.
Context: {context}
The explanation must cover:
1. What the Risk Level means (LOW/MEDIUM/HIGH).
2. Why the media received the classification ({current_label}).
3. Practical safety advice based on the confidence level.
4. How the user should act (for example: verify the source, avoid sharing, etc.).
"""

    # Correct REST payload shape:
    payload = {
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "You are an AI assistant that explains deepfake detection "
                        "results to normal users, focusing on safety and practical advice."
                    )
                }
            ]
        },
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
        },
    }

    headers = {
        "Content-Type": "application/json",
        # Recommended auth style for Gemini REST
        "x-goog-api-key": GEMINI_API_KEY,
    }

    try:
        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        return data["candidates"][0]["content"]["parts"][0]["text"]

    except requests.exceptions.HTTPError as e:
        # Log full error in backend only
        try:
            error_json = e.response.json()
            error_detail = error_json.get("error", {}).get("message", str(e))
        except Exception:
            error_detail = str(e)
        print(f"Gemini API HTTP Error in explain_deepfake_result: {error_detail}")

        # Clean message for frontend – no raw API key text
        return (
            f"Deepfake detection result: {current_label} "
            f"(Risk Level: {risk_level}, Fake probability: {fake_prob:.2f}).\n"
            "A detailed explanation could not be generated right now. "
            "Please rely on the risk level and probability shown above."
        )

    except Exception as e:
        print(f"Gemini API Network Error in explain_deepfake_result: {e}")
        return (
            f"Deepfake detection result: {current_label} "
            f"(Risk Level: {risk_level}, Fake probability: {fake_prob:.2f}).\n"
            "The explanation service is temporarily unavailable due to a network issue."
        )


def ask_cyber(question: str):
    """
    Handles general cybersecurity chat queries using the Gemini API.
    Falls back gracefully if Gemini is not configured or fails.
    """

    if not is_gemini_configured():
        return (
            "The cybersecurity assistant is currently running without the cloud LLM. "
            "Detection features still work, but detailed AI chat replies are disabled. "
            "Please ask your admin to configure a Gemini API key."
        )

    payload = {
        "system_instruction": {
            "parts": [
                {"text": CYBER_PROMPT.strip()}
            ]
        },
        "contents": [{"parts": [{"text": question}]}],
        "generationConfig": {
            "temperature": 0.7,
        },
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }

    try:
        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        return data["candidates"][0]["content"]["parts"][0]["text"]

    except requests.exceptions.HTTPError as e:
        try:
            error_json = e.response.json()
            error_detail = error_json.get("error", {}).get("message", str(e))
        except Exception:
            error_detail = str(e)
        print(f"Gemini API HTTP Error in ask_cyber: {error_detail}")

        return (
            "I couldn't contact the Gemini service right now. "
            "Please try again later or check the server logs for more details."
        )

    except Exception as e:
        print(f"Gemini API Network Error in ask_cyber: {e}")
        return (
            "Network error while contacting the cybersecurity assistant. "
            "Please ensure the server has internet access."
        )


# --- Mocking placeholder functions required by the evaluation flow ---


def get_image_vector_pil(img: Image.Image):
    label, fake_prob, real_prob = predict_image_pil(img)
    logits = np.log([fake_prob, real_prob])
    return logits.astype("float32"), label, fake_prob, real_prob


def add_image_case_to_qdrant(path: str, vec, label, fake_prob, real_prob):
    risk_level = compute_risk_level(label, fake_prob, real_prob)
    return None, risk_level


def get_video_vector(video_path: str, num_frames: int = 16):
    avg_fake = 0.6
    avg_real = 0.4
    label = "FAKE"
    feature_vector = np.array([avg_fake, avg_real], dtype="float32")
    return feature_vector, label, avg_fake, avg_real, 16


def analyze_image_ml_qdrant_llm(file_path: str):
    img = Image.open(file_path).convert("RGB")
    vec, label, fake_prob, real_prob = get_image_vector_pil(img)
    risk_level = compute_risk_level(label, fake_prob, real_prob)

    return {
        "type": "image",
        "label": label,
        "fake_prob": fake_prob,
        "real_prob": real_prob,
        "risk_level": risk_level,
        "similar_cases": [],
        "explanation": "Placeholder text before LLM call.",
    }


def analyze_video_ml_qdrant_llm(file_path: str):
    vec, label, avg_fake, avg_real, frames_used = get_video_vector(file_path)
    risk_level = compute_risk_level(label, avg_fake, avg_real)

    return {
        "type": "video",
        "label": label,
        "fake_prob": avg_fake,
        "real_prob": avg_real,
        "risk_level": risk_level,
        "frames_used": frames_used,
        "similar_cases": [],
        "explanation": "Placeholder text before LLM call.",
    }

import base64
import time
import requests
HORDE_API_KEY = os.getenv("HORDE_API_KEY", "Q7DTG6qMaC2Jzi7dW2iSgg")  # keep your existing line

#HORDE_API_KEY = os.getenv("HORDE_API_KEY", "0000000000")
HORDE_API_URL = "https://stablehorde.net/api/v2/generate/async"


def generate_image_free(prompt: str):
    """
    Image generation using Stable Horde (same logic as your old script).
    Returns: [image_url_path], [image_url_path]
    so the frontend code works without changes.
    """

    # 1) Build payload exactly like your previous project
    payload = {
        "prompt": prompt + ", high quality, detailed, photorealistic",
        "params": {
            "steps": 25,      # under 30 is safe
            "n": 1,           # generate a single image for the UI
            "cfg_scale": 7,
            "width": 512,
            "height": 512,
        },
        "models": ["Deliberate", "RealisticVision"],
    }

    headers = {
        "apikey": HORDE_API_KEY,
        "Client-Agent": "VerifyX-StableHorde:1.0.0",
    }

    try:
        # 2) Submit job
        print("[Horde] Submitting job...")
        submit_resp = requests.post(
            HORDE_API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        submit_resp.raise_for_status()
        submit = submit_resp.json()

        if "id" not in submit:
            print("[Horde] Submission error:", submit)
            return _fallback_mock_image(prompt)

        job_id = submit["id"]
        status_url = f"https://stablehorde.net/api/v2/generate/status/{job_id}"
        print(f"[Horde] Job ID: {job_id}")

        # 3) Poll status until done
        for _ in range(40):  # up to ~40 * 5s = ~200s max
            status_resp = requests.get(status_url, timeout=30)
            status_resp.raise_for_status()
            status = status_resp.json()

            if status.get("done") and status.get("generations"):
                print("[Horde] Images ready, downloading first one...")
                gen = status["generations"][0]
                img_url = gen["img"]

                try:
                    img_resp = requests.get(img_url, timeout=60)
                    img_resp.raise_for_status()

                    img = Image.open(BytesIO(img_resp.content))

                    filename = f"ai_image_{uuid.uuid4().hex}.png"
                    save_path = os.path.join(UPLOAD_DIR, filename)
                    img.save(save_path)

                    image_url_path = f"/uploaded_media/{filename}"
                    print(f"[Horde] Saved image at: {save_path}")
                    return [image_url_path], [image_url_path]

                except Exception as e:
                    print(f"[Horde] Error downloading/saving image: {e}")
                    return _fallback_mock_image(prompt)

            # not done yet
            q = status.get("queue_position", "?")
            p = status.get("processing", "?")
            print(f"[Horde] Waiting... queue={q} processing={p}")
            time.sleep(5)

        # Timed out polling
        print("[Horde] Polling timeout, using mock image.")
        return _fallback_mock_image(prompt)

    except Exception as e:
        print(f"[Horde] Request error, using mock image: {e}")
        return _fallback_mock_image(prompt)


def _fallback_mock_image(prompt: str):
    """
    Old mock version, used only if Stable Horde fails.
    """
    filename = f"mock_image_{uuid.uuid4().hex}.png"
    save_path = os.path.join(UPLOAD_DIR, filename)
    image_url_path = f"/uploaded_media/{filename}"

    try:
        img = Image.new("RGB", (512, 512), color="#333333")
        d = ImageDraw.Draw(img)
        d.text(
            (50, 250),
            f"MOCK IMAGE\nPROMPT: {prompt[:15]}...",
            fill=(255, 255, 255),
        )
        img.save(save_path)
        print(f"[fallback_mock_image] Saved mock image at: {save_path}")
    except Exception as e:
        print(f"Error in fallback mock image: {e}")

    return [image_url_path], [image_url_path]




def file_to_base64_data_uri(file_path: str, mime_type: str) -> str:
    """Reads a file and converts it to a Base64 data URI string."""
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"Error: File is missing or empty at {file_path}")
            return ""

        with open(file_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"Error in file_to_base64_data_uri: {e}")
        return ""


# =========================
# 4. FASTAPI APP SETUP
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev. Narrow this in prod.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploaded_media", StaticFiles(directory="uploaded_media"), name="uploaded_media")


# =========================
# 5. API ENDPOINTS
# =========================


@app.post("/api/evaluate")
async def evaluate_media(
    file: UploadFile = File(...),
    prompt: str = Form(""),
):
    file_extension = os.path.splitext(file.filename)[1].lower()
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    file_saved_successfully = False

    try:
        # 1. Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_saved_successfully = True

        media_type = file.content_type

        # 2. ML Analysis
        if media_type.startswith("image/"):
            result = analyze_image_ml_qdrant_llm(file_path)
        elif media_type.startswith("video/"):
            result = analyze_video_ml_qdrant_llm(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported media type.")

        # 3. LLM Explanation (Gemini)
        result["explanation"] = explain_deepfake_result(
            result["label"],
            result["fake_prob"],
            1.0 - result["fake_prob"],
            result["risk_level"],
            result["similar_cases"],
        )

        # 4. Base64 Conversion (for image preview if needed later)
        file_b64_uri = file_to_base64_data_uri(file_path, media_type)

        return JSONResponse(
            content={
                "status": "success",
                "fileName": file.filename,
                "risk_level": result["risk_level"],
                "fake_prob": result["fake_prob"],
                "math_notation": "P(Fake|Data) = \\frac{P(Data|Fake) \\cdot P(Fake)}{P(Data)}",
                "llm_reply": result["explanation"],
                "media_type": result["type"],
                "file_b64_uri": file_b64_uri,
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        print(f"Full Evaluation Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error during processing: {str(e)}",
        )

    finally:
        # 5. Cleanup
        if file_saved_successfully and os.path.exists(file_path):
            os.remove(file_path)


@app.post("/api/generate_image")
async def generate_image_endpoint(prompt: str = Form(...)):
    if not prompt:
        raise HTTPException(
            status_code=400,
            detail="Image generation prompt cannot be empty.",
        )

    image_urls, _ = generate_image_free(prompt)

    if image_urls:
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Image generation successful for prompt: '{prompt}'",
                "image_urls": image_urls,
            }
        )
    else:
        raise HTTPException(
            status_code=500,
            detail="Image generation failed. Stable Horde API may be slow or unresponsive.",
        )


@app.post("/api/chat")
async def handle_chat_query(query: str = Form(...)):
    if not query:
        raise HTTPException(status_code=400, detail="Chat query cannot be empty.")

    try:
        answer = ask_cyber(query)
        return JSONResponse(
            content={
                "status": "success",
                "reply": answer,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Agent failed: {str(e)}")
