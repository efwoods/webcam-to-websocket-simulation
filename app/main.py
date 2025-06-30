import os
import re
import json
import pickle
import base64
from io import BytesIO
from contextlib import asynccontextmanager
from pydantic import BaseModel

from fastapi import FastAPI
from PIL import Image
import websockets
from dotenv import load_dotenv
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

import uvicorn
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from fastapi import Request
from core.logging import logger

from contextlib import asynccontextmanager

import requests

# Configurations & Metrics
# from core.config import settings
# from core.monitoring import metrics

# API Routes
# from api.routes import router


from data.dataset import ImageWaveformDataset

# -----------------------------
# Load .env and assert
# -----------------------------
load_dotenv()
WS_URI = os.getenv("WS_URI")
assert WS_URI is not None, "WS_URI not loaded from .env"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GIST_ID = os.getenv("GIST_ID")  # Your gist ID
GIST_API_URL = f"https://api.github.com/gists/{GIST_ID}"

USE_PERCEPTUAL_LOSS = False
RESIZE_DIM = (224, 224) if USE_PERCEPTUAL_LOSS else (64, 64)


# -----------------------------
# Dataset paths
# -----------------------------
waveform_dict_path = "data/collected_stimulus_waveforms.pkl"
image_dict_path = "data/image_paths_dict.pkl"
test_metadata_path = "data/test_dataset_metadata.pkl"

# -----------------------------
# Transform
# -----------------------------
image_resize_transform = transforms.Compose(
    [
        transforms.Resize(RESIZE_DIM),
    ]
)


# -----------------------------
# Input Model
# -----------------------------
class SimulationRequest(BaseModel):
    session_id: str  # This is the encrypted hash of the user_id_avatar_id


# -----------------------------
# Load Dataset
# -----------------------------
with open(waveform_dict_path, "rb") as f:
    waveform_dict = pickle.load(f)
with open(image_dict_path, "rb") as f:
    image_paths = pickle.load(f)
with open(test_metadata_path, "rb") as f:
    test_meta = pickle.load(f)
test_indices = test_meta["indices"]
dataset = ImageWaveformDataset(waveform_dict, image_paths, image_resize_transform)


# -----------------------------
# Convert PIL Image → Base64
# -----------------------------
def pil_image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_img}"


# -----------------------------
# WebSocket Sender
# -----------------------------
async def send_image(session_id: str, image: Image.Image):
    image_base64 = pil_image_to_base64(image)
    payload = {
        "type": "simulate",
        "session_id": session_id,
        "image_base64": image_base64,
    }
    try:
        async with websockets.connect(app.state.ws_url_test) as websocket:
            await websocket.send(json.dumps(payload))
            response = await websocket.recv()
            print(f"[{session_id}] Response: {response}")
            return response
    except Exception as e:
        print(f"[ERROR] {session_id}: {e}")
        return json.dumps({"error": str(e)})


# -----------------------------
# Test Image Simulation
# -----------------------------
async def simulate_all_images(session_id: str, index: int):
    idx = test_indices[index % len(test_indices)]
    pil_image, _ = dataset[idx]
    logger.info(f"type(image_tensor):{type(pil_image)}")
    response = await send_image(session_id, pil_image)
    return response


# -----------------------------
# Extract NGROK URL
# -----------------------------
def extract_ngrok_url(text: str) -> str:
    """
    Extract the ngrok URL from raw gist text using regex
    """
    match = re.search(r"https://[\w\-]+\.ngrok-free\.app", text)
    if match:
        return match.group(0).replace("https", "wss")
    raise ValueError("No ngrok URL found in gist")


# -----------------------------
# Fetch NGROK URL
# -----------------------------
def fetch_ngrok_url():
    global ngrok_url

    if not GITHUB_TOKEN:
        print("GitHub token is missing in environment variable `GITHUB_TOKEN`.")
        return

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    try:
        response = requests.get(GIST_API_URL, headers=headers, timeout=10)
        response.raise_for_status()
        gist_data = response.json()

        # Assuming there's only one file in the Gist, extract its content
        file_content = next(iter(gist_data["files"].values()))["content"]

        # Extract ngrok URL
        ngrok_url = extract_ngrok_url(file_content)
        print(f"[Startup] Ngrok URL loaded: {ngrok_url}")

    except Exception as e:
        print(f"[Startup] Failed to fetch or parse Gist: {e}")


# -----------------------------
# FastAPI App
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] FastAPI simulation sender is live")
    app.state.simulation_index = 0
    fetch_ngrok_url()
    app.state.ws_url_test = ngrok_url + WS_URI
    yield
    print("[Shutdown] Shutting down sender...")


app = FastAPI(
    title="Webcam to Websocket Simulation API",
    root_path="/webcam-to-websocket-simulation-api",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
# app.include_router(router, prefix="/simulate", tags=["Simulate"])


@app.post("/simulate-test-images")
async def simulate_test_images(payload: SimulationRequest):
    index = app.state.simulation_index
    response = await simulate_all_images(payload.session_id, index)
    app.state.simulation_index += 1
    return {"response": response}


@app.post("/simulate-webcam-stream")
async def simulate_test_images(payload: SimulationRequest):
    # Simulate Webcam Images
    return {"response": "success"}


@app.post("/simulate-test-waveform")
async def simulate_test_images(payload: SimulationRequest):
    # Simulate Test Waveform Logic
    return {"response": "success"}


@app.get("/")
async def root(request: Request):
    return RedirectResponse(url=request.scope.get("root_path", "") + "/docs")


@app.get("/health")
async def health():
    # metrics.health_requests.inc()
    return {"status": "healthy"}


@app.router.get("/metrics")
def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
