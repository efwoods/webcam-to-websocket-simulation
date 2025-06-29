import os
import json
import pickle
import base64
from io import BytesIO
from contextlib import asynccontextmanager

from fastapi import FastAPI
from PIL import Image
import websockets
from dotenv import load_dotenv
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from app.data.dataset import ImageWaveformDataset

# -----------------------------
# Load .env and assert
# -----------------------------
load_dotenv()
WS_BASE_PATH = os.getenv("WS_BASE_PATH")
WS_URI = os.getenv("WS_URI")
assert WS_BASE_PATH is not None, "WS_BASE_PATH not loaded from .env"
assert WS_URI is not None, "WS_URI not loaded from .env"
ws_uri = WS_BASE_PATH + WS_URI

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
async def send_image(ws_uri: str, session_id: str, image: Image.Image):
    image_base64 = pil_image_to_base64(image)
    payload = {
        "type": "simulate",
        "session_id": session_id,
        "image_base64": image_base64,
    }
    try:
        async with websockets.connect(ws_uri) as websocket:
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
async def simulate_all_images(index: int):
    idx = test_indices[index % len(test_indices)]
    image_tensor, _ = dataset[idx]
    pil_img = to_pil_image(image_tensor)  # Convert tensor to original PIL Image
    session_id = f"test-{index}"
    response = await send_image(ws_uri, session_id, pil_img)
    return response


# -----------------------------
# FastAPI App
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] FastAPI simulation sender is live")
    app.state.simulation_index = 0
    yield
    print("[Shutdown] Shutting down sender...")


app = FastAPI(lifespan=lifespan)


@app.get("/simulate-test-images")
async def simulate_test_images():
    index = app.state.simulation_index
    response = await simulate_all_images(index)
    app.state.simulation_index += 1
    return {"response": response}
