from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
import tempfile
import torch
from diffusers import StableDiffusionPipeline

# === Setup FastAPI ===
app = FastAPI()

# === CORS: Allow all origins (for frontend) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load model only once ===
pipe = StableDiffusionPipeline.from_pretrained(
    "Soumyajit94298/patent-backend",
    torch_dtype=torch.float16
).to("cuda")

# === Request schema ===
class PromptRequest(BaseModel):
    prompt: str
    num_images: int = 2
    guidance_scale: float = 7.0

# === Endpoint ===

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/generate")
async def root():
    return {"status": "ok"}

@app.post("/generate-image/")
async def generate_image(req: PromptRequest):
    print(f"Received prompt: {req.prompt}")

    # Generate list of images
    images = [pipe(req.prompt, guidance_scale=req.guidance_scale).images[0]
              for _ in range(req.num_images)]

    # Combine them horizontally
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    combined_img = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for img in images:
        combined_img.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        temp_path = tmp.name
        combined_img.save(temp_path)

    print(f"Saved to: {temp_path}")
    return FileResponse(temp_path, media_type="image/png", filename="generated.png")

