from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse,StreamingResponse
from pydantic import BaseModel
from PIL import Image
import tempfile
import torch
from diffusers import StableDiffusionPipeline
import base64
import numpy as np
import cv2
from io import BytesIO
import traceback
 
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


def convert_to_dotted_line_on_mask(line_img, mask_img, dot_length=10, gap_length=10):
    if line_img.shape[2] == 4:
        line_img = cv2.cvtColor(line_img, cv2.COLOR_BGRA2BGR)
    if mask_img.shape[2] == 4:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGRA2BGR)
    if line_img.shape[:2] != mask_img.shape[:2]:
        mask_img = cv2.resize(mask_img, (line_img.shape[1], line_img.shape[0]))

    gray_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_mask, 10, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    safe_mask = cv2.erode(binary_mask, kernel, iterations=3)

    gray_line = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    _, binary_line = cv2.threshold(gray_line, 127, 255, cv2.THRESH_BINARY_INV)

    masked_lines = cv2.bitwise_and(binary_line, safe_mask)

    output = line_img.copy()
    white_background = np.full_like(line_img, 255)
    inverse_mask = cv2.bitwise_not(binary_mask)
    outside = cv2.bitwise_and(line_img, line_img, mask=inverse_mask)
    inside_white = cv2.bitwise_and(white_background, white_background, mask=binary_mask)
    output = cv2.add(inside_white, outside)

    contours, _ = cv2.findContours(masked_lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        points = [tuple(pt[0]) for pt in contour]
        if len(points) < 2:
            continue

        current_segment_length = 0
        is_dot = True

        for i in range(1, len(points)):
            pt1, pt2 = points[i - 1], points[i]
            segment_length = np.linalg.norm(np.array(pt2) - np.array(pt1))
            remaining_length = segment_length
            start_pt = pt1

            while remaining_length > 0:
                seg_len = min(remaining_length, dot_length - current_segment_length) if is_dot else min(remaining_length, gap_length - current_segment_length)
                ratio = seg_len / segment_length
                end_x = int(start_pt[0] + ratio * (pt2[0] - start_pt[0]))
                end_y = int(start_pt[1] + ratio * (pt2[1] - start_pt[1]))
                end_pt = (end_x, end_y)

                if is_dot:
                    cv2.line(output, start_pt, end_pt, (0, 0, 0), 1)

                current_segment_length += seg_len
                remaining_length -= seg_len
                start_pt = end_pt

                if (is_dot and current_segment_length >= dot_length) or (not is_dot and current_segment_length >= gap_length):
                    is_dot = not is_dot
                    current_segment_length = 0

    return output
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
    print("starting generation...")

    # Generate list of images
    images = [pipe(req.prompt, guidance_scale=req.guidance_scale).images[0]
              for _ in range(req.num_images)]

    # Save each image to a temporary file and encode as base64
    image_b64_list = []
    for img in images:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img.save(tmp.name)
            with open(tmp.name, "rb") as f:
                img_bytes = f.read()
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                image_b64_list.append(img_b64)

    return {"images": image_b64_list}

@app.post("/process/")
async def process_image(
    image: UploadFile = File(...),
    mask: UploadFile = File(...)
):
    try:
        image_bytes = await image.read()
        mask_bytes = await mask.read()

        line_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        mask_img = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

        if line_img is None or mask_img is None:
            raise HTTPException(status_code=400, detail="Invalid image or mask.")

        result = convert_to_dotted_line_on_mask(line_img, mask_img, dot_length=10, gap_length=10)
        _, encoded_image = cv2.imencode('.png', result)

        return StreamingResponse(BytesIO(encoded_image.tobytes()), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
