from PIL import Image
import os
import uuid

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def save_image(image: Image.Image, prompt: str) -> str:
    os.makedirs("generated", exist_ok=True)
    filename = f"{prompt[:50].replace(' ', '_')}_{uuid.uuid4().hex[:6]}.png"
    path = os.path.join("generated", filename)
    image.save(path)
    return path
