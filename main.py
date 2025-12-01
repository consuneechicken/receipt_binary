from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import cv2
import joblib
import base64
from pillow_heif import read_heif
from PIL import Image

from feature import extract_features_from_array

app = FastAPI()

# Load model
model = joblib.load("svm_model.pkl")

class ImageData(BaseModel):
    image_base64: str


def decode_base64_image(b64_string: str):
    """Base64 이미지 → OpenCV 이미지(ndarray)
       JPG/PNG/HEIC 자동 처리
    """

    b64 = b64_string.strip()

    # 1) base64 헤더 제거 ("data:image/...;base64," 형태)
    if "," in b64:
        b64 = b64.split(",")[1]

    # 2) base64 패딩 보정
    missing_padding = len(b64) % 4
    if missing_padding:
        b64 += "=" * (4 - missing_padding)

    # 3) base64 → bytes
    image_bytes = base64.b64decode(b64)

    # 4) HEIC인지 매직헤더로 판단
    #    HEIC/HEIF 파일은 내부에 "ftypheic", "ftypheif" 등이 포함됨
    header = image_bytes[:16]
    if b"ftypheic" in header or b"ftypheif" in header:
        heif = read_heif(image_bytes)
        img = Image.frombytes(
            heif.mode,
            heif.size,
            heif.data,
            "raw",
            heif.mode
        )
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 5) 일반 JPG/PNG 처리
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return img


@app.get("/")
def home():
    return {"status": "ok", "message": "receipt classifier running"}


@app.post("/predict")
async def predict(data: ImageData):
    try:
        img = decode_base64_image(data.image_base64)

        if img is None:
            return {"error": "failed to decode base64 image"}

        features = extract_features_from_array(img)
        pred = model.predict([features])[0]
        label = "paper_receipt" if pred == 0 else "online_receipt"

        return {"result": label}

    except Exception as e:
        return {"error": str(e)}
