from io import BytesIO
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, UploadFile, Response
from inference import get_model
import supervision as sv
from inference.core.utils.image_utils import load_image_bgr
import cv2
from starlette.responses import StreamingResponse
from ultralytics import YOLO
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/upload-image")
async def upload_image(file: UploadFile):
    # content = await file.read()
    # return Response(content=content, media_type=file.content_type)
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    image = load_image_bgr(temp_file_path)
    model = YOLO(os.getcwd()+'/runs/detect/train6/weights/best.pt')
    # model = YOLO('yolov8l.pt')
    results = model(image)[0].plot()

    _, encoded_image = cv2.imencode('.jpg', results)
    image_bytes = BytesIO(encoded_image.tobytes())

    return StreamingResponse(image_bytes, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
