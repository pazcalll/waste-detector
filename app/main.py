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
    model = YOLO(os.getcwd()+'/runs/detect/train5/weights/best.pt')
    results = model(image)[0]
    boxes = results.boxes.cls.numpy()
    array_of_object = []
    for i in range(len(boxes)):
        push: bool = True
        model_name: str = model.names[int(boxes[i])]
        for j in range(len(array_of_object)):
            if array_of_object[j]["name"] == model_name:
                array_of_object[j]["count"] += 1
                push = False
        if push:
            array_of_object.append({"name": model_name, "count": 1})

    print(array_of_object)

    results = model(image)[0].plot()

    _, encoded_image = cv2.imencode('.jpg', results)
    image_bytes = BytesIO(encoded_image.tobytes())

    # plastik, kertas, kaca, logam
    return StreamingResponse(image_bytes, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
