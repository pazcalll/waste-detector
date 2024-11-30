from io import BytesIO
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, UploadFile, Response
from inference import get_model
import supervision as sv
from inference.core.utils.image_utils import load_image_bgr
import cv2
from starlette.responses import StreamingResponse

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
    model = get_model(model_id="yolov8n-640")
    results = model.infer(image)[0]
    results = sv.Detections.from_inference(results)
    annotator = sv.BoxAnnotator(thickness=4)
    annotated_image = annotator.annotate(image, results)
    annotator = sv.LabelAnnotator(text_scale=2, text_thickness=2)
    annotated_image = annotator.annotate(annotated_image, results)
    sv.plot_image(annotated_image)

    _, encoded_image = cv2.imencode('.jpg', annotated_image)
    image_bytes = BytesIO(encoded_image.tobytes())

    return StreamingResponse(image_bytes, media_type="image/jpeg")