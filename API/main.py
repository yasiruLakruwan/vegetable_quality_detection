import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import base64
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust if frontend runs on a different address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load both models
mobilenet_model = load_model('hybrid_veg_model.h5')  # MobileNetV2 model
yolo_model = YOLO('rotten_detection.pt')  # YOLOv8 model

# Class names for the MobileNetV2 model
mobilenet_class_names = ['Carrot', 'Potato', 'Tomato']

# Class names for the YOLOv8 model
yolo_class_names = ['healthy carrot', 'rotten carrot', 'healthy tomato', 'rotten tomato', 'healthy potato', 'rotten potato']

def read_imagefile(file) -> Image.Image:
    img = Image.open(BytesIO(file))
    return img

def preprocess_image(img: Image.Image):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.post("/predict_quality/")
async def predict_quality(file: UploadFile = File(...)):
    try:
        img = read_imagefile(await file.read())
        img_array = preprocess_image(img)
        predictions = mobilenet_model.predict(img_array)
        predicted_class = mobilenet_class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        return JSONResponse(content={"class": predicted_class, "confidence": confidence})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



# POST endpoint for vegetable damage detection (YOLOv8) with bounding box overlay
@app.post("/predict_damage/")
async def predict_damage(file: UploadFile = File(...)):
    try:
        # Load the uploaded image
        img = read_imagefile(await file.read())
        
        # Convert image to OpenCV format
        img_cv2 = np.array(img)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

        # Run prediction using YOLOv8
        results = yolo_model.predict(source=img_cv2, conf=0.25)

        # Draw bounding boxes and prepare data for response
        predictions = []
        for result in results[0].boxes:
            box = result.xyxy[0].tolist()  # Bounding box coordinates
            conf = float(result.conf[0].item())  # Confidence score
            cls = int(result.cls[0].item())  # Class ID
            label = yolo_class_names[cls]  # Get class label

            # Draw bounding box on the image
            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[2]), int(box[3]))
            color = (255, 0, 0)  # Red color for bounding box
            cv2.rectangle(img_cv2, start_point, end_point, color, 2)
            cv2.putText(img_cv2, f"{label} {conf:.2f}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            predictions.append({
                "class": label,
                "confidence": conf,
                "bbox": box
            })

        # Convert image with bounding box to base64
        _, buffer = cv2.imencode('.jpg', img_cv2)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        # Return bounding box information and image
        return JSONResponse(content={"predictions": predictions, "image_base64": img_base64})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run the API server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
