from fastapi import FastAPI, UploadFile, File, HTTPException
from predictor import predict_from_image

app = FastAPI(title="Grass Growth Prediction API")

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()

    try:
        result = predict_from_image(image_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
