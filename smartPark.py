import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from modular import SmartParkDetector  # Ensure SmartParkDetector is in modular.py

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = SmartParkDetector()

# --- HELPER: VIDEO GENERATOR ---
def gen_frames():
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)  # Use 0 for local webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Process frame with your existing logic
            processed_frame, detected_text = detector.process_frame(frame, mode="advanced")
            
            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- ROUTE 1: LIVE CAMERA STREAM ---
@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# --- ROUTE 2: UPLOAD IMAGE ---
@app.post("/detect_upload")
async def detect_plate(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process the uploaded image
    _, detected_text = detector.process_frame(frame, mode="advanced")
    
    return {"plate_number": detected_text if detected_text else "Not Detected"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)