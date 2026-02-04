import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
import re
from difflib import SequenceMatcher
import os
from tkinter import filedialog, Tk

# --- CONFIGURATION ---
YOLO_MODEL_PATH = "best.pt"

class SmartParkDetector:
    def __init__(self):
        print("[INFO] Loading YOLO Model...")
        self.model = YOLO(YOLO_MODEL_PATH)
        print("[INFO] Loading PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def normalize_region(self, text):
        """Levenshtein/Fuzzy matching for Pakistan regions."""
        text = text.upper().replace(" ", "").replace("-", "").strip()
        valid_cities = ["ISLAMABAD", "PUNJAB", "SINDH", "KPK", "BALOCHISTAN", "GILGIT", "AJK"]
        
        best_match = text
        highest_ratio = 0.0
        for city in valid_cities:
            ratio = SequenceMatcher(None, text, city).ratio()
            if ratio > highest_ratio:
                highest_ratio, best_match = ratio, city
        
        return best_match if highest_ratio >= 0.6 else text

    def deskew_plate(self, image):
        """Perspective Correction to fix tilted plates."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 50, 200)
        
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]

                (tl, tr, br, bl) = rect
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))
                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))

                dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                return cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return image

    def format_plate_text(self, ocr_text):
        raw = ocr_text.upper().replace(" ", "").replace("-", "")
        patterns = [
            (r"^([A-Z]+)(\d+)([A-Z]+)$", lambda m: f"{m[0]} {m[1]} {self.normalize_region(m[2])}"),
            (r"^([A-Z]+)([A-Z]+)(\d+)$", lambda m: f"{m[1]} {m[2]} {self.normalize_region(m[0])}"),
            (r"^([A-Z]+)(\d+)$", lambda m: f"{m[0]} {m[1]}")
        ]
        for pattern, fmt in patterns:
            match = re.match(pattern, raw)
            if match:
                try: return fmt(match.groups())
                except: continue
        return self.normalize_region(raw)

    def preprocess_image(self, crop, mode="simple"):
        """Advanced Preprocessing: Denoising, CLAHE, and Adaptive Thresholding."""
        if mode == "advanced":
            # 1. Perspective Correction
            crop = self.deskew_plate(crop)
            # 2. Denoising
            crop = cv2.fastNlMeansDenoisingColored(crop, None, 10, 10, 7, 21)
            # 3. Contrast Enhancement (CLAHE)
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            crop = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return crop
            
        elif mode == "adaptive":
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        return crop

    def process_frame(self, frame, mode="simple"):
        results = self.model.predict(source=frame, conf=0.5, device='cpu', verbose=False)
        detected_text = None

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            for x1, y1, x2, y2 in boxes:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                #cv2.imshow("1. Original Plate Crop", crop)

                processed_crop = self.preprocess_image(crop, mode=mode)
                result = self.ocr.predict(processed_crop)
                
                if result and len(result) > 0:
                    res_dict = result[0]
                    texts=res_dict.get('rec_texts',[])

                    scores=res_dict.get('rec_scores',[])
                    boxes=res_dict.get('rec_boxes',[])
                    combined=zip(texts,scores,boxes)
                    print("text",texts)
                    print("scores",scores)
                    validSegments=[t for t,s,b in combined if s>0.9]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    if validSegments:
                        formattedText=" ".join(validSegments)
                        
                        cv2.putText(frame, formattedText, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        detected_text = formattedText
        return frame, detected_text

def get_user_choices():
    root = Tk()
    root.withdraw()
    print("\n" + "="*30 + "\n   SmartPark Main Menu\n" + "="*30)
    print("1) Live Camera\n2) Select File\n3) Select Folder\n4) Exit")
    source = input("\nEnter choice: ")
    if source == '4': return '4', None, None

    path = filedialog.askopenfilename() if source == '2' else filedialog.askdirectory() if source == '3' else None
    print("\n--- OCR Preprocessing Mode ---\n1) Simple\n2) Adaptive\n3) Advanced (Deskew + CLAHE)")
    m = input("Enter choice: ")
    mode = "advanced" if m == '3' else "adaptive" if m == '2' else "simple"
    root.destroy()
    return source, path, mode

if __name__ == "__main__":
    detector = SmartParkDetector()
    while True:
        src, path, mode = get_user_choices()
        if src == '4': break

        if src == '1':
            cap = cv2.VideoCapture(0)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame, _ = detector.process_frame(frame, mode=mode)
                cv2.imshow("SmartPark LPR", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            cap.release()
        elif src == '2' and path:
            img, _ = detector.process_frame(cv2.imread(path), mode=mode)
            cv2.imshow("SmartPark LPR", img); cv2.waitKey(0)
        elif src == '3' and path:
            for f in os.listdir(path):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img, _ = detector.process_frame(cv2.imread(os.path.join(path, f)), mode=mode)
                    cv2.imshow("SmartPark LPR", img)
                    cv2.waitKey(1)
                    while True:

                        key = cv2.waitKey(0) & 0xFF
                        if key == 32:  # 32 is the ASCII for Spacebar
                            break      # Break internal loop to show next image
                        if key == ord('q'):
                            cv2.destroyAllWindows()
                            # 'q' dabane se folder loop se bahar nikal jayenge
                            goto_menu = True 
                            break
                
                    if 'goto_menu' in locals() and goto_menu:
                        del goto_menu
                        break
       