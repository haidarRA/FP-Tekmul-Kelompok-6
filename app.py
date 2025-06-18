import os
import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template
import json
import numpy as np
import cv2
from ultralytics import YOLO

from model.model_definition import load_pytorch_model

app = Flask(__name__)

MODEL_DIR = "model"
CLASSIFIER_MODEL_FILE = "resnet50-model-augmentation.pth"
CLASS_NAMES_FILE = "class_names.txt"
DESCRIPTIONS_FILE = "cat_descriptions.json"
YOLO_MODEL_PATH = "yolov8n.pt"

CLASSIFIER_MODEL_PATH = os.path.join(MODEL_DIR, CLASSIFIER_MODEL_FILE)
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, CLASS_NAMES_FILE)
DESCRIPTIONS_PATH = os.path.join(DESCRIPTIONS_FILE)

IMAGE_SIZE_PREPROCESS_RESNET = 256
IMAGE_SIZE_CROP_RESNET = 224
NORM_MEAN_RESNET = [0.485, 0.456, 0.406]
NORM_STD_RESNET = [0.229, 0.224, 0.225]

breed_classifier_model = None
cat_detector_model = None
class_names = []
NUM_CLASSES = 0
cat_descriptions = {}
cat_class_id_coco = 15

try:
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    NUM_CLASSES = len(class_names)

    with open(DESCRIPTIONS_PATH, 'r') as f:
        cat_descriptions = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    breed_classifier_model = load_pytorch_model(CLASSIFIER_MODEL_PATH, NUM_CLASSES, device)
    print(f"Model Klasifikasi Ras '{CLASSIFIER_MODEL_FILE}' berhasil dimuat.")

    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"Model YOLO '{YOLO_MODEL_PATH}' tidak ditemukan.")
    cat_detector_model = YOLO(YOLO_MODEL_PATH)
    print(f"Model Deteksi Kucing YOLO '{YOLO_MODEL_PATH}' berhasil dimuat.")

    print(f"{len(cat_descriptions)} deskripsi ras kucing dimuat.")

except Exception as e:
    print(f"CRITICAL Error loading resources: {e}")
    import traceback
    traceback.print_exc()


resnet_image_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMAGE_SIZE_PREPROCESS_RESNET),
    transforms.CenterCrop(IMAGE_SIZE_CROP_RESNET),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN_RESNET, NORM_STD_RESNET)
])

def preprocess_numpy_roi_for_resnet(numpy_img_rgb_roi):
    """ Preprocesses a NumPy RGB ROI for ResNet50. """
    try:
        return resnet_image_transforms(numpy_img_rgb_roi).unsqueeze(0)
    except Exception as e:
        print(f"Error preprocessing numpy ROI for resnet: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_ar_realtime', methods=['POST'])
def predict_ar_cat_detection_route():
    if breed_classifier_model is None or cat_detector_model is None or not class_names or not cat_descriptions:
         return jsonify({'error': 'Sumber daya server tidak dapat dimuat.', 'predictions': []}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided', 'predictions': []}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected', 'predictions': []}), 400

    predictions_output = []

    if file:
        try:
            image_bytes = file.read()

            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            frame_np_rgb = np.array(pil_image)

            yolo_results = cat_detector_model(frame_np_rgb, verbose=False, conf=0.3)

            if yolo_results and len(yolo_results) > 0:
                result = yolo_results[0]
                boxes = result.boxes
                
                print(f"Debug YOLO: Ditemukan {len(boxes)} objek sebelum filter kelas.")

                for i in range(len(boxes)):
                    detected_class_id = int(boxes.cls[i])
                    confidence_yolo = float(boxes.conf[i])
                    print(f"Debug YOLO: Deteksi obj ke-{i}, kelas: {detected_class_id}, conf: {confidence_yolo:.2f}")

                    if detected_class_id == cat_class_id_coco:
                        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                        
                        if x2 <= x1 or y2 <= y1:
                            print(f"Debug YOLO: Bbox tidak valid [{x1},{y1},{x2},{y2}]")
                            continue
                        
                        print(f"Debug YOLO: Kucing terdeteksi! Bbox: [{x1},{y1},{x2},{y2}]")
                        
                        pad = 10
                        crop_x1 = max(0, x1 - pad)
                        crop_y1 = max(0, y1 - pad)
                        crop_x2 = min(frame_np_rgb.shape[1], x2 + pad)
                        crop_y2 = min(frame_np_rgb.shape[0], y2 + pad)

                        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                            print(f"Debug ResNet: Crop bbox tidak valid setelah padding.")
                            continue
                            
                        cat_roi_np_rgb = frame_np_rgb[crop_y1:crop_y2, crop_x1:crop_x2]

                        if cat_roi_np_rgb.size == 0:
                            print(f"Debug ResNet: ROI kucing kosong.")
                            continue
                        
                        resnet_input_tensor = preprocess_numpy_roi_for_resnet(cat_roi_np_rgb)
                        if resnet_input_tensor is None:
                            print(f"Debug ResNet: Preprocessing ROI gagal.")
                            continue
                        
                        resnet_input_tensor = resnet_input_tensor.to(device)
                        with torch.no_grad():
                            outputs_resnet = breed_classifier_model(resnet_input_tensor)
                            probabilities_resnet = torch.exp(outputs_resnet[0])
                            confidence_resnet, predicted_idx_tensor = torch.max(probabilities_resnet, 0)
                            predicted_idx = predicted_idx_tensor.item()
                            prob_value_resnet = confidence_resnet.item()

                        breed_name = "Unknown"
                        description = "Deskripsi tidak tersedia."
                        if 0 <= predicted_idx < len(class_names):
                            breed_name = class_names[predicted_idx]
                            description = cat_descriptions.get(breed_name, "Deskripsi ras ini tidak ditemukan.")
                        
                        print(f"Debug ResNet: Prediksi Ras: {breed_name}, Prob: {prob_value_resnet:.2f}")
                        
                        predictions_output.append({
                            'bbox_2d': [x1, y1, x2, y2],
                            'breed': breed_name,
                            'probability': prob_value_resnet,
                            'description': description
                        })
            
            if not predictions_output:
                 print("Debug Akhir: Tidak ada kucing yang berhasil diproses hingga akhir.")


            return jsonify({'predictions': predictions_output})

        except Exception as e:
            print(f"Error during AR prediction route: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Server error: {str(e)}', 'predictions': []}), 500

    return jsonify({'error': 'Unknown error or no file provided', 'predictions': []}), 500


if __name__ == '__main__':
    if breed_classifier_model is None or cat_detector_model is None or not class_names or not cat_descriptions:
        print("CRITICAL: Satu atau lebih sumber daya (model/detektor/kelas/deskripsi) tidak dapat dimuat. Flask app mungkin tidak berjalan dengan benar.")
    else:
        try:
            context = ('cert.pem', 'key.pem')
            print("Mencoba menjalankan Flask dengan HTTPS...")
            app.run(host='0.0.0.0', port=5000, debug=True, ssl_context=context)
        except FileNotFoundError:
            print("cert.pem atau key.pem tidak ditemukan. Menjalankan dengan HTTP sebagai fallback.")
            print("WARNING: Akses kamera dari perangkat lain atau di beberapa browser mungkin memerlukan HTTPS.")
            app.run(host='0.0.0.0', port=5000, debug=True)
        except Exception as e_run:
            print(f"Error saat menjalankan Flask server: {e_run}")
            print("Menjalankan dengan HTTP sebagai fallback.")
            app.run(host='0.0.0.0', port=5000, debug=True)