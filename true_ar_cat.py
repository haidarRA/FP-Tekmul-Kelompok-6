#!/usr/bin/env python3
"""
True AR Cat Breed Classification
AR sejati yang mendeteksi kucing, melacak posisinya, dan menempatkan overlay yang mengikuti
"""

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import time
import os
import json

class TrueARCatClassifier:
    def __init__(self, model_path='best_cat_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Device: {self.device}")
        
        # Load breed info dan model
        self._load_breed_info()
        self.model = self._load_model(model_path)
        
        # Setup transforms untuk klasifikasi
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Setup object detection untuk kucing
        self._setup_object_detection()
        
        # AR tracking variables
        self.tracked_cats = {}  # Dictionary untuk tracking multiple cats
        self.next_cat_id = 0
        self.frame_count = 0
        
        # AR overlay settings
        self.confidence_threshold = 0.3
        self.detection_confidence = 0.5
        
        # Database informasi breed kucing
        self.breed_info = self._create_breed_database()
        
    def _load_breed_info(self):
        """Load breed information"""
        try:
            if os.path.exists('final_cat_model.pth'):
                checkpoint = torch.load('final_cat_model.pth', map_location=self.device)
                self.cat_breeds = [checkpoint['idx_to_breed'][i] for i in range(len(checkpoint['idx_to_breed']))]
                self.num_classes = checkpoint['num_classes']
                print(f"✅ Loaded {self.num_classes} breeds")
            else:
                # Fallback
                self.cat_breeds = sorted([
                    'abyssinian', 'american_bobtail', 'american_curl', 'american_shorthair', 
                    'american_wirehair', 'balinese', 'bengal', 'birman', 'bombay', 
                    'british_shorthair', 'burmese', 'chartreux', 'chausie', 'cornish_rex', 
                    'cymric', 'cyprus', 'devon_rex', 'donskoy', 'egyptian_mau', 
                    'european_shorthair', 'exotic_shorthair', 'german_rex', 'havana_brown', 
                    'himalayan', 'japanese_bobtail', 'karelian_bobtail', 'khao_manee', 
                    'korat', 'korean_bobtail', 'kurilian_bobtail', 'laperm', 'lykoi', 
                    'maine_coon', 'manx', 'mekong_bobtail', 'munchkin', 'nebelung', 
                    'norwegian_forest_cat', 'ocicat', 'oregon_rex', 'oriental_shorthair', 
                    'persian', 'peterbald', 'pixie_bob', 'ragamuffin', 'ragdoll', 
                    'russian_blue', 'safari', 'savannah', 'scottish_fold', 'selkirk_rex', 
                    'serengeti', 'siamese', 'siberian', 'singapura', 'sokoke', 'somali', 
                    'sphynx', 'thai', 'tonkinese', 'toyger', 'turkish_angora', 
                    'turkish_van', 'ukrainian_levkoy', 'ural_rex', 'vankedisi'
                ])
                self.num_classes = len(self.cat_breeds)
                print(f"⚠️ Using fallback: {self.num_classes} breeds")
        except Exception as e:
            print(f"❌ Error loading breed info: {e}")
    
    def _create_breed_database(self):
        """Load breed database dari file JSON"""
        try:
            # Coba load dari file JSON
            if os.path.exists('breed_database.json'):
                with open('breed_database.json', 'r', encoding='utf-8') as f:
                    breed_data = json.load(f)
                print(f"✅ Loaded breed database from JSON: {len(breed_data)} breeds")
                return breed_data
            else:
                print("⚠️ breed_database.json not found, using fallback database")
                # Fallback database untuk breed yang paling umum
                return {
                    'persian': {
                        'description': 'Kucing berbulu panjang dengan wajah datar dan mata besar. Berasal dari Iran, dikenal sangat tenang dan penyayang.',
                        'temperament': 'Tenang, Manis, Penyayang',
                        'origin': 'Iran (Persia)',
                        'lifespan': '12-17 tahun'
                    },
                    'siamese': {
                        'description': 'Kucing aktif dengan mata biru menawan dan pola colorpoint. Sangat vokal dan suka berinteraksi dengan manusia.',
                        'temperament': 'Aktif, Vokal, Sosial',
                        'origin': 'Thailand',
                        'lifespan': '12-20 tahun'
                    },
                    'maine_coon': {
                        'description': 'Salah satu kucing terbesar dengan bulu tebal dan ekor berbulu. Dikenal ramah dan suka bermain.',
                        'temperament': 'Ramah, Cerdas, Suka Bermain',
                        'origin': 'Amerika Serikat',
                        'lifespan': '12-15 tahun'
                    },
                    'british_shorthair': {
                        'description': 'Kucing dengan tubuh bulat dan bulu pendek padat. Memiliki pipi chubby yang menggemaskan.',
                        'temperament': 'Tenang, Mandiri, Penyayang',
                        'origin': 'Inggris',
                        'lifespan': '12-20 tahun'
                    },
                    'sphynx': {
                        'description': 'Kucing tanpa bulu dengan kulit hangat. Sangat energik dan suka menjadi pusat perhatian.',
                        'temperament': 'Energik, Sosial, Suka Perhatian',
                        'origin': 'Kanada',
                        'lifespan': '13-15 tahun'
                    },
                    'ragdoll': {
                        'description': 'Kucing besar dengan bulu semi-panjang dan mata biru. Terkenal karena sifatnya yang sangat jinak.',
                        'temperament': 'Jinak, Tenang, Penyayang',
                        'origin': 'Amerika Serikat',
                        'lifespan': '12-15 tahun'
                    },
                    'bengal': {
                        'description': 'Kucing dengan pola spotted seperti macan tutul. Sangat aktif dan atletis.',
                        'temperament': 'Aktif, Atletis, Cerdas',
                        'origin': 'Amerika Serikat',
                        'lifespan': '13-16 tahun'
                    },
                    'munchkin': {
                        'description': 'Kucing dengan kaki pendek yang unik. Meskipun kaki pendek, mereka sangat lincah dan playful.',
                        'temperament': 'Playful, Outgoing, Cerdas',
                        'origin': 'Amerika Serikat',
                        'lifespan': '12-15 tahun'
                    },
                    'scottish_fold': {
                        'description': 'Kucing dengan telinga yang terlipat ke depan. Memiliki wajah bulat dan mata besar yang ekspresif.',
                        'temperament': 'Tenang, Manis, Penyayang',
                        'origin': 'Skotlandia',
                        'lifespan': '11-15 tahun'
                    },
                    'american_shorthair': {
                        'description': 'Kucing kekar dengan bulu pendek yang mudah dirawat. Kucing keluarga yang ideal.',
                        'temperament': 'Ramah, Easy-going, Sehat',
                        'origin': 'Amerika Serikat',
                        'lifespan': '13-17 tahun'
                    },
                    'russian_blue': {
                        'description': 'Kucing dengan bulu abu-abu biru yang indah dan mata hijau zamrud. Sifatnya pemalu tapi setia.',
                        'temperament': 'Pemalu, Setia, Cerdas',
                        'origin': 'Rusia',
                        'lifespan': '15-18 tahun'
                    },
                    'abyssinian': {
                        'description': 'Kucing aktif dengan bulu ticked yang unik. Sangat curious dan suka memanjat.',
                        'temperament': 'Aktif, Curious, Atletis',
                        'origin': 'Ethiopia',
                        'lifespan': '12-15 tahun'
                    }
                }
        except Exception as e:
            print(f"❌ Error loading breed database: {e}")
            # Return minimal fallback jika terjadi error
            return {
                'unknown': {
                    'description': 'Breed kucing yang terdeteksi',
                    'temperament': 'Bervariasi',
                    'origin': 'Tidak diketahui',
                    'lifespan': '12-15 tahun'
                }
            }
    
    def _load_model(self, model_path):
        """Load classification model"""
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        
        try:
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                print(f"✅ Classification model loaded: {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _setup_object_detection(self):
        """Setup YOLOv8 untuk deteksi kucing"""
        try:
            # Gunakan YOLOv8 terbaru dari ultralytics
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8s.pt')  # YOLOv8 small model
            print("✅ YOLOv8 model loaded untuk object detection")
        except Exception as e:
            print(f"❌ Error loading YOLOv8: {e}")
            print("⚠️ Menggunakan fallback detection method")
            self.yolo_model = None
    
    def detect_cats(self, frame):
        """Deteksi kucing dalam frame menggunakan YOLOv8"""
        if self.yolo_model is None:
            # Fallback: anggap seluruh frame adalah kucing (untuk demo)
            h, w = frame.shape[:2]
            return [{'bbox': [w//4, h//4, w//2, h//2], 'confidence': 0.8}]
        
        try:
            # YOLOv8 detection
            results = self.yolo_model(frame, verbose=False)  # verbose=False untuk mengurangi output
            detections = []
            
            # Extract detections from YOLOv8 results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class ID and confidence
                        cls_id = int(box.cls.cpu().numpy()[0])
                        confidence = float(box.conf.cpu().numpy()[0])
                        
                        # Filter hanya kucing (class 15 dalam COCO dataset)
                        if cls_id == 15 and confidence > self.detection_confidence:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            detections.append({
                                'bbox': [x1, y1, x2-x1, y2-y1],  # [x, y, width, height]
                                'confidence': confidence
                            })
            
            return detections
        
        except Exception as e:
            print(f"Error in cat detection: {e}")
            return []
    
    def classify_cat_region(self, frame, bbox):
        """Klasifikasi breed dari region kucing yang terdeteksi"""
        try:
            x, y, w, h = bbox
            
            # Extract region kucing
            cat_region = frame[y:y+h, x:x+w]
            if cat_region.size == 0:
                return []
            
            # Convert ke RGB dan transform
            rgb_region = cv2.cvtColor(cat_region, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_region)
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probabilities, 3)
                
                predictions = []
                for i in range(3):
                    breed = self.cat_breeds[top_indices[0][i].item()]
                    prob = top_probs[0][i].item()
                    predictions.append((breed, prob))
                
                return predictions
        
        except Exception as e:
            print(f"Error in classification: {e}")
            return []
    
    def update_tracking(self, detections):
        """Update tracking untuk multiple cats"""
        self.frame_count += 1
        
        # Simple tracking berdasarkan proximity
        updated_cats = {}
        
        for detection in detections:
            bbox = detection['bbox']
            det_center = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
            
            # Cari cat yang paling dekat
            best_match = None
            min_distance = float('inf')
            
            for cat_id, cat_data in self.tracked_cats.items():
                if self.frame_count - cat_data['last_seen'] > 30:  # Expired
                    continue
                    
                cat_center = cat_data['center']
                distance = np.sqrt((det_center[0] - cat_center[0])**2 + 
                                 (det_center[1] - cat_center[1])**2)
                
                if distance < min_distance and distance < 100:  # Threshold proximity
                    min_distance = distance
                    best_match = cat_id
            
            if best_match is not None:
                # Update existing cat
                updated_cats[best_match] = {
                    'bbox': bbox,
                    'center': det_center,
                    'confidence': detection['confidence'],
                    'last_seen': self.frame_count,
                    'predictions': self.tracked_cats[best_match].get('predictions', []),
                    'stable_prediction': self.tracked_cats[best_match].get('stable_prediction', None)
                }
            else:
                # New cat
                updated_cats[self.next_cat_id] = {
                    'bbox': bbox,
                    'center': det_center,
                    'confidence': detection['confidence'],
                    'last_seen': self.frame_count,
                    'predictions': [],
                    'stable_prediction': None
                }
                self.next_cat_id += 1
        
        self.tracked_cats = updated_cats
    
    def draw_ar_overlay(self, frame, cat_id, cat_data):
        """Gambar AR overlay yang mengikuti kucing"""
        bbox = cat_data['bbox']
        x, y, w, h = bbox
        
        # Klasifikasi breed untuk cat ini (setiap beberapa frame untuk efisiensi)
        if self.frame_count % 10 == 0 or not cat_data['predictions']:
            predictions = self.classify_cat_region(frame, bbox)
            cat_data['predictions'] = predictions
            
            # Update stable prediction jika confidence tinggi
            if predictions and predictions[0][1] > 0.7:
                cat_data['stable_prediction'] = predictions[0]
        
        predictions = cat_data['predictions']
        stable_pred = cat_data['stable_prediction']
        
        # Gambar bounding box yang mengikuti kucing
        color = (0, 255, 0) if predictions and predictions[0][1] > 0.7 else (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # AR Tag di atas kucing
        if predictions:
            breed, confidence = predictions[0]
            formatted_breed = breed.replace('_', ' ').title()
            
            # Posisi tag mengikuti kucing
            tag_x = x + w//2 - 100
            tag_y = max(20, y - 60)  # Di atas kucing, tapi tidak keluar frame
            
            # Background tag dengan transparansi
            overlay = frame.copy()
            cv2.rectangle(overlay, (tag_x, tag_y), (tag_x + 200, tag_y + 50), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Border tag
            cv2.rectangle(frame, (tag_x, tag_y), (tag_x + 200, tag_y + 50), 
                         color, 2)
            
            # Text breed
            cv2.putText(frame, formatted_breed, (tag_x + 10, tag_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Confidence bar
            conf_text = f"{confidence:.0%}"
            cv2.putText(frame, conf_text, (tag_x + 10, tag_y + 42), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Confidence indicator
            bar_width = int(180 * confidence)
            cv2.rectangle(frame, (tag_x + 10, tag_y + 47), 
                         (tag_x + 10 + bar_width, tag_y + 49), color, -1)
        
        # ID tracking (untuk debugging)
        cv2.putText(frame, f"Cat #{cat_id}", (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Breed information panel (jika ada stable prediction)
        if stable_pred:
            stable_breed = stable_pred[0]
            self._draw_breed_info_panel(frame, stable_breed, x, y + h + 20)
    
    def _draw_breed_info_panel(self, frame, breed_name, start_x, start_y):
        """Gambar panel informasi breed yang lengkap"""
        # Cek apakah breed ada di database
        if breed_name not in self.breed_info:
            # Fallback untuk breed yang tidak ada di database
            formatted_breed = breed_name.replace('_', ' ').title()
            cv2.putText(frame, f"Breed: {formatted_breed}", (start_x, start_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            return
        
        breed_data = self.breed_info[breed_name]
        height, width = frame.shape[:2]
        
        # Tentukan ukuran panel
        panel_width = min(500, width - start_x - 20)
        panel_height = 120
        
        # Pastikan panel tidak keluar dari frame
        panel_x = max(10, min(start_x, width - panel_width - 10))
        panel_y = max(10, min(start_y, height - panel_height - 10))
        
        # Background panel dengan transparansi
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Border panel
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 255, 0), 2)
        
        # Header - nama breed
        formatted_breed = breed_name.replace('_', ' ').title()
        cv2.putText(frame, formatted_breed, (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Description (text utama)
        description = breed_data['description']
        # Break text menjadi beberapa baris jika terlalu panjang
        max_chars_per_line = 65
        words = description.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= max_chars_per_line:
                current_line += " " + word if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        # Tampilkan deskripsi (maksimal 2 baris)
        y_offset = 45
        for i, line in enumerate(lines[:2]):
            cv2.putText(frame, line, (panel_x + 10, panel_y + y_offset + (i * 18)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Info tambahan di bawah
        info_y = panel_y + 85
        info_text = f"Asal: {breed_data['origin']} | Sifat: {breed_data['temperament']}"
        cv2.putText(frame, info_text, (panel_x + 10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Lifespan info
        lifespan_text = f"Umur hidup: {breed_data['lifespan']}"
        cv2.putText(frame, lifespan_text, (panel_x + 10, info_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    def process_frame_for_stream(self, frame):
        """
        Menerima satu frame, melakukan deteksi dan tracking, menggambar overlay AR,
        dan mengembalikan frame yang sudah diproses.
        """
        frame = cv2.flip(frame, 1)  # Efek cermin

        # Deteksi kucing di dalam frame
        detections = self.detect_cats(frame)
        
        # Update data tracking
        self.update_tracking(detections)
        
        # Gambar overlay AR untuk setiap kucing yang terlacak
        active_cats_count = 0
        for cat_id, cat_data in self.tracked_cats.items():
            if self.frame_count - cat_data['last_seen'] <= 5:  # Hanya tampilkan yang baru terlihat
                self.draw_ar_overlay(frame, cat_id, cat_data)
                active_cats_count += 1
        
        # Info status di pojok atas
        cv2.putText(frame, f"Cats Tracked: {active_cats_count}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "True AR Stream", 
                   (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (0, 255, 255), 2, cv2.LINE_AA)
        
        return frame
    
    def run_true_ar(self, camera_index=0):
        """Jalankan True AR Cat Classification"""
        print("🚀 Starting True AR Cat Classification...")
        print("Features:")
        print("- Real object detection (YOLO)")
        print("- Spatial tracking of cats")
        print("- AR overlays that follow cat movement")
        print("- Multi-cat support")
        print()
        print("Controls:")
        print("- Q: Quit")
        print("- S: Screenshot")
        print("- R: Reset tracking")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        screenshot_counter = 0
        fps_start = time.time()
        fps_counter = 0
        
        print("✅ Starting AR loop...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror effect
            
            # Detect cats in frame
            detections = self.detect_cats(frame)
            
            # Update tracking
            self.update_tracking(detections)
            
            # Draw AR overlays for each tracked cat
            for cat_id, cat_data in self.tracked_cats.items():
                if self.frame_count - cat_data['last_seen'] <= 5:  # Recent cats only
                    self.draw_ar_overlay(frame, cat_id, cat_data)
            
            # Status info
            cv2.putText(frame, f"Cats detected: {len(self.tracked_cats)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, "True AR Cat Classification", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 255), 1)
            
            # FPS counter
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                fps = fps_counter / (time.time() - fps_start)
                fps_counter = 0
                fps_start = time.time()
                
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('True AR Cat Classification', frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"true_ar_screenshot_{screenshot_counter:03d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot: {filename}")
                screenshot_counter += 1
            elif key == ord('r'):
                self.tracked_cats = {}
                self.next_cat_id = 0
                print("🔄 Tracking reset")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ True AR completed")

def main():
    print("🐱 True AR Cat Breed Classification 🐱")
    print("=" * 50)
    print()
    print("Fitur AR Sejati:")
    print("✅ Object detection dengan YOLO")
    print("✅ Spatial tracking kucing")
    print("✅ AR overlay mengikuti gerakan kucing")
    print("✅ Multi-cat support")
    print("✅ Stable prediction system")
    print()
    
    try:
        ar_system = TrueARCatClassifier()
        
        print("Ready to start!")
        input("Press Enter to begin...")
        
        ar_system.run_true_ar()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 