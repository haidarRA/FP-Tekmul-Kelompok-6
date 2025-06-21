#!/usr/bin/env python3
"""
Comprehensive Cat Breed Training
Training optimized berdasarkan dataset_stats.csv analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import pandas as pd
import numpy as np
import json
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

class CatBreedDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except:
            # Return next image if error
            return self.__getitem__((idx + 1) % len(self.image_paths))

class CatBreedTrainer:
    def __init__(self):
        # Enhanced GPU detection
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 Training device: {self.device}")
            print(f"🎮 GPU: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
            
            # Optimize for RTX 3060
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        else:
            self.device = torch.device('cpu')
            print(f"⚠️  Training device: {self.device} (GPU not available)")
        
        # Load dataset stats untuk strategy
        self.dataset_stats = pd.read_csv('dataset_stats.csv')
        self.setup_training_strategy()
        
        # Load data paths dan labels
        self.load_dataset_paths()
        
        # Setup model
        self.setup_model()
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
    def setup_training_strategy(self):
        """Setup strategi training berdasarkan dataset_stats.csv"""
        print("📊 Setting up training strategy from dataset_stats.csv...")
        
        # Kategorikan breeds berdasarkan jumlah data
        self.low_data_breeds = self.dataset_stats[
            self.dataset_stats['image_count'] < 140]['class'].tolist()
        self.high_data_breeds = self.dataset_stats[
            self.dataset_stats['image_count'] >= 190]['class'].tolist()
        self.balanced_breeds = self.dataset_stats[
            (self.dataset_stats['image_count'] >= 140) & 
            (self.dataset_stats['image_count'] < 190)]['class'].tolist()
        
        print(f"  📉 Low data breeds: {len(self.low_data_breeds)}")
        print(f"  📊 Balanced breeds: {len(self.balanced_breeds)}")
        print(f"  📈 High data breeds: {len(self.high_data_breeds)}")
        
        # Calculate class weights (inverse frequency)
        total_images = self.dataset_stats['image_count'].sum()
        num_classes = len(self.dataset_stats)
        self.class_weights = {}
        
        for _, row in self.dataset_stats.iterrows():
            breed = row['class']
            count = row['image_count']
            weight = total_images / (num_classes * count)
            self.class_weights[breed] = weight
            
        print(f"  ⚖️  Class weights range: {min(self.class_weights.values()):.3f} - {max(self.class_weights.values()):.3f}")
        
    def get_augmentation_transform(self, breed_category):
        """Get transform berdasarkan kategori breed"""
        base_transforms = [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224)
        ]
        
        if breed_category == 'low_data':
            # Heavy augmentation untuk breed dengan data sedikit
            aug_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ]
        elif breed_category == 'balanced':
            # Normal augmentation
            aug_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        else:  # high_data
            # Light augmentation
            aug_transforms = [
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        
        final_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        return transforms.Compose(base_transforms + aug_transforms + final_transforms)
    
    def load_dataset_paths(self):
        """Load all image paths and labels from img-cat-breeds"""
        print("📁 Loading dataset paths...")
        
        self.all_paths = []
        self.all_labels = []
        self.breed_to_idx = {}
        self.idx_to_breed = {}
        
        breeds = sorted([d for d in os.listdir('img-cat-breeds') 
                        if os.path.isdir(os.path.join('img-cat-breeds', d))])
        
        for idx, breed in enumerate(breeds):
            self.breed_to_idx[breed] = idx
            self.idx_to_breed[idx] = breed
            
            breed_path = os.path.join('img-cat-breeds', breed)
            images = [f for f in os.listdir(breed_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img in images:
                img_path = os.path.join(breed_path, img)
                self.all_paths.append(img_path)
                self.all_labels.append(idx)
        
        self.num_classes = len(breeds)
        print(f"  📸 Total images: {len(self.all_paths)}")
        print(f"  🏷️  Total classes: {self.num_classes}")
        
        # Stratified split menggunakan 80-20
        self.train_paths, self.val_paths, self.train_labels, self.val_labels = train_test_split(
            self.all_paths, self.all_labels, test_size=0.2, 
            stratify=self.all_labels, random_state=42
        )
        
        print(f"  🚂 Train samples: {len(self.train_paths)}")
        print(f"  ✅ Val samples: {len(self.val_paths)}")
        
    def create_data_loaders(self):
        """Create data loaders dengan balanced sampling"""
        print("🔄 Creating balanced data loaders...")
        
        # Training transform - mixed strategy berdasarkan breed
        train_dataset = CatBreedDataset(
            self.train_paths, self.train_labels,
            transform=self.get_augmentation_transform('balanced')  # Default untuk semua
        )
        
        # Validation transform
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_dataset = CatBreedDataset(
            self.val_paths, self.val_labels, transform=val_transform
        )
        
        # Weighted sampler untuk balanced training
        train_weights = []
        for label in self.train_labels:
            breed = self.idx_to_breed[label]
            train_weights.append(self.class_weights[breed])
        
        sampler = WeightedRandomSampler(
            weights=train_weights, num_samples=len(train_weights), replacement=True
        )
        
        # Optimize batch size and num_workers for RTX 3060
        batch_size = 32 if torch.cuda.is_available() else 16
        num_workers = 4 if torch.cuda.is_available() else 2
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler, 
            num_workers=num_workers, pin_memory=True, persistent_workers=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True, persistent_workers=True
        )
        
        print(f"  ✅ Data loaders ready!")
        
    def setup_model(self):
        """Setup ResNet50 model"""
        print("🤖 Setting up model...")
        
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        
        # Try loading existing features
        if os.path.exists('resnet50-model-augmentation.pth'):
            try:
                print("  📥 Loading existing model features...")
                state_dict = torch.load('resnet50-model-augmentation.pth', map_location=self.device)
                
                # Load only CNN features (not FC layer)
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items() 
                                 if k in model_dict and 'fc' not in k}
                
                if pretrained_dict:
                    model_dict.update(pretrained_dict)
                    self.model.load_state_dict(model_dict)
                    print(f"  ✅ Loaded {len(pretrained_dict)} pretrained layers")
            except Exception as e:
                print(f"  ⚠️  Could not load existing model: {e}")
        
        self.model.to(self.device)
        
        # Setup loss with class weights
        weight_tensor = torch.FloatTensor([
            self.class_weights[self.idx_to_breed[i]] for i in range(self.num_classes)
        ]).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        
        # Optimizer dengan different learning rates
        fc_params = [p for n, p in self.model.named_parameters() if 'fc' in n]
        other_params = [p for n, p in self.model.named_parameters() if 'fc' not in n]
        
        self.optimizer = optim.Adam([
            {'params': other_params, 'lr': 0.0001},  # Lower LR for pretrained
            {'params': fc_params, 'lr': 0.001}       # Higher LR for new FC
        ])
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        print(f"  ✅ Model ready with {sum(p.numel() for p in self.model.parameters())} parameters")
        
    def train_epoch(self):
        """Train one epoch with progress bar"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar untuk training
        train_pbar = tqdm(self.train_loader, desc="  🚂 Training", 
                         bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar dengan informasi real-time
            current_acc = 100.0 * correct / total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate one epoch with progress bar"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar untuk validation
        val_pbar = tqdm(self.val_loader, desc="  ✅ Validation", 
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar dengan informasi real-time
                current_acc = 100.0 * correct / total
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc
    
    def train(self, epochs=20):
        """Main training loop"""
        print(f"\n🚀 Starting training for {epochs} epochs...")
        print("=" * 60)
        
        # Create data loaders
        self.create_data_loaders()
        
        best_val_acc = 0.0
        patience_counter = 0
        patience = 8
        
        start_time = time.time()
        
        # Overall progress bar untuk semua epochs
        epoch_pbar = tqdm(range(epochs), desc="🚀 Overall Progress", 
                         bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} epochs [{elapsed}<{remaining}]')
        
        for epoch in epoch_pbar:
            print(f'\n📅 Epoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate_epoch()
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            print(f'\n📊 Epoch {epoch+1} Summary:')
            print(f'  🚂 Train: Loss {train_loss:.4f}, Acc {train_acc:.2f}%')
            print(f'  ✅ Val:   Loss {val_loss:.4f}, Acc {val_acc:.2f}%')
            print(f'  ⏱️  Time: {epoch_time:.1f}s')
            
            # Update overall progress bar
            epoch_pbar.set_postfix({
                'Best_Val_Acc': f'{best_val_acc:.2f}%',
                'Current_Val_Acc': f'{val_acc:.2f}%'
            })
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_cat_model.pth')
                print(f'  ✨ New best model saved! Acc: {val_acc:.2f}%')
                patience_counter = 0
            else:
                patience_counter += 1
                print(f'  ⏳ Patience: {patience_counter}/{patience}')
            
            # Early stopping
            if patience_counter >= patience:
                print(f'\n⏰ Early stopping after {patience} epochs without improvement')
                break
        
        # Save final model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'breed_to_idx': self.breed_to_idx,
            'idx_to_breed': self.idx_to_breed,
            'num_classes': self.num_classes,
            'best_val_acc': best_val_acc
        }, 'final_cat_model.pth')
        
        total_time = time.time() - start_time
        print(f'\n🎉 Training completed!')
        print(f'⏱️  Total time: {total_time//60:.0f}m {total_time%60:.0f}s')
        print(f'🎯 Best validation accuracy: {best_val_acc:.2f}%')
        
        # Plot history
        self.plot_history()
        
        return best_val_acc
    
    def plot_history(self):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Acc')
        plt.plot(self.history['val_acc'], label='Val Acc')
        plt.title('Training & Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=200, bbox_inches='tight')
        plt.show()
        print(f'📊 Training history saved as training_history.png')

def quick_test():
    """Quick test untuk model yang sudah dilatih"""
    print("\n🧪 Quick Test Trained Model")
    print("=" * 40)
    
    if not os.path.exists('best_cat_model.pth'):
        print("❌ best_cat_model.pth not found. Please train first.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model info
    checkpoint = torch.load('final_cat_model.pth', map_location=device)
    breed_to_idx = checkpoint['breed_to_idx']
    idx_to_breed = checkpoint['idx_to_breed']
    num_classes = checkpoint['num_classes']
    
    # Setup model
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load('best_cat_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    # Test transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Test beberapa breed
    test_breeds = ['persian', 'siamese', 'maine_coon', 'british_shorthair', 'sphynx']
    correct = 0
    total = 0
    
    for breed in test_breeds:
        breed_path = f'img-cat-breeds/{breed}'
        if os.path.exists(breed_path):
            images = [f for f in os.listdir(breed_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Test 3 random images
            import random
            test_images = random.sample(images, min(3, len(images)))
            
            for img_file in test_images:
                img_path = os.path.join(breed_path, img_file)
                
                try:
                    image = Image.open(img_path).convert('RGB')
                    input_tensor = transform(image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted_idx = torch.max(probabilities, 1)
                        
                        predicted_breed = idx_to_breed[predicted_idx.item()]
                        confidence_score = confidence.item()
                        
                        is_correct = predicted_breed == breed
                        if is_correct:
                            correct += 1
                        total += 1
                        
                        status = "✅" if is_correct else "❌"
                        print(f"{status} {img_file}: {breed} -> {predicted_breed} ({confidence_score:.3f})")
                        
                except Exception as e:
                    print(f"Error: {e}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n🎯 Quick test accuracy: {accuracy:.3f} ({correct}/{total})")

def main():
    print("🐱 Cat Breed Comprehensive Training 🐱")
    print("=" * 50)
    
    if not os.path.exists('dataset_stats.csv'):
        print("❌ dataset_stats.csv not found!")
        return
    
    if not os.path.exists('img-cat-breeds'):
        print("❌ img-cat-breeds directory not found!")
        return
    
    while True:
        print("\nPilihan:")
        print("1. 🚀 Start training (20 epochs)")
        print("2. 🧪 Quick test existing model")
        print("3. 📊 Show dataset stats info")
        print("4. 🚪 Exit")
        
        choice = input("\nPilih (1-4): ").strip()
        
        if choice == '1':
            trainer = CatBreedTrainer()
            best_acc = trainer.train(epochs=20)
            print(f"\n🎉 Training selesai dengan akurasi terbaik: {best_acc:.2f}%")
            
        elif choice == '2':
            quick_test()
            
        elif choice == '3':
            df = pd.read_csv('dataset_stats.csv')
            print(f"\n📊 Dataset Statistics:")
            print(f"Total breeds: {len(df)}")
            print(f"Total images: {df['image_count'].sum()}")
            print(f"Min images per breed: {df['image_count'].min()}")
            print(f"Max images per breed: {df['image_count'].max()}")
            print(f"Average images per breed: {df['image_count'].mean():.1f}")
            
        elif choice == '4':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Pilihan tidak valid!")

if __name__ == "__main__":
    main() 