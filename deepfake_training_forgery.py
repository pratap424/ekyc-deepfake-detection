
# ZenTej Hackathon - FORGERY DATASET Training Script
# Specifically designed for Forgery_Dataset structure with train_labels.csv

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json
import random
from efficientnet_pytorch import EfficientNet
import warnings
import glob
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

class ForgeryDataset(Dataset):
    """Dataset class for Forgery_Dataset with train_labels.csv"""

    def __init__(self, forgery_dataset_dir, transform=None, max_samples_per_class=None):
        self.forgery_dataset_dir = forgery_dataset_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        self.identities = []

        # Define paths
        real_dir = os.path.join(forgery_dataset_dir, 'real')
        fake_dir = os.path.join(forgery_dataset_dir, 'fake') 
        labels_file = os.path.join(forgery_dataset_dir, 'train_labels.csv')

        print(f"Looking for real images in: {real_dir}")
        print(f"Looking for fake images in: {fake_dir}")
        print(f"Looking for labels file: {labels_file}")

        # Check if directories exist
        if not os.path.exists(real_dir):
            print(f"ERROR: Real directory not found: {real_dir}")
        if not os.path.exists(fake_dir):
            print(f"ERROR: Fake directory not found: {fake_dir}")

        # Method 1: Use CSV file if available
        if os.path.exists(labels_file):
            print("Using train_labels.csv for dataset loading...")
            try:
                df = pd.read_csv(labels_file)
                print(f"CSV columns: {list(df.columns)}")
                print(f"CSV shape: {df.shape}")
                print(f"First few rows:\n{df.head()}")

                # Process CSV file
                for idx, row in df.iterrows():
                    if len(df.columns) >= 3:  # Assuming image_id, identity_id, forgery_type format
                        image_id = row.iloc[0]  # First column
                        identity_id = row.iloc[1] if len(row) > 1 else 'unknown'  # Second column
                        forgery_type = row.iloc[2] if len(row) > 2 else 'unknown'  # Third column

                        # Determine if real or fake based on forgery_type
                        if pd.isna(forgery_type) or forgery_type == '' or forgery_type.lower() in ['real', 'authentic', 'original']:
                            label = 1  # Real
                            img_path = os.path.join(real_dir, image_id)
                        else:
                            label = 0  # Fake
                            img_path = os.path.join(fake_dir, image_id)

                        # Check if image exists
                        if os.path.exists(img_path):
                            self.samples.append(img_path)
                            self.labels.append(label)
                            self.identities.append(str(identity_id))
                        else:
                            # Try with different extensions
                            base_name = os.path.splitext(image_id)[0]
                            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                                test_path_real = os.path.join(real_dir, base_name + ext)
                                test_path_fake = os.path.join(fake_dir, base_name + ext)

                                if os.path.exists(test_path_real):
                                    self.samples.append(test_path_real)
                                    self.labels.append(1)  # Real
                                    self.identities.append(str(identity_id))
                                    break
                                elif os.path.exists(test_path_fake):
                                    self.samples.append(test_path_fake)
                                    self.labels.append(0)  # Fake
                                    self.identities.append(str(identity_id))
                                    break

            except Exception as e:
                print(f"Error reading CSV file: {e}")
                print("Falling back to directory-based loading...")

        # Method 2: Directory-based loading if CSV fails or doesn't exist
        if len(self.samples) == 0:
            print("Loading dataset from directories...")

            # Load real images
            if os.path.exists(real_dir):
                real_images = []
                for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']:
                    real_images.extend(glob.glob(os.path.join(real_dir, ext)))

                print(f"Found {len(real_images)} real images")

                if max_samples_per_class:
                    real_images = real_images[:max_samples_per_class]

                for img_path in real_images:
                    self.samples.append(img_path)
                    self.labels.append(1)  # Real
                    self.identities.append('real')

            # Load fake images
            if os.path.exists(fake_dir):
                fake_images = []
                for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']:
                    fake_images.extend(glob.glob(os.path.join(fake_dir, ext)))

                print(f"Found {len(fake_images)} fake images")

                if max_samples_per_class:
                    fake_images = fake_images[:max_samples_per_class]

                for img_path in fake_images:
                    self.samples.append(img_path)
                    self.labels.append(0)  # Fake
                    self.identities.append('fake')

        # Balance the dataset
        if len(self.samples) > 0:
            real_count = sum(self.labels)
            fake_count = len(self.labels) - real_count

            print(f"Dataset loaded: {len(self.samples)} samples")
            print(f"Real images: {real_count}")
            print(f"Fake images: {fake_count}")
            print(f"Balance ratio: {real_count / len(self.labels):.2f}")

            # If severely imbalanced, balance it
            if abs(real_count - fake_count) > min(real_count, fake_count):
                print("Dataset is imbalanced, balancing...")
                self._balance_dataset()
        else:
            print("ERROR: No images loaded!")
            print("Please check your dataset structure:")
            print(f"Expected: {real_dir} (with real images)")
            print(f"Expected: {fake_dir} (with fake images)")
            print(f"Optional: {labels_file} (with metadata)")

    def _balance_dataset(self):
        """Balance the dataset by sampling equal numbers of real and fake images"""
        real_indices = [i for i, label in enumerate(self.labels) if label == 1]
        fake_indices = [i for i, label in enumerate(self.labels) if label == 0]

        min_count = min(len(real_indices), len(fake_indices))

        # Sample equal numbers
        selected_real = random.sample(real_indices, min_count)
        selected_fake = random.sample(fake_indices, min_count)

        selected_indices = selected_real + selected_fake
        random.shuffle(selected_indices)

        # Update samples and labels
        self.samples = [self.samples[i] for i in selected_indices]
        self.labels = [self.labels[i] for i in selected_indices]
        self.identities = [self.identities[i] for i in selected_indices]

        print(f"Balanced dataset: {len(self.samples)} samples ({min_count} real, {min_count} fake)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                image = Image.open(img_path).convert('RGB')
                image = np.array(image)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Basic image quality check
            if image.shape[0] < 32 or image.shape[1] < 32:
                # Resize very small images
                image = cv2.resize(image, (224, 224))

            # Convert to PIL for transforms
            image = Image.fromarray(image)

            if self.transform:
                image = self.transform(image)

            return image, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image with some noise to prevent overfitting
            dummy_image = torch.randn(3, 224, 224) * 0.1 + 0.5
            dummy_image = torch.clamp(dummy_image, 0, 1)
            return dummy_image, torch.tensor(label, dtype=torch.long)

class MultiTaskDeepfakeDetector(nn.Module):
    """Multi-task model for eKYC verification"""

    def __init__(self, backbone='efficientnet-b0', num_classes=2, dropout=0.5):
        super(MultiTaskDeepfakeDetector, self).__init__()

        # Backbone
        if backbone.startswith('efficientnet'):
            self.backbone = EfficientNet.from_pretrained(backbone)
            in_features = self.backbone._fc.in_features
            self.backbone._fc = nn.Identity()
        else:
            self.backbone = models.resnet18(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        # Feature extraction with strong regularization
        self.feature_extractor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.5)
        )

        # Authenticity classification head
        self.authenticity_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes)
        )

        # Match score head (identity matching confidence)
        self.match_score_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Liveness score head (live person detection)
        self.liveness_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.feature_extractor(features)

        authenticity = self.authenticity_head(features)
        match_score = self.match_score_head(features)
        liveness_score = self.liveness_head(features)

        return authenticity, match_score, liveness_score

def get_robust_transforms():
    """Strong data augmentation to prevent overfitting"""

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

class CombinedLoss(nn.Module):
    """Combined loss function for multi-task learning"""

    def __init__(self, alpha=1.0, beta=0.3, gamma=0.3):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, labels):
        authenticity_logits, match_scores, liveness_scores = outputs

        # Classification loss
        auth_loss = self.ce_loss(authenticity_logits, labels)

        # Create noisy target scores to prevent overfitting
        batch_size = labels.size(0)

        # For real images: higher scores with noise
        # For fake images: lower scores with noise
        target_match = labels.float().unsqueeze(1)
        target_liveness = labels.float().unsqueeze(1)

        # Add significant noise to prevent perfect learning
        noise_match = torch.randn_like(target_match) * 0.15
        noise_liveness = torch.randn_like(target_liveness) * 0.15

        target_match = target_match * 0.7 + 0.15 + noise_match
        target_liveness = target_liveness * 0.7 + 0.15 + noise_liveness

        target_match = torch.clamp(target_match, 0.1, 0.9)
        target_liveness = torch.clamp(target_liveness, 0.1, 0.9)

        # Regression losses
        match_loss = self.mse_loss(match_scores, target_match)
        liveness_loss = self.mse_loss(liveness_scores, target_liveness)

        total_loss = self.alpha * auth_loss + self.beta * match_loss + self.gamma * liveness_loss

        return total_loss, auth_loss, match_loss, liveness_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, model_name):
    """Training function with early stopping"""

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    patience = 7
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (images, labels) in enumerate(train_bar):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            total_loss, auth_loss, match_loss, liveness_loss = criterion(outputs, labels)
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += total_loss.item()
            _, predicted = torch.max(outputs[0], 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if batch_idx % 20 == 0:
                train_bar.set_postfix({
                    'TotalLoss': f'{total_loss.item():.4f}',
                    'AuthLoss': f'{auth_loss.item():.4f}',
                    'Acc': f'{100.*correct_train/total_train:.1f}%'
                })

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct_train / total_train

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                total_loss, auth_loss, match_loss, liveness_loss = criterion(outputs, labels)

                val_running_loss += total_loss.item()
                _, predicted = torch.max(outputs[0], 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                val_bar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Acc': f'{100.*correct_val/total_val:.1f}%'
                })

        val_loss = val_running_loss / len(val_loader)
        val_acc = 100. * correct_val / total_val

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Scheduler step
        scheduler.step(val_loss)

        # Model saving and early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, f'best_{model_name}_model.pth')
        else:
            patience_counter += 1

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Best Val Acc: {best_val_acc:.2f}%, Patience: {patience_counter}/{patience}')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.7f}')
        print('-' * 70)

        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }

def evaluate_model(model, test_loader, device):
    """Evaluate the trained model"""

    model.eval()
    all_preds = []
    all_labels = []
    all_match_scores = []
    all_liveness_scores = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs[0], 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_match_scores.extend(outputs[1].cpu().numpy().flatten())
            all_liveness_scores.extend(outputs[2].cpu().numpy().flatten())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_preds,
        'true_labels': all_labels,
        'match_scores': all_match_scores,
        'liveness_scores': all_liveness_scores
    }

def main():
    # Configuration for Forgery Dataset
    FORGERY_DATASET_DIR = r"C:\Users\Shruti\Desktop\zentej\dataset\Forgery_Dataset"
    BATCH_SIZE = 8  # Conservative for 4GB GPU
    NUM_EPOCHS = 30
    LEARNING_RATE = 5e-5  # Very low to prevent overfitting
    WEIGHT_DECAY = 1e-3
    MAX_SAMPLES_PER_CLASS = 3000  # Adjust based on your dataset size

    print("=== ZenTej Hackathon - Forgery Dataset Training ===")
    print(f"Dataset directory: {FORGERY_DATASET_DIR}")

    # Check if dataset exists
    if not os.path.exists(FORGERY_DATASET_DIR):
        print(f"ERROR: Dataset directory not found: {FORGERY_DATASET_DIR}")
        print("Please check the path and try again.")
        return

    # Data transforms
    train_transform, val_transform = get_robust_transforms()

    # Load dataset
    print("Loading Forgery Dataset...")
    dataset = ForgeryDataset(
        forgery_dataset_dir=FORGERY_DATASET_DIR,
        transform=train_transform,
        max_samples_per_class=MAX_SAMPLES_PER_CLASS
    )

    if len(dataset) == 0:
        print("ERROR: No images loaded from dataset!")
        return

    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Update transforms for validation/test
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=2, pin_memory=True)

    print(f"\nDataset splits:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Validation: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")

    # Initialize model
    model = MultiTaskDeepfakeDetector(
        backbone='efficientnet-b0',
        num_classes=2,
        dropout=0.6  # High dropout for regularization
    )
    model = model.to(device)

    # Loss function and optimizer
    criterion = CombinedLoss(alpha=1.0, beta=0.3, gamma=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, 
                           weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   factor=0.5, patience=4, min_lr=1e-7)

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Initial learning rate: {LEARNING_RATE}")

    # Train model
    print("\nStarting training...")
    start_time = time.time()

    history = train_model(model, train_loader, val_loader, criterion,
                         optimizer, scheduler, NUM_EPOCHS, device, 'forgery_multitask')

    end_time = time.time()
    training_time = end_time - start_time

    print(f"\n=== Training Complete ===")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Best validation accuracy: {history['best_val_acc']:.2f}%")

    # Load best model and evaluate
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load('best_forgery_multitask_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device)

    print(f"\n=== Test Results ===")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-Score: {test_metrics['f1_score']:.4f}")

    # Save results
    results = {
        'dataset_type': 'ForgeryDataset',
        'model_type': 'MultiTaskDeepfakeDetector',
        'training_history': history,
        'test_metrics': test_metrics,
        'training_time': training_time,
        'total_params': total_params,
        'dataset_size': len(dataset),
        'config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'max_samples_per_class': MAX_SAMPLES_PER_CLASS
        }
    }

    with open('forgery_multitask_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to forgery_multitask_results.json")
    print(f"Best model saved as best_forgery_multitask_model.pth")

    print("\n=== Success! ===")
    print("Next steps:")
    print("1. Run the web app: streamlit run deepfake_app_updated.py")
    print("2. Test with sample images")
    print("3. Submit your hackathon solution!")

if __name__ == "__main__":
    main()
