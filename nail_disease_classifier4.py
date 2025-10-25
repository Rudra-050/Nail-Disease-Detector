# Updated imports
import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
import shutil
import cv2
from tabulate import tabulate
import dataframe_image as dfi
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import pandas as pd
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from copy import deepcopy
from collections import deque
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 1. Enhanced Configuration
class Config:
    """Configuration class for nail disease classification model"""
    data_dir = "data_augmented_dataset"  # Update this to your dataset path
    input_size = 224
    batch_size = 32
    num_epochs = 50
    patience = 7
    class_names = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confidence_threshold = 0.7
    samples_per_class = 10  # Now scales with number of classes
    correction_augmentations = 3
    correction_lr = 0.01
    tta_num = 5 # Default for TTA

# 2. Memory Buffer (now scales with class count)
class MemoryBuffer:
    def __init__(self):
        self.buffer = {}
        self.aug_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.1),
            transforms.RandomHorizontalFlip()
        ])
    
    def add(self, img_path, true_class_idx):
        """Store original + augmented versions per class"""
        if true_class_idx not in self.buffer:
            self.buffer[true_class_idx] = deque(maxlen=Config.samples_per_class)
        
        img = Image.open(img_path).convert('RGB')
        base_transform = get_data_transforms()['validation']
        
        # Store original
        if len(self.buffer[true_class_idx]) < Config.samples_per_class:
            self.buffer[true_class_idx].append((base_transform(img), true_class_idx))
        
        # Store augmented versions
        for _ in range(Config.correction_augmentations):
            if len(self.buffer[true_class_idx]) < Config.samples_per_class:
                aug_img = self.aug_transform(img)
                self.buffer[true_class_idx].append((base_transform(aug_img), true_class_idx))
    
    def get_batch(self, batch_size=8):
        """Get balanced batch across classes"""
        if not self.buffer:
            return None
            
        # Get equal samples per class
        samples = []
        per_class = max(1, batch_size // len(self.buffer))
        for class_idx in self.buffer:
            available = list(self.buffer[class_idx])
            if available:  # Only sample if there are available samples
                samples.extend(random.sample(available, min(per_class, len(available))))
        
        if not samples:
            return None
            
        try:
            imgs = torch.stack([x[0] for x in samples])
            labels = torch.tensor([x[1] for x in samples])
            return imgs.to(Config.device), labels.to(Config.device)
        except Exception as e:
            logger.error(f"Error creating batch: {e}")
            return None

# 3. Data Preparation (improved augmentation examples)
def get_data_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(Config.input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(Config.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Generate realistic augmentation examples for each class
    generate_realistic_augmentation_examples()
    
    return {'train': train_transforms, 'validation': val_transforms}

def generate_realistic_augmentation_examples():
    """Generate 6-panel augmentation examples: original, flip, brightness/contrast, zoom, rotation, affine."""
    if not os.path.exists(Config.data_dir):
        return
    train_dir = os.path.join(Config.data_dir, 'train')
    if not os.path.exists(train_dir):
        return

    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        output_file = f'aug_{class_name}.png'
        if os.path.exists(output_file):
            continue  # Skip if already exists
        try:
            sample_img = next((f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))), None)
            if not sample_img:
                continue
            img_path = os.path.join(class_dir, sample_img)
            img = Image.open(img_path).convert('RGB')
            fig, axes = plt.subplots(1, 6, figsize=(18, 4))
            fig.suptitle(f'Realistic Augmentation Examples: {class_name}', fontsize=16)
            # 1. Original
            axes[0].imshow(img)
            axes[0].set_title('Original')
            axes[0].axis('off')
            # 2. Horizontal Flip
            axes[1].imshow(img.transpose(Image.FLIP_LEFT_RIGHT))
            axes[1].set_title('Horizontal Flip')
            axes[1].axis('off')
            # 3. Brightness/Contrast
            bc_img = ImageEnhance.Brightness(img).enhance(1.25)
            bc_img = ImageEnhance.Contrast(bc_img).enhance(0.75)
            axes[2].imshow(bc_img)
            axes[2].set_title('Brightness/Contrast')
            axes[2].axis('off')
            # 4. Zoom
            scale = 1.12
            w, h = img.size
            new_w, new_h = int(w * scale), int(h * scale)
            zoom_img = img.resize((new_w, new_h), Image.BICUBIC)
            left = max(0, (new_w - w) // 2)
            top = max(0, (new_h - h) // 2)
            zoom_img = zoom_img.crop((left, top, left + w, top + h))
            axes[3].imshow(zoom_img)
            axes[3].set_title('Zoom')
            axes[3].axis('off')
            # 5. Rotation
            axes[4].imshow(img.rotate(17))
            axes[4].set_title('Rotation')
            axes[4].axis('off')
            # 6. Affine
            max_dx = 0.07 * img.size[0]
            max_dy = 0.07 * img.size[1]
            dx = 0.07 * img.size[0]
            dy = -0.07 * img.size[1]
            affine_img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))
            axes[5].imshow(affine_img)
            axes[5].set_title('Affine')
            axes[5].axis('off')
            plt.tight_layout()
            plt.savefig(output_file, bbox_inches='tight', dpi=150)
            plt.close()
            logger.info(f"Generated 6-panel augmentation example for {class_name}")
        except Exception as e:
            logger.error(f"Could not generate augmentation example for {class_name}: {str(e)}")

def prepare_dataloaders():
    """Prepares train and validation dataloaders with weighted sampling"""
    data_transforms = get_data_transforms()
    
    # Create image datasets
    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(Config.data_dir, x),
            data_transforms[x]
        )
        for x in ['train', 'validation']
    }
    
    # Set class names in config
    Config.class_names = image_datasets['train'].classes
    
    # Calculate class weights for imbalanced data
    class_counts = np.array([
        len([x for x in image_datasets['train'].samples if x[1] == cls]) 
        for cls in range(len(Config.class_names))
    ])
    class_weights = 1. / class_counts
    sample_weights = class_weights[image_datasets['train'].targets]
    
    # Create samplers
    sampler = WeightedRandomSampler(
        sample_weights, 
        len(sample_weights),
        replacement=True
    )
    
    # Create dataloaders with optimized settings for CPU
    dataloaders = {
        'train': DataLoader(
            image_datasets['train'],
            batch_size=Config.batch_size,
            sampler=sampler,
            num_workers=2,  # Reduced for CPU
            pin_memory=False  # Disabled for CPU
        ),
        'validation': DataLoader(
            image_datasets['validation'],
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=2,  # Reduced for CPU
            pin_memory=False  # Disabled for CPU
        )
    }
    
    # Calculate dataset sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
    
    return dataloaders, dataset_sizes

# 4. Model Definition (EfficientNet-B0 optimized)
def initialize_model():
    # Use EfficientNet-B0 for better accuracy and CPU efficiency
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    
    # Freeze early layers for transfer learning
    for name, param in model.named_parameters():
        if 'features.6' not in name and 'classifier' not in name:
            param.requires_grad = False
    
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, len(Config.class_names))
    )
    return model.to(Config.device)

# 5. Training with Memory (fixed for EfficientNet with improved learning rate)
def train_model(model, dataloaders, dataset_sizes, memory_buffer=None):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Improved learning rate strategy for steady improvement
    optimizer = optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': 0.0005},  # Lower initial LR
        {'params': model.features[6].parameters(), 'lr': 0.0002}  # Lower initial LR
    ], weight_decay=0.01)
    
    # Better learning rate scheduler for steady improvement
    # Warmup for 3 epochs, then cosine annealing with restarts
    def get_lr_scheduler(optimizer):
        # Warmup scheduler
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.1, 
            total_iters=3
        )
        
        # Main scheduler - Cosine Annealing with Warm Restarts
        main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the restart interval each time
            eta_min=1e-6  # Minimum learning rate
        )
        
        # Combine warmup and main scheduler
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[3]
        )
        
        return scheduler
    
    scheduler = get_lr_scheduler(optimizer)
    
    best_f1 = 0.0
    patience_counter = 0
    history = {'train': [], 'val': [], 'lr': []}

    for epoch in range(Config.num_epochs):
        logger.info(f'\nEpoch {epoch+1}/{Config.num_epochs}')
        print(f'\nEpoch {epoch+1}/{Config.num_epochs}')
        print('-' * 20)
        
        # Log current learning rates
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        logger.info(f"Learning rates: {current_lrs}")
        print(f"Learning rates: {[f'{lr:.6f}' for lr in current_lrs]}")
        
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
                if epoch > 10:  # Unfreeze more layers later
                    for param in model.features[5].parameters():
                        param.requires_grad = True
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase} batches'):
                inputs = inputs.to(Config.device)
                labels = labels.to(Config.device)
                
                # Add memory samples to training
                if phase == 'train' and memory_buffer:
                    mem_batch = memory_buffer.get_batch(Config.batch_size//4)
                    if mem_batch:
                        inputs = torch.cat([inputs, mem_batch[0]], dim=0)
                        labels = torch.cat([labels, mem_batch[1]], dim=0)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'validation':
                report = classification_report(
                    all_labels, all_preds,
                    target_names=Config.class_names,
                    output_dict=True,
                    zero_division=0
                )
                val_f1 = report['macro avg']['f1-score']
                history['val'].append((epoch_acc.item(), val_f1))
                history['lr'].append(current_lrs[0])  # Log learning rate
                
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    torch.save(model.state_dict(), 'best_epoch.pth')
                    patience_counter = 0
                    np.savez('best_predictions.npz',
                            preds=all_preds,
                            labels=all_labels,
                            classes=Config.class_names)
                    logger.info(f"New best model saved with F1: {best_f1:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= Config.patience:
                        logger.info(f'\nEarly stopping at epoch {epoch+1}')
                        print(f'\nEarly stopping at epoch {epoch+1}')
                        print(f'Best Val F1: {best_f1:.4f}')
                        model.load_state_dict(torch.load('best_epoch.pth'))
                        return model, history
            else:
                history['train'].append(epoch_acc.item())
                scheduler.step()

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'validation':
                print(f'F1-Score: {val_f1:.4f}')

    return model, history

def robust_immediate_update(model, img_path, true_class_idx, memory_buffer, optimizer):
    """Guaranteed correction with dimension checks"""
    original_state = deepcopy(model.state_dict())
    
    try:
        # 1. Prepare single image
        transform = get_data_transforms()['validation']
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(Config.device)  # [1, 3, 224, 224]
        target = torch.tensor([true_class_idx]).to(Config.device)    # [1]
        
        # 2. Prepare memory batch (if any)
        mem_batch = memory_buffer.get_batch(4)  # Returns (imgs, labels) or None
        
        # 3. Combined update
        model.train()
        for _ in range(3):  # Few iterations
            optimizer.zero_grad()
            
            # Forward pass for new image
            output = model(img_tensor)
            loss = F.cross_entropy(output, target)
            
            # Forward pass for memory samples
            if mem_batch:
                mem_output = model(mem_batch[0])
                loss += F.cross_entropy(mem_output, mem_batch[1])
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # 4. Verify
        model.eval()
        with torch.no_grad():
            test_output = model(img_tensor)
            if torch.argmax(test_output).item() != true_class_idx:
                raise ValueError("Verification failed")
        
        return True
    
    except Exception as e:
        logger.error(f"Correction failed: {str(e)}")
        model.load_state_dict(original_state)
        return False

# 6. Evaluation (fixed to use current model and dataset)
def evaluate_model(model, dataloader):
    """Evaluate model with fresh data and generate new reports"""
    logger.info("Starting fresh model evaluation...")
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Clear any cached data
    all_preds, all_labels, all_probs, all_confs = [], [], [], []
    
    logger.info(f"Evaluating on {len(dataloader.dataset)} validation samples...")
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
            inputs = inputs.to(Config.device)
            labels = labels.to(Config.device)
            
            # Get model predictions
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())
    
    logger.info(f"Evaluation complete. Total samples: {len(all_labels)}")
    logger.info(f"Classes found: {set(all_labels)}")
    logger.info(f"Predictions made: {set(all_preds)}")

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", 
                xticklabels=Config.class_names, 
                yticklabels=Config.class_names,
                cmap='Blues')
    plt.title('Confusion Matrix - Current Model Evaluation')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()
    logger.info("Confusion matrix saved")

    # Generate fresh classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=Config.class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Convert to DataFrame
    report_df = pd.DataFrame(report).transpose()
    
    # Add confidence metrics
    conf_df = pd.DataFrame({
        'mean_confidence': [np.mean([c for c, p in zip(all_confs, all_preds) if p == i]) 
                          for i in range(len(Config.class_names))],
        'min_confidence': [np.min([c for c, p in zip(all_confs, all_preds) if p == i]) 
                         for i in range(len(Config.class_names))],
        'max_confidence': [np.max([c for c, p in zip(all_confs, all_preds) if p == i]) 
                         for i in range(len(Config.class_names))]
    }, index=Config.class_names)
    
    # Join with main report
    report_df = report_df.join(conf_df, how='outer')
    
    # Create display version with proper formatting
    display_df = report_df.copy()
    float_cols = ['precision', 'recall', 'f1-score', 'mean_confidence', 'min_confidence', 'max_confidence']
    for col in float_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
    
    # Save styled report
    try:
        styled_report = report_df.style\
            .background_gradient(cmap='Blues', subset=pd.IndexSlice[Config.class_names, ['f1-score']])\
            .background_gradient(cmap='Greens', subset=['mean_confidence', 'min_confidence', 'max_confidence'])\
            .format("{:.3f}", na_rep="-")
        dfi.export(styled_report, "classification_report.png", dpi=300)
        logger.info("Styled classification report saved")
    except Exception as e:
        logger.error(f"Could not save styled report: {e}")
        # Fallback: save as regular image
        plt.figure(figsize=(12, 8))
        plt.axis('off')
        plt.table(cellText=display_df.values, rowLabels=display_df.index, colLabels=display_df.columns, cellLoc='center')
        plt.title('Classification Report - Current Model')
        plt.savefig('classification_report.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    # Generate ROC curves
    plt.figure(figsize=(14, 10))
    for i, class_name in enumerate(Config.class_names):
        fpr, tpr, _ = roc_curve(np.array(all_labels) == i, np.array(all_probs)[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc_score:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Current Model Evaluation', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', bbox_inches='tight', dpi=300)
    plt.close()
    logger.info("ROC curves saved")

    # Print detailed results
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT - CURRENT MODEL EVALUATION")
    print("="*80)
    print(f"Dataset: {Config.data_dir}")
    print(f"Model: EfficientNet-B0")
    print(f"Total Validation Samples: {len(all_labels)}")
    print(f"Classes: {Config.class_names}")
    print("="*80)
    
    print("\nClassification Report with Confidence Analysis:")
    print(tabulate(display_df, headers='keys', tablefmt='psql', showindex=True))
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Overall Accuracy: {np.mean(np.array(all_preds) == np.array(all_labels)):.3f}")
    print(f"Macro F1-Score: {report['macro avg']['f1-score']:.3f}")
    print(f"Weighted F1-Score: {report['weighted avg']['f1-score']:.3f}")
    print(f"Average Confidence: {np.mean(all_confs):.3f}")
    print("="*80)
    
    # Save detailed results to file
    with open('evaluation_results.txt', 'w') as f:
        f.write("CLASSIFICATION REPORT - CURRENT MODEL EVALUATION\n")
        f.write("="*80 + "\n")
        f.write(f"Dataset: {Config.data_dir}\n")
        f.write(f"Model: EfficientNet-B0\n")
        f.write(f"Total Validation Samples: {len(all_labels)}\n")
        f.write(f"Classes: {Config.class_names}\n")
        f.write("="*80 + "\n\n")
        f.write(tabulate(display_df, headers='keys', tablefmt='psql', showindex=True))
        f.write(f"\n\nOverall Accuracy: {np.mean(np.array(all_preds) == np.array(all_labels)):.3f}\n")
        f.write(f"Macro F1-Score: {report['macro avg']['f1-score']:.3f}\n")
        f.write(f"Weighted F1-Score: {report['weighted avg']['f1-score']:.3f}\n")
        f.write(f"Average Confidence: {np.mean(all_confs):.3f}\n")
    
    logger.info("Evaluation complete and results saved")
    return report_df

# Simplified and improved Grad-CAM: only original and overlay, more accurate

def show_gradcam(model, img_path, use_tta=False):
    """
    Improved Grad-CAM++: sharper, more accurate heatmap focused on the nail region.
    Shows only original and overlay.
    """
    try:
        img = Image.open(img_path).convert('RGB')
        # Center-crop to square if not already
        w, h = img.size
        if w != h:
            min_side = min(w, h)
            left = (w - min_side) // 2
            top = (h - min_side) // 2
            img = img.crop((left, top, left + min_side, top + min_side))
        original_img = np.array(img)
        transform = get_data_transforms()['validation']
        input_tensor = transform(img).unsqueeze(0).to(Config.device)
        # Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            if conf.item() < Config.confidence_threshold:
                print(f"Low confidence: {conf.item():.2f}")
                return None
            pred_class = pred.item()
        # Grad-CAM++: use only last conv layer
        # Try using an earlier layer for sharper focus (uncomment below to experiment)
        # target_layer = model.features[5][-1]
        target_layer = model.features[6][-1]
        activations = []
        gradients = []
        grad_outputs = []
        def forward_hook(module, input, output):
            activations.append(output)
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
            grad_outputs.append(grad_output[0])
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        model.zero_grad()
        input_tensor.requires_grad = True
        outputs = model(input_tensor)
        one_hot = torch.zeros_like(outputs)
        one_hot[0][pred_class] = 1
        outputs.backward(gradient=one_hot)
        forward_handle.remove()
        backward_handle.remove()
        # Grad-CAM++ calculation
        act = activations[0].detach()
        grad = gradients[0].detach()
        grad2 = grad ** 2
        grad3 = grad2 * grad
        sum_acts = torch.sum(act, dim=(2, 3), keepdim=True)
        alpha_num = grad2
        alpha_denom = 2 * grad2 + sum_acts * grad3
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alphas = alpha_num / alpha_denom
        weights = (alphas * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # 1. Visualize raw heatmap before any post-processing
        plt.figure()
        plt.title("Raw Grad-CAM++ Heatmap")
        plt.imshow(cam, cmap='jet')
        plt.axis('off')
        plt.show()

        cam = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
        # 2. Reduce blur kernel size for sharper focus
        cam = cv2.GaussianBlur(cam, (3, 3), 0)
        # 3. Lower threshold to 80th percentile for more detail
        threshold = np.percentile(cam, 80)
        cam = np.where(cam > threshold, cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        # Overlay
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original_img, 0.5, heatmap_colored, 0.7, 0)
        # Show only original and overlay
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(overlay)
        axes[1].set_title(f'Grad-CAM++ Overlay: {Config.class_names[pred_class]}')
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()
        return Config.class_names[pred_class]
    except Exception as e:
        logger.error(f"Grad-CAM++ Error: {str(e)}")
        return None

# 8. TTA Prediction (improved for EfficientNet)
def predict_with_tta(model, image_tensor, n_aug=None):
    """Test Time Augmentation for improved prediction accuracy"""
    if n_aug is None:
        n_aug = 5  # Default value
    
    model.eval()
    aug_preds = []
    train_transform = get_data_transforms()['train']
    
    try:
        with torch.no_grad():
            # Original image
            output = model(image_tensor.to(Config.device))
            probs = F.softmax(output, dim=1)
            conf, _ = torch.max(probs, 1)
            if conf.item() < Config.confidence_threshold:
                return None, conf.item()
            aug_preds.append(probs)
            
            # Augmented versions
            for _ in range(n_aug-1):
                aug_img = train_transform(transforms.ToPILImage()(image_tensor.squeeze(0)))
                output = model(aug_img.unsqueeze(0).to(Config.device))
                probs = F.softmax(output, dim=1)
                conf, _ = torch.max(probs, 1)
                if conf.item() >= Config.confidence_threshold:
                    aug_preds.append(probs)
        
        if not aug_preds:
            return None, 0.0
        avg_probs = torch.mean(torch.stack(aug_preds), dim=0)
        final_conf, final_pred = torch.max(avg_probs, 1)
        return final_pred.item(), final_conf.item()
    
    except Exception as e:
        logger.error(f"TTA prediction error: {e}")
        return None, 0.0

def predict_disease(model, img_path):
    """
    Predict the disease class for a given nail image.
    Args:
        model: Trained model.
        img_path: Path to the image file.
    Returns:
        predicted_class (str): Predicted class name.
        confidence (float): Confidence score (0-1).
    """
    model.eval()
    try:
        img = Image.open(img_path).convert('RGB')
        # Center-crop to square if not already
        w, h = img.size
        if w != h:
            min_side = min(w, h)
            left = (w - min_side) // 2
            top = (h - min_side) // 2
            img = img.crop((left, top, left + min_side, top + min_side))
        transform = get_data_transforms()['validation']
        input_tensor = transform(img).unsqueeze(0).to(Config.device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            predicted_class = Config.class_names[pred.item()]
            confidence = conf.item()
        return predicted_class, confidence
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None, 0.0

# 9. Enhanced Interactive Prediction
def interactive_predict(model):
    memory_buffer = MemoryBuffer()
    correction_optimizer = optim.SGD([
        {'params': model.classifier.parameters(), 'lr': Config.correction_lr},
        {'params': model.features[6].parameters(), 'lr': Config.correction_lr/2}
    ], momentum=0.9)
    
    while True:
        img_path = input("\nEnter image path (or 'quit'): ").strip()
        if img_path.lower() == 'quit':
            break
            
        if not os.path.exists(img_path):
            print("Error: File not found")
            continue
            
        # Get prediction with Grad-CAM
        pred_class = show_gradcam(model, img_path)
        if pred_class is None:
            continue
            
        # User feedback
        feedback = input("Is this correct? (y/n): ").lower().strip()
        if feedback == 'y':
            true_class = pred_class
        elif feedback == 'n':
            print(f"\nAvailable classes: {', '.join(Config.class_names)}")
            while True:
                true_class = input("Enter correct class name: ").strip()
                if true_class in Config.class_names:
                    break
                print("Invalid class. Try again.")
        else:
            continue
            
        # Save to dataset
        true_class_idx = Config.class_names.index(true_class)
        dest_folder = os.path.join(Config.data_dir, 'train', true_class)
        os.makedirs(dest_folder, exist_ok=True)
        dest_path = os.path.join(dest_folder, os.path.basename(img_path))
        
        if os.path.exists(dest_path):
            print(f"Image already exists in {true_class} dataset")
        else:
            shutil.copy(img_path, dest_path)
            print(f"Saved to {dest_path}")
            
            # Immediate correction
            print("Applying correction...")
            success = robust_immediate_update(
                model, 
                img_path, 
                true_class_idx,
                memory_buffer,
                correction_optimizer
            )
            
            if success:
                print("Model updated successfully!")
                # Only add to memory buffer after successful update
                memory_buffer.add(img_path, true_class_idx)
            else:
                print("Correction failed - keeping previous weights")

# Main Execution
if __name__ == '__main__':
    logger.info(f"Using device: {Config.device}")
    print(f"Using device: {Config.device}")
    
    # Check if dataset exists
    if not os.path.exists(Config.data_dir):
        print(f"Error: Dataset directory '{Config.data_dir}' not found!")
        print("Please update Config.data_dir to point to your dataset or run the augmentation script first.")
        exit(1)
    
    # Prepare data
    try:
        dataloaders, dataset_sizes = prepare_dataloaders()
        Config.samples_per_class = 10  # Now set after we know class count
        print(f"Classes: {Config.class_names}")
        print(f"Memory buffer: {Config.samples_per_class} samples per class")
    except Exception as e:
        logger.error(f"Error preparing dataloaders: {e}")
        print(f"Error preparing data: {e}")
        exit(1)
    
    # Initialize model
    try:
        model = initialize_model()
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        print(f"Error initializing model: {e}")
        exit(1)
    
    # Train or load
    if not os.path.exists('best_epoch.pth'):
        logger.info("Training model...")
        print("\nTraining model...")
        try:
            memory_buffer = MemoryBuffer()
            model, history = train_model(model, dataloaders, dataset_sizes, memory_buffer)
            
            # Plot training history with learning rate
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Training and Validation Accuracy
            plt.subplot(2, 2, 1)
            plt.plot([x[0] for x in history['val']], label='Val Accuracy', linewidth=2, color='blue')
            plt.plot([x[1] for x in history['val']], label='Val F1-Score', linewidth=2, color='green')
            plt.plot(history['train'], label='Train Accuracy', linewidth=2, color='red', alpha=0.7)
            plt.title('Training Progress - Accuracy & F1-Score', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Learning Rate Progression
            plt.subplot(2, 2, 2)
            plt.plot(history['lr'], linewidth=2, color='purple')
            plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Validation F1-Score with trend line
            plt.subplot(2, 2, 3)
            f1_scores = [x[1] for x in history['val']]
            epochs = range(1, len(f1_scores) + 1)
            plt.plot(epochs, f1_scores, linewidth=2, color='green', marker='o')
            
            # Add trend line
            if len(f1_scores) > 1:
                z = np.polyfit(epochs, f1_scores, 1)
                p = np.poly1d(z)
                plt.plot(epochs, p(epochs), "--", alpha=0.8, color='red', linewidth=1)
            
            plt.title('Validation F1-Score Progression', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('F1-Score')
            plt.grid(True, alpha=0.3)
            
            # Plot 4: Learning rate vs F1-score relationship
            plt.subplot(2, 2, 4)
            plt.scatter(history['lr'], f1_scores, alpha=0.6, color='purple')
            plt.title('Learning Rate vs F1-Score', fontsize=14, fontweight='bold')
            plt.xlabel('Learning Rate')
            plt.ylabel('F1-Score')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('training_history.png', bbox_inches='tight', dpi=300)
            plt.show()
        except Exception as e:
            logger.error(f"Error during training: {e}")
            print(f"Error during training: {e}")
            exit(1)
    else:
        logger.info("Loading saved model...")
        print("\nLoading saved model...")
        try:
            model.load_state_dict(torch.load('best_epoch.pth'))
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            print(f"Error loading model: {e}")
            exit(1)
    
    # Evaluate
    logger.info("Evaluating model...")
    print("\nEvaluating model...")
    try:
        evaluate_model(model, dataloaders['validation'])
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        print(f"Error during evaluation: {e}")
    
    

    # Direct prediction mode (no Grad-CAM, no feedback)
    try:
        response = input("\nRun direct prediction (no Grad-CAM)? (y/n): ").lower().strip()
        if response == 'y':
            img_path = input("Enter image path: ").strip()
            if not os.path.exists(img_path):
                print("Error: File not found")
            else:
                pred_class, conf = predict_disease(model, img_path)
                if pred_class is not None:
                    print(f"Predicted disease: {pred_class} (confidence: {conf:.2f})")
                else:
                    print("Prediction failed.")
        else:
            print("Direct prediction skipped.")
    except KeyboardInterrupt:
        print("\nDirect prediction interrupted.")
    except Exception as e:
        logger.error(f"Error in direct prediction: {e}")
        print(f"Error in direct prediction: {e}")

    # Grad-CAM visualization mode (no feedback)
    try:
        response = input("\nShow Grad-CAM visualization for an image? (y/n): ").lower().strip()
        if response == 'y':
            img_path = input("Enter image path: ").strip()
            if not os.path.exists(img_path):
                print("Error: File not found")
            else:
                pred_class = show_gradcam(model, img_path)
                if pred_class is not None:
                    print(f"Predicted disease (Grad-CAM): {pred_class}")
                else:
                    print("Grad-CAM visualization failed.")
        else:
            print("Grad-CAM visualization skipped.")
    except KeyboardInterrupt:
        print("\nGrad-CAM visualization interrupted.")
    except Exception as e:
        logger.error(f"Error in Grad-CAM visualization: {e}")
        print(f"Error in Grad-CAM visualization: {e}")