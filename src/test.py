#!/usr/bin/env python3
"""
Test script for fruit classification model evaluation.

This script loads a trained model and evaluates it on the test set,
providing detailed metrics and optionally generating visualizations.
"""

import argparse
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import lightning as pl

from src.lightning_model import ResNetLightningModule
from src.dataset import get_data_loaders


def load_model(checkpoint_path: str, num_classes: int, model_name: str = "resnet50") -> ResNetLightningModule:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        num_classes: Number of classes in the dataset
        model_name: Name of the ResNet model architecture
        
    Returns:
        Loaded model
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Initialize the model with the same parameters as training
    model = ResNetLightningModule(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=False,  # We're loading from checkpoint
    )
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def get_test_data_loader(data_dir: str, batch_size: int = 32, num_classes: Optional[int] = None) -> Tuple[DataLoader, List[str]]:
    """
    Create a data loader for the test set.
    
    Args:
        data_dir: Directory containing the test data
        batch_size: Batch size for evaluation
        num_classes: Number of classes to use (if None, uses all classes)
        
    Returns:
        Test data loader and class names
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Test directory not found: {data_dir}")
    
    # Use the same transforms as validation (no augmentation)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # Create test dataset
    test_dataset = datasets.ImageFolder(root=data_dir, transform=test_transforms)
    
    # Filter by number of classes if specified
    all_classes = test_dataset.classes
    if num_classes is not None:
        sorted_classes = sorted(all_classes)
        selected_classes = sorted_classes[:num_classes]
        
        # Get indices of selected classes
        class_indices = [all_classes.index(cls) for cls in selected_classes]
        
        # Filter dataset to only include selected classes
        test_indices = [i for i, (_, class_idx) in enumerate(test_dataset.samples) 
                       if class_idx in class_indices]
        
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, test_indices)
        # Add classes attribute to the subset
        classes = selected_classes
    else:
        classes = all_classes
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    return test_loader, classes


def evaluate_model(model: ResNetLightningModule, test_loader: DataLoader, classes: List[str], device: str = 'cpu') -> Dict:
    """
    Evaluate the model on the test set.
    
    Args:
        model: The trained model
        test_loader: Test data loader
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    total_loss = 0.0
    num_batches = 0
    
    criterion = nn.CrossEntropyLoss()
    
    print("Running evaluation on test set...")
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

            correct = (predictions == targets).sum().item()
            total = targets.size(0)
            print (f"batch accuracy: {correct / total}")
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Calculate metrics
    avg_loss = total_loss / num_batches
    accuracy = accuracy_score(all_targets, all_predictions)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # Generate classification report
    class_report = classification_report(
        all_targets, 
        all_predictions, 
        target_names=classes,
        output_dict=True
    )
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    results = {
        'test_loss': avg_loss,
        'test_accuracy': accuracy,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'num_samples': len(all_targets)
    }
    
    return results


def save_results(results: Dict, output_dir: str, model_name: str, classes: List[str]):
    """
    Save evaluation results to files.
    
    Args:
        results: Evaluation results dictionary
        output_dir: Directory to save results
        model_name: Name of the model for file naming
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to JSON
    metrics = {
        'test_loss': results['test_loss'],
        'test_accuracy': results['test_accuracy'],
        'num_samples': results['num_samples']
    }
    
    metrics_file = os.path.join(output_dir, f"{model_name}_test_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {metrics_file}")
    
    # Save detailed classification report
    report_file = os.path.join(output_dir, f"{model_name}_classification_report.txt")
    with open(report_file, 'w') as f:
        f.write("Classification Report:\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test Loss: {results['test_loss']:.4f}\n")
        f.write(f"Test Accuracy: {results['test_accuracy']:.4f}\n")
        f.write(f"Number of Samples: {results['num_samples']}\n\n")
        
        # Convert classification report back to string format
        report_str = classification_report(
            results['targets'], 
            results['predictions'],
            target_names=classes
        )
        f.write(str(report_str))
    
    print(f"Detailed report saved to {report_file}")


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], output_dir: str, model_name: str):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix
        classes: List of class names
        output_dir: Directory to save plot
        model_name: Name of the model for file naming
    """
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    
    plt.title(f'Normalized Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix plot saved to {plot_file}")


def plot_top_k_accuracy(results: Dict, classes: List[str], output_dir: str, model_name: str, k_values: List[int] = [1, 3, 5]):
    """
    Plot top-k accuracy for different k values.
    
    Args:
        results: Evaluation results dictionary
        classes: List of class names
        output_dir: Directory to save plot
        model_name: Name of the model for file naming
        k_values: List of k values to evaluate
    """
    probabilities = results['probabilities']
    targets = results['targets']
    
    top_k_accuracies = []
    
    for k in k_values:
        # Get top-k predictions
        top_k_indices = np.argsort(probabilities, axis=1)[:, -k:]
        
        # Check if true label is in top-k predictions
        correct = 0
        for i, true_label in enumerate(targets):
            if true_label in top_k_indices[i]:
                correct += 1
        
        accuracy = correct / len(targets)
        top_k_accuracies.append(accuracy)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(k_values)), top_k_accuracies)
    plt.xlabel('k')
    plt.ylabel('Top-k Accuracy')
    plt.title(f'Top-k Accuracy - {model_name}')
    plt.xticks(range(len(k_values)), [str(k) for k in k_values])
    
    # Add value labels on bars
    for i, acc in enumerate(top_k_accuracies):
        plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, f"{model_name}_top_k_accuracy.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Top-k accuracy plot saved to {plot_file}")


def plot_random_test_images(model: ResNetLightningModule, test_loader: DataLoader, classes: List[str], output_dir: str, model_name: str, device: str = 'cpu', num_images: int = 10):
    """
    Plot a subplot of randomly selected test images with their predictions and success/failure indicators.
    
    Args:
        model: The trained model
        test_loader: Test data loader
        classes: List of class names
        output_dir: Directory to save plot
        model_name: Name of the model for file naming
        device: Device to run model on
        num_images: Number of images to plot (default 10 for 2x5 grid)
    """
    model.eval()
    
    # Get a random sample of images from the entire dataset
    dataset_size = len(test_loader.dataset)
    all_indices = list(range(dataset_size))
    selected_indices = random.sample(all_indices, num_images)
    
    # Create a figure with subplots (2 rows, 5 columns)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()  # Flatten for easier indexing
    
    for i, idx in enumerate(selected_indices):
        # Get the image and target
        image, target = test_loader.dataset[idx]
        
        # Prepare image for model input
        image_batch = image.unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(image_batch)
            probabilities = torch.softmax(output, dim=1)
            predicted_class_idx = torch.argmax(output, dim=1).item()
        
        # Convert image to numpy array for plotting
        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np * 0.5 + 0.5)  # Inverse normalization for display
        image_np = np.clip(image_np, 0, 1)  # Clip to valid range
        
        # Get class names - handle both tensor and int targets
        if hasattr(target, 'item'):
            target_idx = int(target.item())
        else:
            target_idx = int(target)
        true_class_name = classes[target_idx]
        predicted_class_name = classes[predicted_class_idx]
        
        # Determine success/failure
        is_correct = (predicted_class_idx == target_idx)
        
        # Plot image
        axes[i].imshow(image_np)
        axes[i].set_title(f"True: {true_class_name}\nPred: {predicted_class_name}", fontsize=10)
        axes[i].axis('off')
        
        # Add success/failure indicator
        if is_correct:
            # Green circle for correct prediction
            circle = patches.Circle((0.1, 0.9), 0.05, transform=axes[i].transAxes, 
                                  facecolor='green', edgecolor='black', linewidth=2)
            axes[i].add_patch(circle)
            axes[i].text(0.1, 0.9, '✓', transform=axes[i].transAxes, 
                        ha='center', va='center', fontsize=16, fontweight='bold', color='white')
        else:
            # Red X for incorrect prediction
            circle = patches.Circle((0.1, 0.9), 0.05, transform=axes[i].transAxes, 
                                  facecolor='red', edgecolor='black', linewidth=2)
            axes[i].add_patch(circle)
            axes[i].text(0.1, 0.9, '✗', transform=axes[i].transAxes, 
                        ha='center', va='center', fontsize=16, fontweight='bold', color='white')
    
    plt.suptitle(f'Random Test Images - {model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, f"{model_name}_random_test_images.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Random test images plot saved to {plot_file}")


def main():
    checkpoint_path = "/home/chadi/Dev/fruit-classification-kaggle/lightning_logs/resnet50-basic/version_5/checkpoints/epoch=12-val_acc=0.922.ckpt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    no_plots = False
    output_dir = "test_results"
    
    print(f"Using device: {device}")
    
    # Load test data
    print("Loading test data...")
    test_loader, classes = get_test_data_loader(
        data_dir="data/Fruit_dataset/val1",
        batch_size=32,
        num_classes=20
    )
    
    print(f"Test set size: {len(test_loader)} samples")
    print(f"Number of classes: {len(classes)}")
    
    # Load model
    model = load_model(
        checkpoint_path=checkpoint_path,
        num_classes=len(classes),
        model_name="resnet50"
    )
    
    # Evaluate model
    results = evaluate_model(model, test_loader, classes, device)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Number of Samples: {results['num_samples']}")
    print("="*50)
    
    # Save results
    model_name = Path(checkpoint_path).stem
    save_results(results, "test_results", model_name, classes)
    
    # Generate plots if requested
    if not no_plots:
        print("\nGenerating plots...")
        plot_confusion_matrix(results['confusion_matrix'], classes, "test_results", model_name)
        plot_top_k_accuracy(results, classes, "test_results", model_name)
        plot_random_test_images(model, test_loader, classes, "test_results", model_name, device)
    
    print(f"\nEvaluation completed! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
