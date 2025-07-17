from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from typing import Optional, List


def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    use_augmentation: bool = True,
    augmentation_strength: str = "medium",
    classes: Optional[List[str]] = None,  # New parameter to select subset of classes
    num_classes: Optional[int] = None,  # Number of classes to use (first N when sorted)
):
    """
    Get data loaders with optional augmentation.

    Args:
        data_dir: Directory containing train1 and val1 folders
        batch_size: Batch size for data loaders
        use_augmentation: Whether to apply augmentation to training data
        augmentation_strength: "light", "medium", or "heavy" augmentation
        classes: List of class names to include (None for all classes)
        num_classes: Number of classes to use (first N when sorted alphabetically)
    """

    # Define augmentation strength levels
    augmentation_configs = {
        "light": {
            "horizontal_flip_p": 0.3,
            "rotation_degrees": 10,
            "color_jitter": (0.1, 0.1, 0.1, 0.05),
            "translation": (0.05, 0.05),
        },
        "medium": {
            "horizontal_flip_p": 0.5,
            "rotation_degrees": 15,
            "color_jitter": (0.2, 0.2, 0.2, 0.1),
            "translation": (0.1, 0.1),
        },
        "heavy": {
            "horizontal_flip_p": 0.5,
            "rotation_degrees": 25,
            "color_jitter": (0.3, 0.3, 0.3, 0.15),
            "translation": (0.15, 0.15),
        },
    }

    # Training transforms with optional augmentation
    if use_augmentation:
        config = augmentation_configs[augmentation_strength]
        train_transforms = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),  # Resize to fixed size
                transforms.RandomHorizontalFlip(p=config["horizontal_flip_p"]),
                transforms.RandomRotation(degrees=config["rotation_degrees"]),
                transforms.ColorJitter(
                    brightness=config["color_jitter"][0],
                    contrast=config["color_jitter"][1],
                    saturation=config["color_jitter"][2],
                    hue=config["color_jitter"][3],
                ),
                transforms.RandomAffine(degrees=0, translate=config["translation"]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
        # No augmentation for training
        train_transforms = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    # Validation transforms (no augmentation for proper evaluation)
    val_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Create datasets
    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train1", transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        root=f"{data_dir}/val1", transform=val_transforms
    )

    # Filter datasets by selected classes if specified
    if classes is not None or num_classes is not None:
        # Get all available classes
        all_classes = train_dataset.classes
        
        # Determine which classes to use
        if num_classes is not None:
            # Sort classes alphabetically and take first N
            sorted_classes = sorted(all_classes)
            classes = sorted_classes[:num_classes]
        elif classes is None:
            # This shouldn't happen, but just in case
            classes = all_classes
        
        # Validate that requested classes exist
        missing_classes = [cls for cls in classes if cls not in all_classes]
        if missing_classes:
            raise ValueError(f"Classes not found in dataset: {missing_classes}")
        
        # Get indices of selected classes
        class_indices = [all_classes.index(cls) for cls in classes]
        
        # Filter datasets to only include selected classes
        train_indices = [i for i, (_, class_idx) in enumerate(train_dataset.samples) 
                        if class_idx in class_indices]
        val_indices = [i for i, (_, class_idx) in enumerate(val_dataset.samples) 
                      if class_idx in class_indices]
        
        # Create filtered datasets
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        
        # Update classes to only include selected ones
        selected_classes = classes
    else:
        selected_classes = train_dataset.classes

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader, selected_classes
