from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    use_augmentation: bool = True,
    augmentation_strength: str = "medium",
):
    """
    Get data loaders with optional augmentation.

    Args:
        data_dir: Directory containing train1 and val1 folders
        batch_size: Batch size for data loaders
        use_augmentation: Whether to apply augmentation to training data
        augmentation_strength: "light", "medium", or "heavy" augmentation
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
                transforms.Resize((64, 64)),  # Resize to fixed size
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
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    # Validation transforms (no augmentation for proper evaluation)
    val_transforms = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train1", transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        root=f"{data_dir}/val1", transform=val_transforms
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader, train_dataset.classes
