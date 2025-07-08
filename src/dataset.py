from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(data_dir: str, batch_size: int = 32):
    train_transforms = transforms.Compose(
        [
            transforms.Resize((64, 64)),  # Resize to fixed size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

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
    return train_loader, val_loader, train_dataset.classes
