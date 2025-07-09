import mlflow
import torch

from src.dataset import get_data_loaders
from src.model import ResNet50
from src.train import train


def run_augmentation_experiment():
    """Run experiments with different augmentation settings."""

    data_dir = "data/Fruit_dataset"
    batch_size = 32
    epochs = 25

    # Experiment configurations
    experiments = [
        # {"name": "no-augmentation", "use_aug": False, "strength": None},
        # {"name": "light-augmentation", "use_aug": True, "strength": "light"},
        # {"name": "medium-augmentation", "use_aug": True, "strength": "medium"},
        {"name": "heavy-augmentation", "use_aug": True, "strength": "heavy"},
    ]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    mlflow.set_tracking_uri("http://localhost:8080")
    mlflow.set_experiment("fruit-classification-resnet-50")
    mlflow.pytorch.autolog()

    for exp in experiments:
        print(f"\n{'=' * 50}")
        print(f"Running experiment: {exp['name']}")
        print(f"{'=' * 50}")

        # Get data loaders with current augmentation settings
        train_loader, val_loader, classes = get_data_loaders(
            data_dir,
            batch_size,
            use_augmentation=exp["use_aug"],
            augmentation_strength=exp["strength"] if exp["use_aug"] else "medium",
        )

        # Create model
        model = ResNet50(num_classes=len(classes)).to(device)

        # Train model
        trained_model = train(
            model,
            train_loader,
            val_loader,
            device,
            epochs=epochs,
            run_name=exp["name"],
            use_augmentation=exp["use_aug"],
            augmentation_strength=exp["strength"] if exp["use_aug"] else "medium",
        )

        # Save model
        model_path = f"models/{exp['name']}.pth"
        torch.save(trained_model.state_dict(), model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    run_augmentation_experiment()
