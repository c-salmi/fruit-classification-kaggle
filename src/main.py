import mlflow
import torch

from src.dataset import get_data_loaders
from src.model import SimpleCNN
from src.train import train


def main():
    data_dir = "data/Fruit_dataset"  # adjust as needed
    batch_size = 64
    epochs = 10

    # Augmentation parameters
    use_augmentation = True
    augmentation_strength = "medium"  # "light", "medium", or "heavy"

    train_loader, val_loader, classes = get_data_loaders(
        data_dir,
        batch_size,
        use_augmentation=use_augmentation,
        augmentation_strength=augmentation_strength,
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    mlflow.set_tracking_uri("http://localhost:8080")
    mlflow.set_experiment("fruit-classification")
    mlflow.pytorch.autolog()

    model = SimpleCNN(num_classes=len(classes)).to(device)

    # Create run name based on augmentation settings
    run_name = f"cnn-aug-{augmentation_strength}" if use_augmentation else "cnn-no-aug"

    trained_model = train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=epochs,
        run_name=run_name,
        use_augmentation=use_augmentation,
        augmentation_strength=augmentation_strength,
    )

    torch.save(trained_model.state_dict(), "models/simple_cnn.pth")
    print("Model saved!")


if __name__ == "__main__":
    main()
