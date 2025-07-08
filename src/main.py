import torch

from src.dataset import get_data_loaders
from src.model import SimpleCNN
from src.train import train


def main():
    data_dir = "data/Fruit_dataset"  # adjust as needed
    batch_size = 64
    epochs = 10

    train_loader, val_loader, classes = get_data_loaders(data_dir, batch_size)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleCNN(num_classes=len(classes)).to(device)

    trained_model = train(model, train_loader, val_loader, device, epochs=epochs)

    torch.save(trained_model.state_dict(), "simple_cnn.pth")
    print("Model saved!")


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
