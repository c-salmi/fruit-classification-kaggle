import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.dataset import get_data_loaders
from src.model import SimpleCNN



def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 5,
    lr: float = 0.0001,
    run_name: str = "baseline-cnn",
    use_augmentation: bool = True,
    augmentation_strength: str = "medium",
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 🔑 Start an MLflow run
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("model_type", model.__class__.__name__)
        mlflow.log_param("use_augmentation", use_augmentation)
        if use_augmentation:
            mlflow.log_param("augmentation_strength", augmentation_strength)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]")
            for inputs, labels in loop:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / total
            epoch_acc = correct / total

            val_acc, val_loss = evaluate(model, val_loader, criterion, device)

            # 🔑 Log metrics per epoch
            s = epoch*len(train_loader)
            mlflow.log_metric("train_loss_epoch", epoch_loss, step=s)
            mlflow.log_metric("train_acc_epoch", epoch_acc, step=s)
            mlflow.log_metric("val_acc", val_acc, step=s)
            mlflow.log_metric("val_loss", val_loss, step=s)

            loop.set_postfix(
                train_loss=epoch_loss, train_acc=epoch_acc, val_acc=val_acc
            )

        # mlflow.pytorch.log_model(model, f"models-{run_name}", input_example=inputs)

        print("Training complete. Final model logged to MLflow.")

    return model


# Optionally:
def evaluate(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    loss = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += loss.item()

    return correct / total, loss / len(val_loader)


def main():
    data_dir = "data/Fruit_dataset"  # adjust as needed
    batch_size = 64
    epochs = 20
    experiment_name = "fruit-classification-simple-cnn"
    num_classes = 20

    # Augmentation parameters
    use_augmentation = False
    augmentation_strength = "medium"  # "light", "medium", or "heavy"

    train_loader, val_loader, classes = get_data_loaders(
        data_dir,
        batch_size,
        use_augmentation=use_augmentation,
        augmentation_strength=augmentation_strength,
        num_classes=num_classes,
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
    mlflow.set_experiment(experiment_name)

    model = SimpleCNN(num_classes=num_classes).to(device)

    # Create run name based on augmentation settings
    run_name = f"cnn-aug-{augmentation_strength}-low-lr"

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