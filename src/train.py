import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 5,
    lr: float = 0.001,
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
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_acc", epoch_acc, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

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
