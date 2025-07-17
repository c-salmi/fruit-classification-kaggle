import lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import mlflow


class ResNetLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for ResNet training with MLflow integration.
    """
    
    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        scheduler_gamma: float = 0.1,
        scheduler_step_size: int = 7,
        model_name: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        unfreeze_epoch: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model setup
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_step_size = scheduler_step_size
        self.freeze_backbone = freeze_backbone
        self.unfreeze_epoch = unfreeze_epoch
        
        # Initialize model
        self._setup_model(model_name, pretrained)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def _setup_model(self, model_name: str, pretrained: bool):
        """Setup the ResNet model with specified architecture."""
        if model_name == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
        elif model_name == "resnet34":
            self.model = models.resnet34(pretrained=pretrained)
        elif model_name == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
        elif model_name == "resnet101":
            self.model = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Replace the final layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.num_classes)
        
        # Freeze backbone if specified
        if self.freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all layers except the final classifier."""
        for name, param in self.model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    
    def _unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.model(x)
    
    def _calculate_accuracy(self, outputs, targets):
        """Calculate accuracy manually."""
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        return correct / total
    
    def training_step(self, batch, batch_idx):
        """Training step with loss and accuracy computation."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        acc = self._calculate_accuracy(logits, y)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with loss and accuracy computation."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        acc = self._calculate_accuracy(logits, y)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step with loss and accuracy computation."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        acc = self._calculate_accuracy(logits, y)
        
        # Log metrics
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', acc, on_epoch=True)
        
        return loss
    
    def on_train_epoch_start(self):
        """Unfreeze backbone at specified epoch for fine-tuning."""
        if self.freeze_backbone and self.current_epoch == self.unfreeze_epoch:
            self._unfreeze_backbone()
            print(f"Unfroze backbone at epoch {self.current_epoch}")
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        } 