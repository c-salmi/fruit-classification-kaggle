import argparse
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger
import mlflow

from src.dataset import get_data_loaders
from src.lightning_model import ResNetLightningModule


def main():
    print("starting training")
    parser = argparse.ArgumentParser(description="Train ResNet with PyTorch Lightning")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--model_name", type=str, default="resnet50", 
                       choices=["resnet18", "resnet34", "resnet50", "resnet101"],
                       help="ResNet model to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone initially")
    parser.add_argument("--unfreeze_epoch", type=int, default=5, help="Epoch to unfreeze backbone")
    parser.add_argument("--no_augmentation", action="store_true", help="Disable augmentation")
    parser.add_argument("--augmentation_strength", type=str, default="medium",
                       choices=["light", "medium", "heavy"], help="Augmentation strength")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Training precision")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--experiment_name", type=str, default="fruit-classification-lightning", help="Experiment name")
    args = parser.parse_args()
    print("args: ", args)
    
    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:8080")
    mlflow.set_experiment(args.experiment_name)

    print(" args: ", args)
    
    # Get data loaders
    train_loader, val_loader, classes = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_augmentation=not args.no_augmentation,
        augmentation_strength=args.augmentation_strength,
        num_classes=args.num_classes,
    )
        
    # Initialize the Lightning module
    model = ResNetLightningModule(
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        model_name=args.model_name,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
        unfreeze_epoch=args.unfreeze_epoch,
    )
        
    # Set up callbacks
    callbacks = [
        # Model checkpointing
        ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_top_k=3,
            filename="{epoch:02d}-{val_acc:.3f}",
            save_last=True,
        ),
        # Early stopping
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.patience,
            verbose=True,
        ),
        # Learning rate monitoring
        LearningRateMonitor(logging_interval="epoch"),
    ]
    
    # Set up loggers
    loggers = [
        MLFlowLogger(experiment_name=args.experiment_name),
    ]

    print("starting training")
        
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate twice per epoch
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
    )
        
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Test the model (if test data is available)
    try:
        test_loader, _, _ = get_data_loaders(
            data_dir=args.data_dir.replace("train1", "test1"),
            batch_size=args.batch_size,
            use_augmentation=False,
        )
        trainer.test(model, test_loader)
    except:
        print("No test data found, skipping test evaluation.")
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()