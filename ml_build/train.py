import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.models.video import r3d_18, R3D_18_Weights


class ActionRecognitionModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained model
        self.model = r3d_18(weights=R3D_18_Weights.DEFAULT)

        # Replace the last layer for our number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # Input shape: (batch, channels, frames, height, width)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


def train_model(X_train, y_train, X_test, y_test, classes, epochs=10, batch_size=8):
    # Convert numpy arrays to PyTorch tensors
    # Need to reshape to (batch, channels, frames, height, width)
    X_train = torch.tensor(X_train).permute(0, 4, 1, 2, 3).float()
    y_train = torch.tensor(y_train).long()

    X_test = torch.tensor(X_test).permute(0, 4, 1, 2, 3).float()
    y_test = torch.tensor(y_test).long()

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    model = ActionRecognitionModel(num_classes=len(classes))

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        dirpath="checkpoints/",
        filename="action-recog-{epoch:02d}-{val_acc:.2f}",
    )

    # Train
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
    )

    trainer.fit(model, train_loader, val_loader)

    return model


if __name__ == "__main__":
    from data_preparation import VideoDataset

    # Prepare data
    dataset = VideoDataset("dataset")
    X_train, X_test, y_train, y_test, classes = dataset.prepare_dataset()

    # Train model
    train_model(X_train, y_train, X_test, y_test, classes, epochs=20)
