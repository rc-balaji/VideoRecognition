import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.models.video import r3d_18, R3D_18_Weights

# Enable Tensor Cores optimization for compatible GPUs
torch.set_float32_matmul_precision("high")


class ActionRecognitionModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained 3D ResNet
        self.model = r3d_18(weights=R3D_18_Weights.DEFAULT)

        # Replace last FC layer for action classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # x: (batch, channels, frames, height, width)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)

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
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def train_model(X_train, y_train, X_test, y_test, classes, epochs=10, batch_size=8):
    # Convert and reshape: (N, T, H, W, C) -> (N, C, T, H, W)
    X_train = torch.tensor(X_train).permute(0, 4, 1, 2, 3).float()
    y_train = torch.tensor(y_train).long()

    X_test = torch.tensor(X_test).permute(0, 4, 1, 2, 3).float()
    y_test = torch.tensor(y_test).long()

    # Data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        num_workers=4,
    )

    # Model
    model = ActionRecognitionModel(num_classes=len(classes))

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        dirpath="checkpoints/",
        filename="action-recog-{epoch:02d}-{val_acc:.2f}",
    )

    # Logger
    logger = TensorBoardLogger("lightning_logs", name="action_recognition")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
    )

    trainer.fit(model, train_loader, val_loader)

    return model


if __name__ == "__main__":
    from data_preparation import VideoDataset

    # Prepare dataset
    dataset = VideoDataset("dataset")
    X_train, X_test, y_train, y_test, classes = dataset.prepare_dataset()

    # Train
    train_model(X_train, y_train, X_test, y_test, classes, epochs=20)
