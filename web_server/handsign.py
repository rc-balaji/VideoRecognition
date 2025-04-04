import os
import torch
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
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


class HandSignModel:
    def __init__(
        self,
        checkpoint_path,
        label_file="labels.txt",
        frame_count=16,
        resize=(112, 112),
    ):
        self.label_file = label_file
        self.class_names = self._load_labels()
        self.frame_count = frame_count
        self.resize = resize

        self.model = ActionRecognitionModel.load_from_checkpoint(
            checkpoint_path, num_classes=len(self.class_names)
        )
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def _load_labels(self):
        """Loads class labels from the labels.txt file."""
        if os.path.exists(self.label_file):
            with open(self.label_file, "r") as f:
                return f.read().strip().split(",")
        return []

    def _preprocess_video(self, cap):
        """Preprocesses frames from a video capture object."""
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(
            0, total_frames - 1, self.frame_count, dtype=np.int32
        )

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_indices:
                frame = cv2.resize(frame, self.resize)
                frame = frame / 255.0
                frames.append(frame)
        cap.release()

        while len(frames) < self.frame_count:
            frames.append(np.zeros((*self.resize, 3)))

        frames = np.array(frames)
        frames = torch.tensor(frames).permute(3, 0, 1, 2).float().unsqueeze(0)
        if torch.cuda.is_available():
            frames = frames.cuda()
        return frames

    def predict_from_path(self, video_path):
        """Predicts action label from a video file path."""
        if not os.path.exists(video_path):
            return f"Error: File '{video_path}' not found."

        cap = cv2.VideoCapture(video_path)
        frames = self._preprocess_video(cap)

        return self._predict(frames, video_path)

    def predict_from_request(self, request_video):
        """Predicts action label from a request object (assumed to be a file-like object)."""
        cap = cv2.VideoCapture(request_video)
        frames = self._preprocess_video(cap)

        return self._predict(frames, "Uploaded Video")

    def _predict(self, frames, video_source):
        """Runs the model to predict the action label."""
        with torch.no_grad():
            logits = self.model(frames)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs).item()

        confidence = max(probs[0].cpu().numpy()) * 100
        return {
            "video": video_source,
            "predicted_label": self.class_names[pred_class],
            "confidence": f"{confidence:.2f}%",
        }
