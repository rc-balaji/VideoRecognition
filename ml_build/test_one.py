import os
import torch
import cv2
import numpy as np
from train import ActionRecognitionModel


class ActionPredictor:
    def __init__(self, checkpoint_path, class_names, frame_count=16, resize=(112, 112)):
        self.class_names = class_names
        self.frame_count = frame_count
        self.resize = resize

        self.model = ActionRecognitionModel.load_from_checkpoint(
            checkpoint_path, num_classes=len(class_names)
        )
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def preprocess_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
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

    def predict(self, video_path):
        frames = self.preprocess_video(video_path)
        with torch.no_grad():
            logits = self.model(frames)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs).item()
        return self.class_names[pred_class], probs[0].cpu().numpy()


label_file = "labels.txt"


def get_labels():
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            return f.read().split(",")
    return []


if __name__ == "__main__":
    class_names = get_labels()

    predictor = ActionPredictor(
        checkpoint_path="checkpoints/action-recog-epoch=06-val_acc=1.00.ckpt",
        class_names=class_names,
    )

    video_path = "hell_me.mp4"

    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' not found.")
    else:
        pred_label, probs = predictor.predict(video_path)
        confidence = max(probs) * 100

        print("\nPrediction Result:")
        print(f"Video: {video_path}")
        print(f"Predicted Label: {pred_label}")
        print(f"Confidence: {confidence:.2f}%")
