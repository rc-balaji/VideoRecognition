import torch
import cv2
import numpy as np
from torchvision.models.video import r3d_18
from train import ActionRecognitionModel


class ActionPredictor:
    def __init__(self, checkpoint_path, class_names, frame_count=16, resize=(112, 112)):
        self.class_names = class_names
        self.frame_count = frame_count
        self.resize = resize

        # Load model from checkpoint
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

        # Convert to tensor and reshape
        frames = np.array(frames)
        frames = (
            torch.tensor(frames).permute(3, 0, 1, 2).float().unsqueeze(0)
        )  # Add batch dim

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


if __name__ == "__main__":
    # Example usage
    from data_preparation import VideoDataset

    # Get class names
    dataset = VideoDataset("dataset")
    _, _, _, _, classes = dataset.prepare_dataset()

    # Initialize predictor with best checkpoint
    predictor = ActionPredictor(
        checkpoint_path="checkpoints/action-recog-epoch=06-val_acc=1.00.ckpt",
        class_names=classes,
    )

    # Test on a sample video
    video_path = "dataset/indian_sign_language/indian_sign_language1.mp4"
    pred_class, probs = predictor.predict(video_path)

    print(f"Predicted class: {pred_class}")
    print("Class probabilities:")
    for class_name, prob in zip(classes, probs):
        print(f"{class_name}: {prob:.4f}")
