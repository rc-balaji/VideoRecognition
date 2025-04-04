import os
import torch
import cv2
import numpy as np
from train import ActionRecognitionModel


CHECK_POINT = "./checkpoints/hs_model.ckpt"


def get_test_videos(dataset_path, num_videos=5):
    test_videos = []
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            videos = [
                os.path.join(folder_path, f)
                for f in sorted(os.listdir(folder_path))
                if f.endswith(".mp4")
            ]
            test_videos.extend([(video, folder) for video in videos[:num_videos]])
    return test_videos


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


if __name__ == "__main__":
    dataset_path = "dataset"
    test_videos = get_test_videos(dataset_path)

    predictor = ActionPredictor(
        checkpoint_path=CHECK_POINT,
        class_names=[
            folder
            for folder in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, folder))
        ],
    )

    correct_predictions = 0
    total_tests = len(test_videos)
    test_results = []

    for video_path, actual_label in test_videos:
        pred_label, probs = predictor.predict(video_path)
        correct = pred_label == actual_label
        correct_predictions += int(correct)
        test_results.append(
            f"Video: {video_path} | Predicted: {pred_label} | Actual: {actual_label} | {'✅' if correct else '❌'}"
        )

    accuracy = (correct_predictions / total_tests) * 100
    test_results.append(f"\nTotal Test Videos: {total_tests}")
    test_results.append(f"Correct Predictions: {correct_predictions}")
    test_results.append(f"Accuracy: {accuracy:.2f}%")

    with open("test_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(test_results))

    print("\n".join(test_results))
