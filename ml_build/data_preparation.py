import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


class VideoDataset:
    def __init__(self, root_dir, frame_count=16, resize=(112, 112)):
        self.root_dir = root_dir
        self.frame_count = frame_count
        self.resize = resize
        self.classes = []
        self.video_paths = []
        self.labels = []

        # Discover classes and videos
        self._discover_classes()

    def _discover_classes(self):
        self.classes = [
            d
            for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ]
        self.classes.sort()

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            videos = [f for f in os.listdir(class_dir) if f.endswith(".mp4")]

            for video in videos:
                self.video_paths.append(os.path.join(class_dir, video))
                self.labels.append(label)

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame indices to sample
        frame_indices = np.linspace(
            0, total_frames - 1, self.frame_count, dtype=np.int32
        )

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_indices:
                # Resize and normalize
                frame = cv2.resize(frame, self.resize)
                frame = frame / 255.0
                frames.append(frame)

        cap.release()

        # Pad if we didn't get enough frames
        while len(frames) < self.frame_count:
            frames.append(np.zeros((*self.resize, 3)))

        return np.array(frames)

    def prepare_dataset(self, test_size=0.2, random_state=42):
        X = []
        y = []

        for path, label in zip(self.video_paths, self.labels):
            frames = self._load_video_frames(path)
            X.append(frames)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        return X_train, X_test, y_train, y_test, self.classes


if __name__ == "__main__":
    dataset = VideoDataset("dataset")
    X_train, X_test, y_train, y_test, classes = dataset.prepare_dataset()

    print(f"Classes: {classes}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
