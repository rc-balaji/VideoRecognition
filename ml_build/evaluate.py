import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from train import ActionRecognitionModel
from data_preparation import VideoDataset


CHECK_POINT = "./checkpoints/hs_model.ckpt"


def evaluate_model(checkpoint_path, dataset_path):
    # Load data
    dataset = VideoDataset(dataset_path)
    X_train, X_test, y_train, y_test, classes = dataset.prepare_dataset()

    # Convert to tensors
    X_test = torch.tensor(X_test).permute(0, 4, 1, 2, 3).float()
    y_test = torch.tensor(y_test).long()

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Load model
    model = ActionRecognitionModel.load_from_checkpoint(
        checkpoint_path, num_classes=len(classes)
    )
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    # Get predictions
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Generate report
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    evaluate_model(
        checkpoint_path=CHECK_POINT,
        dataset_path="dataset",
    )
