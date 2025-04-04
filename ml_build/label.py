import os


class Labels:
    def __init__(self, label_file="labels.txt"):
        self.dataset_path = "dataset"
        self.label_file = label_file

    def generate_labels(self):
        class_names = [
            folder
            for folder in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, folder))
        ]
        with open(self.label_file, "w") as f:
            f.write(",".join(class_names))
            print("Label created Successfull and Stored in labels.txt")

    def get_labels(self):
        if os.path.exists(self.label_file):
            with open(self.label_file, "r") as f:
                return f.read().split(",")
        return []


Labels().generate_labels()
