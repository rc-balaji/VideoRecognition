import os


def describe_directory(directory, indent=0):
    try:
        items = os.listdir(directory)
        for item in items:
            path = os.path.join(directory, item)
            if os.path.isdir(path):
                print("  " * indent + f"📂 {item}")
                describe_directory(path, indent + 1)
            else:
                print("  " * indent + f"📄 {item}")
    except PermissionError:
        print("  " * indent + "🚫 [Permission Denied]")


if __name__ == "__main__":
    directory = "dataset"
    if os.path.exists(directory):
        print(f"Directory Structure of: {directory}\n")
        describe_directory(directory)
    else:
        print("Directory not found!")
