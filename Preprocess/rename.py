import os


def rename_files_in_directory(directory):
    try:
        for root, dirs, files in os.walk(directory):
            folder_name = os.path.basename(root)
            total_files = len(files)
            for index, file in enumerate(sorted(files), start=1):
                file_ext = os.path.splitext(file)[1]
                new_name = f"{folder_name}{str(index)}{file_ext}"
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    directory = "dataset"
    if os.path.exists(directory):
        rename_files_in_directory(directory)
        print("All files renamed successfully!")
    else:
        print("Directory not found!")
