import os


def rename_files_in_directory(directory):
    try:
        for root, dirs, files in os.walk(directory):
            folder_name = os.path.basename(root)
            total_files = len(files)
            temp_names = []

            # Step 1: Rename all files to temporary unique names
            for index, file in enumerate(sorted(files), start=1):
                old_path = os.path.join(root, file)
                temp_name = f"__temp__{index}{os.path.splitext(file)[1]}"
                temp_path = os.path.join(root, temp_name)
                os.rename(old_path, temp_path)
                temp_names.append((temp_path, file))

            # Step 2: Rename temp files to final names (case-sensitive)
            for index, (temp_path, original_name) in enumerate(temp_names, start=1):
                file_ext = os.path.splitext(original_name)[1]
                new_name = f"{folder_name}{index}{file_ext}"
                new_path = os.path.join(root, new_name)
                os.rename(temp_path, new_path)
                print(f"Renamed: {temp_path} -> {new_path}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    directory = "dataset"
    if os.path.exists(directory):
        rename_files_in_directory(directory)
        print("All files renamed successfully!")
    else:
        print("Directory not found!")
