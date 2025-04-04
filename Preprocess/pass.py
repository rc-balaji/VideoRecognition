import os
import ffmpeg

# List of common video file extensions
video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv"}


def get_video_info(file_path):
    """Get video duration and size using ffmpeg."""
    try:
        probe = ffmpeg.probe(file_path)
        duration = float(probe["format"]["duration"])
        size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        return round(size, 2), round(duration, 2)
    except Exception as e:
        return None, None


def list_files_and_folders(directory, level=0):
    """Recursively list all files and folders."""
    indent = "  " * level  # Indentation for hierarchy visualization
    print(f"{indent}[Folder] {directory}")

    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                list_files_and_folders(
                    item_path, level + 1
                )  # Recursive call for subdirectories
            elif os.path.isfile(item_path):
                file_ext = os.path.splitext(item)[1].lower()
                if file_ext in video_extensions:
                    size, duration = get_video_info(item_path)
                    if size and duration:
                        print(
                            f"{indent}  [Video] {item} - Size: {size} MB, Duration: {duration} sec"
                        )
                    else:
                        print(f"{indent}  [Video] {item} - Unable to retrieve info")
                else:
                    print(f"{indent}  [File] {item}")
    except PermissionError:
        print(f"{indent}  [Access Denied] {directory}")


# Run the function on the current directory
current_directory = os.getcwd()
list_files_and_folders(current_directory)
