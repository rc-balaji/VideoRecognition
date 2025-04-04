import os
import subprocess

# Define base input and output paths
base_path = "./dataset"  # Change this if needed
output_base = "./slow_data/"  # Output directory

# Ensure the output directory exists
os.makedirs(output_base, exist_ok=True)


# Function to process videos
def slow_down_videos(base_path, output_base):
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        output_folder = os.path.join(output_base, folder)

        if os.path.isdir(folder_path):
            os.makedirs(output_folder, exist_ok=True)  # Create output folder

            for file in os.listdir(folder_path):
                if file.endswith(".mp4"):  # Process only MP4 files
                    input_video = os.path.join(folder_path, file)
                    output_video = os.path.join(output_folder, file)

                    # FFmpeg command to slow down video (0.75x speed)
                    cmd = [
                        "ffmpeg",
                        "-i",
                        input_video,
                        "-filter:v",
                        "setpts=1.3333*PTS",  # 1/0.75 = 1.3333
                        "-c:a",
                        "aac",
                        "-b:a",
                        "128k",  # Maintain audio quality
                        "-movflags",
                        "+faststart",
                        output_video,
                    ]

                    print(f"Processing: {input_video} -> {output_video}")
                    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print("✅ All videos processed and saved in 'slow/' directory!")


# Run the function
slow_down_videos(base_path, output_base)
