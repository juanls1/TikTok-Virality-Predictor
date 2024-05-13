import os
from moviepy.editor import VideoFileClip
import sys
from pathlib import Path

# Get the absolute path of the folder containing the module
root_dir = Path(__file__).resolve().parent.parent.parent

# Add the folder path to sys.path
sys.path.append(str(root_dir))

def extract_audios():
    # Create the "audios" folder inside the compressed folder if it doesn't exist
    audios_dir = os.path.join(root_dir, "data\\inputs\\audios")
    if not os.path.exists(audios_dir):
        os.makedirs(audios_dir)

    # Iterate over the video files and extract the audios
    for video_file in os.listdir(os.path.join(root_dir, "data\\inputs\\Raw\\videos")):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(root_dir, "data\\inputs\\Raw\\videos", video_file)
            audio_path = os.path.join(audios_dir, video_file.replace(".mp4", "_audio.mp3"))
            print(f"Extracting audio from {video_file}")
            video_clip = VideoFileClip(video_path)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(audio_path)
            audio_clip.close()
            video_clip.close()

# Call the function to extract the audios
extract_audios()
