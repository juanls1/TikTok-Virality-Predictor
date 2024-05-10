import cv2
import os
import numpy as np



def try_capture_frame(cap, frame_id):
    while frame_id >= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            return frame, True, frame_id
        print(f"Error capturing frame: {frame_id}, trying previous frame")
        frame_id -= 1
    print("No valid frames available to capture.")
    return None, False, -1

def extract_frames(n_frames):
    video_dir = '../../data/inputs/videos'
    frame_dir = '../../data/inputs/frames'

    os.makedirs(frame_dir, exist_ok=True)

    videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]

    for video in videos:
        video_name = video.split('.')[0]
        video_path = os.path.join(video_dir, video)
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_ids = [round(i) for i in np.linspace(0, frame_count - 1, n_frames)]
        success_count = 0  
        for count, frame_id in enumerate(frame_ids):
            frame, success, id = try_capture_frame(cap, frame_id)
            if success:
                frame_filename = f"{video_name}_frame_{count}.png"
                frame_path = os.path.join(frame_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                print(f"Saved: {frame_path} | Frame ID: {id}")
                success_count += 1

        cap.release()

        if success_count != n_frames:
            missing_frames = [i for i in range(n_frames) if i not in range(success_count)]
            print(f"Warning: Not all frames were extracted from {video}. Missing frames: {missing_frames}")

    print("Frames extraction completed")

def remove_mp4_extensions():
    frame_dir = '../../data/inputs/frames'
    files = os.listdir(frame_dir)
    for file in files:
        if '_frame_' in file and file.endswith('.png'):
            new_name = file.replace('.mp4', '')
            os.rename(os.path.join(frame_dir, file), os.path.join(frame_dir, new_name))
            print(f"Archivo renombrado de {file} a {new_name}")

def rename_frames_ids():
    frame_dir = '../../data/inputs/frames'
    frames_dict = {}
    
    # Recolectar todos los nombres de archivo por ID
    for filename in os.listdir(frame_dir):
        if filename.endswith(".png"):
            parts = filename.split('_')
            id_video = parts[0]
            if id_video in frames_dict:
                frames_dict[id_video].append(filename)
            else:
                frames_dict[id_video] = [filename]

    # Renombrar los archivos para que sean consecutivos por cada ID
    for id_video, filenames in frames_dict.items():
        # Ordenar los nombres de archivo por el número de frame actual
        filenames.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        
        # Renombrar los archivos
        for i, filename in enumerate(filenames):
            nuevo_nombre = f"{id_video}_frame_{i}.png"
            os.rename(os.path.join(frame_dir, filename), os.path.join(frame_dir, nuevo_nombre))
            print(f"Renombrado: {filename} a {nuevo_nombre}")

