import cv2
import os
import numpy as np



def extract_frames():
    # Directorio donde están los videos
    video_dir = '../../data/inputs/videos'
    # Directorio donde guardar los frames
    frame_dir = '../../data/inputs/frames'
    os.makedirs(frame_dir, exist_ok=True)

    # Obtener lista de archivos de video
    videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]  # Asegúrate de incluir aquí los formatos de tus videos

    for video in videos:
        video_path = os.path.join(video_dir, video)
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_ids = [round(i) for i in np.linspace(0, frame_count - 1, 10)]  # Generar 10 puntos equidistantes
        
        for frame_id in frame_ids:
            # Establecer la posición actual del video
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                # Crear el nombre del archivo de salida
                frame_filename = f"{video}_frame_{frame_id}.png"
                frame_path = os.path.join(frame_dir, frame_filename)
                # Guardar frame como imagen
                cv2.imwrite(frame_path, frame)
                print(f"Saved: {frame_path}")
            else:
                print("Error capturing frame:", frame_id)
        
        # Liberar el capturador de video
        cap.release()

    print("Frames extracted")

def remove_mp4_extensions():
    frame_dir = '../../data/inputs/frames'
    files = os.listdir(frame_dir)
    for file in files:
        if '_frame_' in file and file.endswith('.png'):
            new_name = file.replace('.mp4', '')
            os.rename(os.path.join(frame_dir, file), os.path.join(frame_dir, new_name))
            print(f"Archivo renombrado de {file} a {new_name}")
