import cv2
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.preprocessing import image
import pandas as pd
from pathlib import Path
import sys


root_dir = Path.cwd().resolve().parent.parent
sys.path.append(str(root_dir))
from config.variables import text_path, json_file, csv_file, textcsv_file

def load_images(num_frames=8, img_height=224, img_width=224):
    """
    Load images from frames directory.
    """
    dir = os.path.join(root_dir, 'data', 'inputs', 'frames')
    csv_path = os.path.join(root_dir, csv_file)
    df = pd.read_csv(csv_path, dtype={'id': str})
    X = []
    y = []

    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        id_video = row['id']
        print(id_video)
        frames = []
        
        for j in range(num_frames):
            img_path = os.path.join(dir, f"{id_video}_frame_{j}.png")
            if os.path.exists(img_path):
                img = image.load_img(img_path, target_size=(img_height, img_width))
                img_array = image.img_to_array(img)
                frames.append(img_array)

        if len(frames) == num_frames:
            X.append(frames)
            y.append(row['norm_virality'])

    X = np.array(X) / 255.0
    y = np.array(y)

    return X, y


def train_and_metrics_pretrained(model, train_loader, optimizer, device, mse_metric, rmse_metric, mae_metric):
    """
    Train the model and compute metrics on training data.
    """
    model.train()
    total_loss = 0
    mse_metric.reset()
    rmse_metric.reset()
    mae_metric.reset()

    for batch in tqdm(train_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        outputs = model(batch['image'].to(device))
        outputs = torch.squeeze(outputs)
        targets = batch['label'].to(device)
        loss = nn.functional.mse_loss(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        mse_metric(outputs, targets)
        rmse_metric(outputs, targets)
        mae_metric(outputs, targets)

    mean_loss = total_loss / len(train_loader)
    mse = mse_metric.compute()
    rmse = rmse_metric.compute()
    mae = mae_metric.compute()
    return mean_loss, mse.item(), rmse.item(), mae.item()


def validate_pretrained(model, loader, device, mse_metric, rmse_metric, mae_metric):
    """
    Validate the model and compute metrics on validation data.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    mse_metric.reset()
    rmse_metric.reset()
    mae_metric.reset()

    # Make sure the metric runs on the appropriate device
    mse_metric.to(device)
    rmse_metric.to(device)
    mae_metric.to(device)

    with torch.no_grad():  # Instruct PyTorch not to manage gradients during validation
        for batch in tqdm(loader, desc="Validation", leave=False):
            inputs = batch['image'].to(device)  # Send inputs to GPU
            targets = batch['label'].to(device)  # Send labels to GPU
            outputs = model(inputs)
            outputs = torch.squeeze(outputs)  # Adjust dimensions if necessary
            
            loss = nn.functional.mse_loss(outputs, targets)
            total_loss += loss.item()
            
            mse_metric(outputs, targets)
            rmse_metric(outputs, targets)
            mae_metric(outputs, targets)

    mean_loss = total_loss / len(loader)
    mse = mse_metric.compute()
    rmse = rmse_metric.compute()
    mae = mae_metric.compute()
    return mean_loss, mse.item(), rmse.item(), mae.item()

def train_and_metrics_3dcnn(model, data_loader, optimizer, device, mse, rmse, mae):
    """
    Train 3D CNN model and compute metrics.
    """
    model.train()
    total_loss, total_mse, total_rmse, total_mae, count = 0, 0, 0, 0, 0
    progress_bar = tqdm(data_loader, desc='Training', leave=False)
    for data in progress_bar:
        inputs = data['image'].to(device)
        labels = data['label'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.squeeze()
        loss = F.mse_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update progress bar with loss information
        progress_bar.set_postfix({'loss': loss.item()})

        total_loss += loss.item() * inputs.size(0)
        total_mse += mse(outputs, labels).item() * inputs.size(0)
        total_rmse += rmse(outputs, labels).item() * inputs.size(0)
        total_mae += mae(outputs, labels).item() * inputs.size(0)
        count += inputs.size(0)
    
    return total_loss / count, total_mse / count, total_rmse / count, total_mae / count

def validate_3dcnn(model, data_loader, device, mse, rmse, mae):
    """
    Validate 3D CNN model and compute metrics.
    """
    model.eval()
    total_loss, total_mse, total_rmse, total_mae, count = 0, 0, 0, 0, 0
    progress_bar = tqdm(data_loader, desc='Validation', leave=False)
    with torch.no_grad():
        for data in progress_bar:
            inputs = data['image'].to(device)
            labels = data['label'].to(device)

            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = F.mse_loss(outputs, labels)

            # Update progress bar
            progress_bar.set_postfix({'val_loss': loss.item()})

            total_loss += loss.item() * inputs.size(0)
            total_mse += mse(outputs, labels).item() * inputs.size(0)
            total_rmse += rmse(outputs, labels).item() * inputs.size(0)
            total_mae += mae(outputs, labels).item() * inputs.size(0)
            count += inputs.size(0)
    
    return total_loss / count, total_mse / count, total_rmse / count, total_mae / count

def rmse_3dcnn(outputs, labels):
    """
    Compute RMSE for 3D CNN.
    """
    return torch.sqrt(F.mse_loss(outputs, labels))

def mae_3dcnn(outputs, labels):
    """
    Compute MAE for 3D CNN.
    """
    return F.l1_loss(outputs, labels)

def try_capture_frame(cap, frame_id):
    """
    Try to capture a frame from video.
    """
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
    """
    Extract frames from videos.
    """
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
    """
    Remove mp4 extensions from frame filenames.
    """
    frame_dir = '../../data/inputs/frames'
    files = os.listdir(frame_dir)
    for file in files:
        if '_frame_' in file and file.endswith('.png'):
            new_name = file.replace('.mp4', '')
            os.rename(os.path.join(frame_dir, file), os.path.join(frame_dir, new_name))
            print(f"Rename file from {file} to {new_name}")

def rename_frames_ids():
    """
    Rename frames IDs to make them consecutive.
    """
    frame_dir = '../../data/inputs/frames'
    frames_dict = {}
    
    # Collect all file names by ID
    for filename in os.listdir(frame_dir):
        if filename.endswith(".png"):
            parts = filename.split('_')
            id_video = parts[0]
            if id_video in frames_dict:
                frames_dict[id_video].append(filename)
            else:
                frames_dict[id_video] = [filename]

    # Rename files to be consecutive for each ID
    for id_video, filenames in frames_dict.items():
        # Sort file names by current frame number
        filenames.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        
        # Rename the files
        for i, filename in enumerate(filenames):
            nuevo_nombre = f"{id_video}_frame_{i}.png"
            os.rename(os.path.join(frame_dir, filename), os.path.join(frame_dir, nuevo_nombre))
            print(f"Renamed: {filename} to {nuevo_nombre}")

