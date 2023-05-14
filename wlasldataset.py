from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import cv2
import numpy as np
import torch
from PIL import Image

IW = 112
IH = 112

frame_transform = transforms.Compose([
        transforms.RandomResizedCrop(100),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=(0.1,0.6), contrast=1,saturation=0, hue=0.4),
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class WLASL100(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file, dtype=str)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        '''
        returns length of the dataset
        '''
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        returns a stack of frames and its labels for a given index in the dataet
        '''
        video_id = self.data['id'][idx]
        label = self.data['label'][idx]
        labels = np.zeros(25)
        labels[int(label)] = 1.
        frames = get_frames(self.root_dir + f'/{video_id}.mp4')
        for frame in frames:
            frame = Image.fromarray(frame)
            frame_transform(frame)
        frames = torch.stack([transforms.functional.to_tensor(frame) for frame in frames])
        frames = frames.permute(1,0,2,3)
        labels = torch.as_tensor(labels)
        return frames, labels


def get_frames(filename, n_frames = 25, resize = (IW, IH)):
    '''
    function to format the input video into an array of resized frames
    '''
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    count = 0
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if resize:
                frame = cv2.resize(frame, resize)
                frames.append(frame)
            if len(frames) == n_frames:
                break
    finally:
        cap.release()
        
    return np.array(frames)
