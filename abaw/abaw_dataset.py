import cv2
import imageio_ffmpeg
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
import time
import pickle
import timm
from transformers import AutoProcessor, Wav2Vec2FeatureExtractor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
import soundfile as sf
import abaw.utils
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence
from pathlib import Path
from transformers import AutoTokenizer
import audiomentations
cv2.setNumThreads(2)


class HumeDatasetTrain(Dataset, abaw.utils.AverageMeter):

    def __init__(self, data_folder, label_file=None, model=None):
        super().__init__()
        self.data_folder = data_folder
        self.label_file = pd.read_csv(label_file)
        self.vision_model = model[0]
        self.audio_model = model[1]
        self.text_model = model[2]

        self.wave_transforms = audiomentations.Compose([
                            audiomentations.AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.15, p=0.5),
                            audiomentations.TimeStretch(min_rate=0.95, max_rate=1.05, p=0.5),
                            audiomentations.PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
                            audiomentations.Shift(min_shift=-0.1, max_shift=0.1, p=0.5),
                            ])
        if self.vision_model != 'linear':
            self.data_config = timm.data.resolve_model_data_config(self.vision_model)
            self.transform = A.Compose([
                A.Resize(height=self.data_config['input_size'][1], width=self.data_config['input_size'][2]),
                A.Normalize(mean=self.data_config['mean'], std=self.data_config['std']),
                ToTensorV2(),
            ])

        if self.audio_model != 'linear':
            if "audeering" in self.audio_model:
                self.processor = AutoProcessor.from_pretrained(self.audio_model)
            else:
                self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.audio_model)
            self.processor_vision = AutoProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.processor_text = AutoTokenizer.from_pretrained(self.text_model)

    def __getitem__(self, index):
        row = self.label_file.iloc[index]

        if self.vision_model == 'linear':
            #vision = torch.randn(1024)
            try:
                vit_file_path = f"{self.data_folder}googlevit/{str(int(row['Filename'])).zfill(5)}.pkl"
                with open(vit_file_path, 'rb') as file:
                    data = pickle.load(file)
                    if isinstance(data, torch.Tensor):
                        tensor = data
                    else:
                        tensor = torch.tensor(data)
                    length = tensor.size(0)
                    max_length = 400
                    if length < max_length:
                        pad_size = (0, 0, 0, max_length - length)
                        vision = torch.nn.functional.pad(tensor, pad_size)
                    else:
                        vision = tensor[:max_length]
                        length = 400
            except Exception as e:
                print(e)
                vision = torch.randn(768)
        else:
            vision = self.process_images(index)
        
        if self.audio_model == 'linear':
            wav2vec2_file_path = f"{self.data_folder}wav2vec2/{str(int(row['Filename'])).zfill(5)}.pkl"
            with open(wav2vec2_file_path, 'rb') as file:
                audio = torch.mean(torch.tensor(pickle.load(file)), dim=0)
        else:
            audio = self.process_audio(row['Filename'])
        labels = torch.tensor(
            row[['Admiration', 'Amusement', 'Determination', 'Empathic Pain', 'Excitement', 'Joy']].values,
            dtype=torch.float)
        text = self.process_text(row['Filename'])
        return audio, vision, torch.tensor(length).long(), text, labels, self.avg

    def process_images(self, index):
        try:
            img_folder_path = f"{self.data_folder}face_images/{str(int(index)).zfill(5)}/"
            img_files = sorted(os.listdir(img_folder_path), key=lambda x: x.zfill(15))
            images = []
            """
            meta = next(imageio_ffmpeg.read_frames(f"{self.data_folder}raw/{str(int(index)).zfill(5)}.mp4"))
            fps_est = len(img_files)/meta['duration']
            if 'Thumbs.db' in img_files:
                img_files.remove('Thumbs.db')
            selected_indices = np.linspace(0, len(img_files) - 1, min(12*5, max(1, round(5/fps_est*len(img_files)))), dtype=int)
            images = []
            for idx in selected_indices:#range(len(img_files[:12*5])):
                img_path = os.path.join(img_folder_path, img_files[idx])
                img = np.array(Image.open(img_path))#.convert('RGB')#.resize((160, 160))
                #images.append(self.transform(image=np.array(img))['image'])
                images.append(torch.tensor(img))
            #self.update(1-len(images)/50)
            # Add black images if there are less than 50 images
            """
            while len(images) < 1:
                black_img = Image.new('RGB', (224, 224))
                images.append(self.transform(image=np.array(black_img))['image'])
                
            return torch.stack(images)
        except Exception as e:
            images = []
            print(e)
            while len(images) < 1: # correct when face images are there
                black_img = Image.new('RGB', (160, 160))
                images.append(self.transform(image=np.array(black_img))['image'])
            print(f"No image found for index: {index}")
            return torch.stack(images)

    def process_audio(self, filename):
        audio_file_path = f"{self.data_folder}audio/{str(int(filename)).zfill(5)}.mp3"
        try:
            audio_data, sr = sf.read(audio_file_path)
            audio_data = audio_data.astype(np.float32)
            if sr != 16000:
                print(audio_file_path)
                raise ValueError
        except Exception as e:
            print(f"Error processing audio file {audio_file_path}: {e}")
            audio_data = np.zeros(120*16000+1, dtype=np.float32)
        self.update(1 - len(audio_data[:12*sr])/len(audio_data))

        
        # Ensure correct shape (Mono -> (1, samples))
        if audio_data.ndim == 1:
            audio_data = np.expand_dims(audio_data, axis=0)
        elif audio_data.shape[1] == 1:
            audio_data = audio_data.T

        # Apply wave transformations if available

        # Remove channel dimension (1D output)
        audio_data =  audio_data.squeeze(axis=0)
        return audio_data[:12*sr]

    def process_text(self, filename):
        text_file_path = f"{self.data_folder}text/{str(int(filename)).zfill(5)}.txt"
        with open(text_file_path, 'r', encoding='utf-8') as file:
            text = file.read().strip()
        return text


    def __len__(self):
        return len(self.label_file)

    def collate_fn(self, batch):
        audio_data, vision_data, max_length, text_data, labels_data, avg = zip(*batch)
        audio_data_padded = self.processor(audio_data, padding=True, sampling_rate=16000, return_tensors="pt", truncation=True, max_length=12*16000, return_attention_mask=True)
        lengths, permutation = audio_data_padded['attention_mask'].sum(axis=1).sort(descending=True)
        #audio_packed = pack_padded_sequence(audio_data_padded['input_values'][permutation], lengths.cpu().numpy(), batch_first=True)  # 'input_features' for w2v2-bert
        # assumption: audio lengths match vision lengths; it does not hold.
        #vision_data = [self.processor_vision(x, return_tensors='pt')['pixel_values'] for x in vision_data]
        #vision_packed = pack_sequence([vision_data[x] for x in permutation], enforce_sorted=False)
    
        labels_stacked = torch.stack(labels_data)
        encoded_text = self.processor_text(
            text_data,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )   
        return audio_data_padded, torch.stack(vision_data), torch.stack(max_length), encoded_text, labels_stacked, np.mean(avg)


class HumeDatasetEval(Dataset):

    def __init__(self, data_folder, label_file=None, model=None):
        super().__init__()
        self.data_folder = data_folder
        self.label_file = pd.read_csv(label_file)
        self.vision_model = model[0]
        self.audio_model = model[1]
        self.text_model = model[2]

        if self.vision_model != 'linear':
            self.data_config = timm.data.resolve_model_data_config(self.vision_model)
            self.transform = A.Compose([
                A.Resize(height=self.data_config['input_size'][1], width=self.data_config['input_size'][2]),
                A.Normalize(mean=self.data_config['mean'], std=self.data_config['std']),
                ToTensorV2(),
            ])
        if self.audio_model != 'linear':
            if "audeering" in self.audio_model:
                self.processor = AutoProcessor.from_pretrained(self.audio_model)
            else:
                self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.audio_model) 

            self.processor_vision = AutoProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.processor_text = AutoTokenizer.from_pretrained(self.text_model)

    def __getitem__(self, index):
        #row = self.label_file.iloc[int(2*len(self.label_file)/3)+index]
        row = self.label_file.iloc[index]

        if self.vision_model == 'linear':
            #vision = torch.randn(1024)
            try:
                vit_file_path = f"{self.data_folder}googlevit/{str(int(row['Filename'])).zfill(5)}.pkl"
                with open(vit_file_path, 'rb') as file:
                    data = pickle.load(file)
                    if isinstance(data, torch.Tensor):
                        tensor = data
                    else:
                        tensor = torch.tensor(data)
                    length = tensor.size(0)
                    max_length = 400
                    if length < max_length:
                        pad_size = (0, 0, 0, max_length - length)
                        vision = torch.nn.functional.pad(tensor, pad_size)
                    else:
                        vision = tensor[:max_length]
                        length = 400

            except Exception as e:
                vision = torch.randn(768)
        else:
            vision = self.process_images(index)

        if self.audio_model == 'linear':
            wav2vec2_file_path = f"{self.data_folder}wav2vec2/{str(int(row['Filename'])).zfill(5)}.pkl"
            with open(wav2vec2_file_path, 'rb') as file:
                audio = torch.mean(torch.tensor(pickle.load(file)), dim=0)
        else:
            audio = self.process_audio(row['Filename'])
        labels = torch.tensor(
            row[['Admiration', 'Amusement', 'Determination', 'Empathic Pain', 'Excitement', 'Joy']].values,
            dtype=torch.float)
        text = self.process_text(row['Filename'])
        return audio, vision, torch.tensor(length).long(), text, labels, int(row['Filename'])

    def process_images(self, index):
        try:
            img_folder_path = f"{self.data_folder}face_images/{str(int(index)).zfill(5)}/"
            img_files = sorted(os.listdir(img_folder_path), key=lambda x: x.zfill(15))
            images = []
            """
            meta = next(imageio_ffmpeg.read_frames(f"{self.data_folder}raw/{str(int(index)).zfill(5)}.mp4"))
            fps_est = len(img_files)/meta['duration']
            if 'Thumbs.db' in img_files:
                img_files.remove('Thumbs.db')
            selected_indices = np.linspace(0, len(img_files) - 1, min(12*5, max(1, round(5/fps_est*len(img_files)))), dtype=int)
            images = []
            for idx in selected_indices:  # range(len(img_files[:12*5])):
                img_path = os.path.join(img_folder_path, img_files[idx])
                img = np.array(Image.open(img_path))#.convert('RGB')#.resize((160, 160))
                #images.append(self.transform(image=np.array(img))['image'])
                images.append(torch.tensor(img))
            """
            # Add black images if there are less than 50 images
            while len(images) < 1:
                black_img = Image.new('RGB', (160, 160))
                images.append(self.transform(image=np.array(black_img))['image'])

            return torch.stack(images)
        except:
            images = []
            while len(images) < 1: # TODO correct when faceimage are there
                black_img = Image.new('RGB', (160, 160))
                images.append(self.transform(image=np.array(black_img))['image'])
            print(f"No image found for index: {index}")
            return torch.stack(images)

    def process_audio(self, filename):
        audio_file_path = f"{self.data_folder}audio/{str(int(filename)).zfill(5)}.mp3"
        try:
            audio_data, sr = sf.read(audio_file_path)
            if sr != 16000:
                print(audio_file_path)
                raise ValueError
        except Exception as e:
            print(f"Error processing audio file {audio_file_path}: {e}")
            audio_data = np.zeros((128,), dtype=np.float32)
            sr = 1

        return audio_data[:12*sr]
    def process_text(self, filename):
        text_file_path = f"{self.data_folder}text/{str(int(filename)).zfill(5)}.txt"
        with open(text_file_path, 'r', encoding='utf-8') as file:
            text = file.read().strip()
        return text

    def __len__(self):
        return len(self.label_file)
        #return int(len(self.label_file)/3)

    def collate_fn(self, batch):
        audio_data, vision_data, max_length, text_data, labels_data, avg = zip(*batch)
        audio_data_padded = self.processor(audio_data, padding=True, sampling_rate=16000, return_tensors="pt", truncation=True, max_length=12*16000, return_attention_mask=True)
        lengths, permutation = audio_data_padded['attention_mask'].sum(axis=1).sort(descending=True)
        #audio_packed = pack_padded_sequence(audio_data_padded['input_values'][permutation], lengths.cpu().numpy(), batch_first=True)  # 'input_features' for w2v2-bert
        # assumption: audio lengths match vision lengths; it does not hold.
        #vision_data = [self.processor_vision(x, return_tensors='pt')['pixel_values'] for x in vision_data]
        #vision_packed = pack_sequence([vision_data[x] for x in permutation], enforce_sorted=False)
    
        labels_stacked = torch.stack(labels_data)
        encoded_text = self.processor_text(
            text_data,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )   
        return audio_data_padded, torch.stack(vision_data), torch.stack(max_length), encoded_text, labels_stacked, avg

