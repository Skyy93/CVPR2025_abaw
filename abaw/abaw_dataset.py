import os
import cv2
import timm
import yaml
import torch
import pickle
import random

import numpy as np
import pandas as pd
import soundfile as sf
import albumentations as A

from dataclasses import dataclass
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from transformers import AutoProcessor, Wav2Vec2FeatureExtractor

import abaw.utils
from abaw.text_transforms import TextAugmentor

cv2.setNumThreads(2)

DEBUG = False

## Special Tokens
@dataclass
class SpecialTokens:
    #start: str = "[START]"
    #end: str = "[END]"
    time_attention: str = "[ATTENTION]"

    def generate_special_token_dict():
        return {'additional_special_tokens': [SpecialTokens.time_attention]} #, SpecialTokens.start, SpecialTokens.end, ]}
    

def printD(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

class HumeDatasetTrain(Dataset, abaw.utils.AverageMeter):

    def __init__(self, data_folder, label_file_csv=None, model=None, processor_text=None, sr=16*1e3, fps=24.0, vit_frame_width=12*24, max_googlevit_feat_length=400, wave_length_s=12.0, wave_transforms=None, enable_text_augs=False, text_context_width_s=15.0, config=None):
        super().__init__()
        self.data_folder = data_folder
        self.vision_model = model[0]
        self.audio_model = model[1]
        self.text_model = model[2]
        self.sr = sr
        self.fps = fps
        self.vit_frame_width = vit_frame_width
        self.max_googlevit_feat_length = max_googlevit_feat_length
        self.wave_length_s = wave_length_s
        self.wave_transforms = wave_transforms
        self.text_context_width_s = text_context_width_s

        self.meta = pd.read_csv(label_file_csv)
        self.samples = self.meta.index.tolist()

        if self.audio_model:
            if "audeering" in self.audio_model:
                self.processor_audio = AutoProcessor.from_pretrained(self.audio_model)
            else:
                self.processor_audio = Wav2Vec2FeatureExtractor.from_pretrained(self.audio_model)
        
        if self.text_model:
            self.processor_text = processor_text

        # Running the TextAugmentor on CPU!
        self.text_augmentor = TextAugmentor(enable=enable_text_augs, device="cpu", prob_synonym=config.prob_synonym, prob_deletion=config.prob_deletion, prob_swap=config.prob_swap, prob_bert=config.prob_bert)

    def sample_balanced(self):
        print("🚀 Launching balanced sample resampling...")
        # Total number of samples in self // 2
        #half_samples = len(self.meta) // 2

        # Select positives and negatives from meta.
        pos = self.meta[self.meta['class_id'] == 1]
        neg = self.meta[self.meta['class_id'] == 0]
        
        if len(pos) >= len(neg):
            minority = neg
            majority = pos
        else:
            minority = pos
            majority = neg
        
        # Minority Sampling
        minority_ids = minority.index.tolist()
        #while len(minority_ids) < half_samples:
        #    minority_ids.append(random.choice(minority.index.tolist()))
        
        # If oversampled -> only take as many as needed!
        #minority_ids = minority_ids[:half_samples]
        majority_ids = random.sample(majority.index.tolist(), len(minority_ids))#half_samples)

        # Combine the two lists and shuffle
        new_train_ids = minority_ids + majority_ids
        random.shuffle(new_train_ids)
        self.samples = new_train_ids

    def __getitem__(self, index):
        idx = self.samples[index]
        row = self.meta.iloc[idx]
        general_path = row['general_path']
        class_id = int(row['class_id']) 

        sound_start_index = row['sound_start_index']
        sound_end_index = row['sound_end_index']

        frame_start_index = row['frame_start_index']
        frame_end_index = row['frame_end_index']

        random_midpoint = np.random.uniform(0.12, 0.88)
        #random_midpoint = np.random.uniform(0.495, 0.505)
        #random_midpoint = np.random.uniform(0.00, 1.00)

        sound_index = int(sound_start_index + (sound_end_index - sound_start_index) * random_midpoint)
        frame_index = int(frame_start_index + (frame_end_index - frame_start_index) * random_midpoint) # maybe useful later!
        time_point = sound_index / self.sr

        if self.vision_model == 'googlevit':
            vision, length = self.process_images_pkl(general_path, frame_index)
        else:
            vision = torch.randn(1024)
            length = torch.tensor(1)
        
        audio = self.process_audio(general_path, sound_index)
        text = self.process_text(general_path, time_point)
        label_id = torch.tensor(class_id, dtype=torch.float)

        #df_idx = idx

        return audio, vision, length, text, label_id, self.avg #, torch.tensor(sound_index, dtype=torch.float), torch.tensor(time_point, dtype=torch.float),  torch.tensor(df_idx, dtype=torch.float)

    def process_images_pkl(self, subpath_no_ext, frame_index):
        vit_file_path = os.path.join(self.data_folder, "googlevit", f"{subpath_no_ext}.pkl")

        with open(vit_file_path, 'rb') as file:
            data = pickle.load(file)
            if isinstance(data, torch.Tensor):
                tensor = data
            else:
                tensor = torch.tensor(torch.tensor(data))

            length = tensor.size(0)
            if length < self.max_googlevit_feat_length:
                pad_size = (0, 0, 0, self.max_googlevit_feat_length - length)
                vision = torch.nn.functional.pad(tensor, pad_size)
            else:
                start_frame = max(0, frame_index - self.vit_frame_width // 2)
                end_frame = min(length - 1, frame_index + self.vit_frame_width // 2)
                indices = torch.linspace(start_frame, end_frame, steps=self.max_googlevit_feat_length).int()
                vision = tensor[indices]
                length = self.max_googlevit_feat_length

        return vision, torch.tensor(length)
            
    def process_audio(self, subpath_no_ext, sound_index):   
        if not self.audio_model:
            return np.zeros(int(self.wave_length_s*self.sr), dtype=np.float32)

        audio_file_path = os.path.join(self.data_folder, f"sound_{int(self.sr/1000)}kHz", f"{subpath_no_ext}.wav")
        audio_data, sr = sf.read(audio_file_path, dtype="float32")

        # Ensure correct sample rate
        if abs(sr - self.sr) > 1e-6:
            print(audio_file_path)
            raise ValueError
        
        # Calculate the start and end of the random choosen audio sample
        segment_duration = int(self.wave_length_s * sr) # not in seconds!!

        start_index = max(sound_index - segment_duration//2, 0)
        stop_index = min(sound_index + segment_duration//2, len(audio_data))
        
        # Extract the segment
        audio_data = audio_data[start_index:stop_index]

        # Ensure a minimum length for FFT-based augmentations (e.g., n_fft=2048)
        # n_fft = 2048
        # Some transformations, which internally use librosa, require at least 2048 samples.
        # If the audio is shorter, pad it with zeros to avoid warnings.
        # This modification affects only very short audio files, which should be rare in the dataset.
        # if audio_data.shape[-1] < n_fft:
        #     pad_size = n_fft - audio_data.shape[-1]
        #     # Randomly distribute silence on both sides
        #     left_pad = random.randint(0, pad_size)
        #     right_pad = pad_size - left_pad
        #     audio_data = np.pad(audio_data, (left_pad, right_pad), mode="constant")
        # ⚠️ This padding is **not** for ensuring the final sequence length!
        # → It only prevents warnings/errors caused by augmentations using librosa under the hood.
        # → Full padding happens later in `collate_fn`, where an attention mask is also created.

        # Update with trimmed length
        self.update(1 - len(audio_data) / (self.wave_length_s * sr)) 

        # Ensure correct shape (Mono -> (1, samples))
        if audio_data.ndim == 1:
            audio_data = np.expand_dims(audio_data, axis=0)
        elif audio_data.shape[1] == 1:
            audio_data = audio_data.T

        # Apply wave transformations if available
        if self.wave_transforms is not None:
            audio_data = self.wave_transforms(samples=audio_data, sample_rate=sr)

        # Remove channel dimension (1D output)
        processed_audio_data = audio_data.squeeze(axis=0)

        return processed_audio_data

    def process_text(self, subpath_no_ext, time_point):
        if not self.text_model:
            return " "
        
        printD(f"time_point: {time_point}")  # Debug print
        yaml_path = os.path.join(self.data_folder, "transcription", f"{subpath_no_ext}.yml")
        printD(f"Loading YAML file: {yaml_path}")  # Debug print

        with open(yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        printD(f"Raw YAML Data: {data}")  # Debug print  

        text_df = pd.DataFrame(data['chunks'])
        printD(f"Loaded {len(text_df)} text chunks")  # Debug print  

        printD(f"DataFrame Columns: {text_df.columns}")  # Debug print 
        printD(f"DataFrame Sample:\n{text_df.head()}")  # Debug print  
        for index, row in text_df.iterrows():
            printD(f"Row {index}: {row.to_dict()}")  # Debug print 

        # Fix for incorrect timestamp resets
        offset = 0.0  # Holds the last ending time of a timestamp
        for i in range(len(text_df)):
            start, end = text_df.at[i, 'timestamp']
            text_df.at[i, 'timestamp'] = (start + offset, end + offset)  # Fixing the timestamp
            if i+1 < len(text_df):
                next_start, _ = text_df.at[i+1, 'timestamp']
                if next_start == 0.0:
                    offset = end

        printD("DataFrame Sample (fixed):")  # Debug print 
        for index, row in text_df.iterrows():
            printD(f"Row {index}: {row.to_dict()}")  # Debug print 

        last_timepoint = max(x[1] for x in text_df['timestamp'])
        context_start_s = max(time_point - self.text_context_width_s/2, 0)
        context_end_s = min(time_point + self.text_context_width_s/2, last_timepoint) 

        affected_text_chunks = text_df[text_df['timestamp'].apply(lambda x: (
            (x[1] > context_start_s and x[0] <= context_start_s)  # Overlaps at the beginning
            or (x[0] < context_end_s and x[1] >= context_end_s)   # Overlaps at the end
            or (x[0] <= context_start_s and x[1] >= context_end_s)  # Fully covers the context
            or (x[0] >= context_start_s and x[1] <= context_end_s)  # Fully inside the context
            ))].reset_index(drop=True)
        
        printD(f"Affected text chunks count: {len(affected_text_chunks)}")  # Debug print

        timestamps = affected_text_chunks['timestamp'].tolist()
        printD(f"Extracted timestamps: {timestamps}")  # Debug print

        time_attention_set = False
        text_list = []
        for idx, row in affected_text_chunks.iterrows():
            timestamp = row['timestamp']
            text_chunk = row['text']

            words = text_chunk.split()
            word_count = len(words)

            chunk_duration = timestamp[1] - timestamp[0]
            words = text_chunk.split()
            word_count = len(words)

            chunk_duration = timestamp[1] - timestamp[0]

            if timestamp[0] >= context_start_s:
                begin_word_cut = 0  # remove nothing
            else:
                cut_time = context_start_s - timestamp[0]
                cut_rel = cut_time/chunk_duration
                begin_word_cut = int(word_count * cut_rel)

            if timestamp[1] <= context_end_s:
                end_word_cut = word_count  # remove nothing
            else:
                cut_time = timestamp[1] - context_end_s
                cut_rel = cut_time/chunk_duration
                end_word_cut = int(word_count * cut_rel)

            printD(f"timepoint: {time_point}")  # Debug print 

            printD(f"context_start_s: {context_start_s}")  # Debug print 
            printD(f"context_end_s: {context_end_s}")  # Debug print 

            printD(f"begin_word_cut: {begin_word_cut}")  # Debug print 
            printD(f"end_word_cut: {end_word_cut}")  # Debug print

            cutted_words = words[begin_word_cut:end_word_cut]

            if not time_attention_set:
                # Insert the attention token within the current chunk when the time_point falls inside it.
                if timestamp[0] <= time_point <= timestamp[1] :
                    relative_position = (time_point - timestamp[0]) / chunk_duration  # Normalize
                    insert_index = max(0, min(word_count - 1, round(relative_position * len(cutted_words))))
                    cutted_words.insert(insert_index, SpecialTokens.time_attention)
                    time_attention_set = True

                # If the time_point is before the start of the first chunk, insert the token at the very beginning.
                elif idx == affected_text_chunks.index[0] and time_point <= timestamp[0]:
                        cutted_words.insert(0, SpecialTokens.time_attention)
                        time_attention_set = True

                # If the time_point falls between the end of the previous chunk and the start of the current chunk,
                # insert the attention token at the beginning of the current chunk.
                elif idx > affected_text_chunks.index[0] and  affected_text_chunks.iloc[idx-1]['timestamp'][1] <= time_point <= timestamp[0]:
                        cutted_words.insert(0, SpecialTokens.time_attention)
                        time_attention_set = True

                # If the time_point is after the end of the last chunk, append the token at the end of the current chunk.
                elif idx == len(affected_text_chunks) - 1 and time_point >= timestamp[1]:
                        cutted_words.insert(len(cutted_words), SpecialTokens.time_attention)
                        time_attention_set = True

            text_list.append(" ".join(cutted_words))

        text = " ".join(text_list)

        if not text:
            return " " + SpecialTokens.time_attention + " "

        if time_attention_set == False:
            [print(f"time_point {time_point} - within {timestamp[0]} and {timestamp[1]}??") for timestamp in affected_text_chunks['timestamp'].tolist()]
            # print("NO MASK ADDED")


        # if abs(context_start_s) <= 1e-3:
        #     text = SpecialTokens.start + " " + text  # Add space for readability
        # if abs(context_end_s - last_timepoint) <= 1e-3:
        #     text = text + " " + SpecialTokens.end    # Add space for readability

        printD(f"Text: {text}")  # Debug print
        printD(f"Type of Text: {type(text)}")

        if len(text_list) < 8:
            return text

        text_aug = self.text_augmentor.augment(text)
    
        printD(f"Augmented Text: {text_aug}") 

        return text_aug

    def __len__(self):
        return len(self.samples)

    def collate_fn(self, batch):
        audio_data, vision_data, max_length, text_data, label_id_data, avg = zip(*batch) #, s, t, idx = zip(*batch)

        if self.audio_model:
            audio_data_padded = self.processor_audio(audio_data, padding=True, sampling_rate=self.sr, return_tensors="pt", truncation=True, max_length=int(self.wave_length_s*self.sr), return_attention_mask=True)
        else:
            audio_data_padded = {"input_ids": torch.zeros(len(label_id_data), device=label_id_data[0].device)} 

        if self.text_model:
            encoded_text = self.processor_text(
                text_data,
                add_special_tokens=True,
                #max_length=128, #TODO: größer??
                padding=True,  #'max_length', # True
                truncation=True,
                return_tensors='pt'
            )  
            text_time_attention_token_id = self.processor_text.convert_tokens_to_ids(SpecialTokens.time_attention)
            text_time_attention_position = (encoded_text['input_ids'] == text_time_attention_token_id).nonzero()[:, 1]
        else: 
            encoded_text = {"input_ids": torch.zeros(len(label_id_data), device=label_id_data[0].device)} 
            text_time_attention_position = torch.zeros(len(label_id_data), device=label_id_data[0].device)

        label_id_stacked = torch.stack(label_id_data)

        #batch_size = encoded_text['input_ids'].size(0)
        #attention_positions = []

        #for i in range(batch_size):
        #    positions = (encoded_text['input_ids'][i] == text_time_attention_token_id).nonzero(as_tuple=True)[0]
        #    if len(positions) == 0:
        #        raise ValueError(f"Batch-Element {i} enthält keinen time_attention Token!")
        #    attention_position = positions[0]  # Wähle ersten gefundenen Token (falls mehrere vorkommen)
        #    attention_positions.append(attention_position)

        #text_time_attention_position = torch.stack(attention_positions)

        #print(encoded_text['input_ids'])
        #print(text_time_attention_position)

        return audio_data_padded, torch.stack(vision_data), torch.stack(max_length), encoded_text, text_time_attention_position, label_id_stacked, np.mean(avg) # ,torch.stack(s),torch.stack(t),torch.stack(idx)


class HumeDatasetEval(Dataset):

    def __init__(self, data_folder, label_file_csv=None, model=None, processor_text=None, sr=16*1e3, fps=24.0, eval_fps=0.3, vit_frame_width=12*24, max_googlevit_feat_length=400, wave_length_s=6.0, text_context_width_s=15.0):
        super().__init__()
        self.data_folder = data_folder
        self.vision_model = model[0]
        self.audio_model = model[1]
        self.text_model = model[2]
        self.sr = sr
        self.fps = fps
        self.vit_frame_width = vit_frame_width
        self.max_googlevit_feat_length = max_googlevit_feat_length
        self.eval_fps = eval_fps
        self.wave_length_s = wave_length_s
        self.text_context_width_s = text_context_width_s
        self.meta = self.generate_eval_meta(label_file_csv)

        if self.audio_model:
            if "audeering" in self.audio_model:
                self.processor_audio = AutoProcessor.from_pretrained(self.audio_model)
            else:
                self.processor_audio = Wav2Vec2FeatureExtractor.from_pretrained(self.audio_model)
        
        if self.text_model:
            self.processor_text = processor_text

    def generate_eval_meta(self, label_file_csv):
        raw_meta_df = pd.read_csv(label_file_csv)

        eval_rows = []
        step = int(self.fps / self.eval_fps)
        
        for _, row in raw_meta_df.iterrows():
            frame_start = row['frame_start_index']
            frame_end = row['frame_end_index']
            class_id = row['class_id']
            general_path = row['general_path']
            
            for frame_index in np.arange(frame_start, frame_end + 1, step):
                sound_index = int(np.floor((frame_index / self.fps) * self.sr))
                eval_rows.append({
                    'frame_index': int(frame_index),
                    'sound_index': sound_index,
                    'class_id': class_id,
                    'general_path': general_path
                })

        eval_df = pd.DataFrame(eval_rows)
        eval_df.to_csv("debug.csv", index=False)
        
        return eval_df

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        general_path = row['general_path']
        class_id = int(row['class_id']) 

        sound_index = row['sound_index']
        frame_index = row['frame_index'] # maybe useful later!
        time_point = sound_index / self.sr 

        if self.vision_model == 'googlevit':
            vision, length = self.process_images_pkl(general_path, frame_index)
        else:
            vision = torch.randn(1024)
            length = torch.tensor(1)
        
        audio = self.process_audio(general_path, sound_index)
        label_id = torch.tensor(class_id, dtype=torch.float)
        text = self.process_text(general_path, time_point)

        # print("text: ", text)
        # print("sound_index: ", sound_index)
        # print("frame_index: ", frame_index)
        # print("time_point: ", time_point)
        # print("general_path: ", general_path)
        # print("class_id: ", class_id)
        #return audio, vision, text, label_id

        # df_idx = index

        return audio, vision, length, text, label_id # , torch.tensor(sound_index, dtype=torch.float), torch.tensor(time_point, dtype=torch.float),  torch.tensor(df_idx, dtype=torch.float)

    def process_images_pkl(self, subpath_no_ext, frame_index):
        vit_file_path = os.path.join(self.data_folder, "googlevit", f"{subpath_no_ext}.pkl")

        with open(vit_file_path, 'rb') as file:
            data = pickle.load(file)
            if isinstance(data, torch.Tensor):
                tensor = data
            else:
                tensor = torch.tensor(data)

            length = tensor.size(0)
            if length < self.max_googlevit_feat_length:
                pad_size = (0, 0, 0, self.max_googlevit_feat_length - length)
                vision = torch.nn.functional.pad(tensor, pad_size)
            else:
                start_frame = max(0, frame_index - self.vit_frame_width // 2)
                end_frame = min(length - 1, frame_index + self.vit_frame_width // 2)
                indices = torch.linspace(start_frame, end_frame, steps=self.max_googlevit_feat_length).int()
                vision = tensor[indices]
                length = self.max_googlevit_feat_length

        return vision, torch.tensor(length)

    def process_audio(self, subpath_no_ext, sound_index):
        if not self.audio_model:
            return np.zeros(int(self.wave_length_s*self.sr), dtype=np.float32)

        audio_file_path = os.path.join(self.data_folder, f"sound_{int(self.sr/1000)}kHz", f"{subpath_no_ext}.wav")
        audio_data, sr = sf.read(audio_file_path, dtype="float32")
        # Ensure correct sample rate
        if abs(sr - self.sr) > 1e-6:
            print(audio_file_path)
            raise ValueError
        # Extract a {self.wave_length_s} second segment centered on the specified sound_index
        segment_duration = int(self.wave_length_s * sr) # not in seconds!!
        
        # Calculate the start and end sample indices
        start_index = max(sound_index - segment_duration//2, 0)
        stop_index = min(sound_index + segment_duration//2, len(audio_data))
        
        # Extract the segment
        audio_data = audio_data[start_index:stop_index]

        # Ensure a minimum length for FFT-based augmentations (e.g., n_fft=2048)
        # n_fft = 2048
        # Some transformations, which internally use librosa, require at least 2048 samples.
        # If the audio is shorter, pad it with zeros to avoid warnings.
        # This modification affects only very short audio files, which should be rare in the dataset.
        # if audio_data.shape[-1] < n_fft:
        #     pad_size = n_fft - audio_data.shape[-1]
        #     # Randomly distribute silence on both sides
        #     left_pad = random.randint(0, pad_size)
        #     right_pad = pad_size - left_pad
        #     audio_data = np.pad(audio_data, (left_pad, right_pad), mode="constant")
        # ⚠️ This padding is **not** for ensuring the final sequence length!
        # → It only prevents warnings/errors caused by augmentations using librosa under the hood.
        # → Full padding happens later in `collate_fn`, where an attention mask is also created.

        return audio_data

    def process_text(self, subpath_no_ext, time_point):
        if not self.text_model:
            return " "

        printD(f"time_point: {time_point}")  # Debug print
        yaml_path = os.path.join(self.data_folder, "transcription", f"{subpath_no_ext}.yml")
        printD(f"Loading YAML file: {yaml_path}")  # Debug print
        
        with open(yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        
        printD(f"Raw YAML Data: {data}")  # Debug print  

        text_df = pd.DataFrame(data['chunks'])
        printD(f"Loaded {len(text_df)} text chunks")  # Debug print  

        printD(f"DataFrame Columns: {text_df.columns}")  # Debug print 
        printD("DataFrame Sample:")  # Debug print 
        for index, row in text_df.iterrows():
            printD(f"Row {index}: {row.to_dict()}")  # Debug print 

        # Fix for incorrect timestamp resets
        offset = 0.0  # Holds the last ending time of a timestamp
        for i in range(len(text_df)):
            start, end = text_df.at[i, 'timestamp']
            text_df.at[i, 'timestamp'] = (start + offset, end + offset)  # Fixing the timestamp
            if i+1 < len(text_df):
                next_start, _ = text_df.at[i+1, 'timestamp']
                if next_start == 0.0:
                    offset = end

        printD("DataFrame Sample (fixed):")  # Debug print 
        for index, row in text_df.iterrows():
            printD(f"Row {index}: {row.to_dict()}")  # Debug print 

        last_timepoint = max(x[1] for x in text_df['timestamp'])
        context_start_s = max(time_point - self.text_context_width_s/2, 0)
        context_end_s = min(time_point + self.text_context_width_s/2, last_timepoint) 

        affected_text_chunks = text_df[text_df['timestamp'].apply(lambda x: (
            (x[1] > context_start_s and x[0] <= context_start_s)  # Overlaps at the beginning
            or (x[0] < context_end_s and x[1] >= context_end_s)   # Overlaps at the end
            or (x[0] <= context_start_s and x[1] >= context_end_s)  # Fully covers the context
            or (x[0] >= context_start_s and x[1] <= context_end_s)  # Fully inside the context
            ))].reset_index(drop=True)
        
        printD(f"Affected text chunks count: {len(affected_text_chunks)}")  # Debug print

        timestamps = affected_text_chunks['timestamp'].tolist()
        printD(f"Extracted timestamps: {timestamps}")  # Debug print

        time_attention_set = False
        text_list = []
        for idx, row in affected_text_chunks.iterrows():
            timestamp = row['timestamp']
            text_chunk = row['text']

            words = text_chunk.split()
            word_count = len(words)

            chunk_duration = timestamp[1] - timestamp[0]

            if timestamp[0] >= context_start_s:
                begin_word_cut = 0  # remove nothing
            else:
                cut_time = context_start_s - timestamp[0]
                cut_rel = cut_time/chunk_duration
                begin_word_cut = int(word_count * cut_rel)

            if timestamp[1] <= context_end_s:
                end_word_cut = word_count  # remove nothing
            else:
                cut_time = timestamp[1] - context_end_s
                cut_rel = cut_time/chunk_duration
                end_word_cut = int(word_count * cut_rel)

            printD(f"timepoint: {time_point}")  # Debug print 

            printD(f"context_start_s: {context_start_s}")  # Debug print 
            printD(f"context_end_s: {context_end_s}")  # Debug print 

            printD(f"begin_word_cut: {begin_word_cut}")  # Debug print 
            printD(f"end_word_cut: {end_word_cut}")  # Debug print

            cutted_words = words[begin_word_cut:end_word_cut]
            
            if not time_attention_set:
                # Insert the attention token within the current chunk when the time_point falls inside it.
                if timestamp[0] <= time_point <= timestamp[1] :
                    relative_position = (time_point - timestamp[0]) / chunk_duration  # Normalize
                    insert_index = max(0, min(word_count - 1, round(relative_position * len(cutted_words))))
                    cutted_words.insert(insert_index, SpecialTokens.time_attention)
                    time_attention_set = True

                # If the time_point is before the start of the first chunk, insert the token at the very beginning.
                elif idx == affected_text_chunks.index[0] and time_point <= timestamp[0]:
                        cutted_words.insert(0, SpecialTokens.time_attention)
                        time_attention_set = True

                # If the time_point falls between the end of the previous chunk and the start of the current chunk,
                # insert the attention token at the beginning of the current chunk.
                elif idx > affected_text_chunks.index[0] and  affected_text_chunks.iloc[idx-1]['timestamp'][1] <= time_point <= timestamp[0]:
                        cutted_words.insert(0, SpecialTokens.time_attention)
                        time_attention_set = True

                # If the time_point is after the end of the last chunk, append the token at the end of the current chunk.
                elif idx == len(affected_text_chunks) - 1 and time_point >= timestamp[1]:
                        cutted_words.insert(len(cutted_words), SpecialTokens.time_attention)
                        time_attention_set = True

            text_list.append(" ".join(cutted_words))

        text = " ".join(text_list)

        if not text:
            return " " + SpecialTokens.time_attention + " "
        
        if time_attention_set == False:
            [print(f"time_point {time_point} - within {timestamp[0]} and {timestamp[1]}??") for timestamp in affected_text_chunks['timestamp'].tolist()]
            #print("NO MASK ADDED")


       # if abs(context_start_s) <= 1e-3:
       #     text = SpecialTokens.start + " " + text  # Add space for readability
       # if abs(context_end_s - last_timepoint) <= 1e-3:
       #     text = text + " " + SpecialTokens.end    # Add space for readability

        printD(f"Text: {text}")  # Debug print
        printD(f"Type of Text: {type(text)}")

        return text

    def __len__(self):
        return len(self.meta)

    def collate_fn(self, batch):
        audio_data, vision_data, max_length, text_data, label_id_data = zip(*batch) # , s, t, idx = zip(*batch)

        if self.audio_model:
            audio_data_padded = self.processor_audio(audio_data, padding=True, sampling_rate=self.sr, return_tensors="pt", truncation=True, max_length=int(self.wave_length_s*self.sr), return_attention_mask=True)
        else:
            audio_data_padded = {"input_ids": torch.zeros(len(label_id_data), device=label_id_data[0].device)} 

        if self.text_model:
            encoded_text = self.processor_text(
                text_data,
                add_special_tokens=True,
                #max_length=128, #TODO: größer??
                padding=True,  #'max_length', # True
                truncation=True,
                return_tensors='pt'
            )  
            text_time_attention_token_id = self.processor_text.convert_tokens_to_ids(SpecialTokens.time_attention)
            text_time_attention_position = (encoded_text['input_ids'] == text_time_attention_token_id).nonzero()[:, 1]
        else: 
            encoded_text = {"input_ids": torch.zeros(len(label_id_data), device=label_id_data[0].device)} 
            text_time_attention_position = torch.zeros(len(label_id_data), device=label_id_data[0].device)

        label_id_stacked = torch.stack(label_id_data)

        #print(encoded_text['input_ids'])
        #print(text_time_attention_position)

        return audio_data_padded, torch.stack(vision_data), torch.stack(max_length), encoded_text, text_time_attention_position, label_id_stacked # , torch.stack(s), torch.stack(t), torch.stack(idx)