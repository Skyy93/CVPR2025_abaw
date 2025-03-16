import torch
import timm
import numpy as np
import torch.nn as nn
from transformers import Wav2Vec2BertModel, Wav2Vec2Model, ViTForImageClassification, AutoModel
from torch.nn.utils.rnn import unpack_sequence, pack_sequence
from abaw.audeer import EmotionModel

class Model(nn.Module):

    def __init__(self,
                 model_name,
                 task=None
                 ):

        super(Model, self).__init__()
        self.linear = False
        self.model_name = model_name
        if "linear" in model_name[1]:
            self.model = nn.Linear(1152, 6)
            self.linear = True
        else:
            if "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim" in model_name[1]:
                self.audio_model = EmotionModel.from_pretrained(model_name[1])
                self.text_model = AutoModel.from_pretrained(model_name[2],trust_remote_code=True)
                self.task = task
                if task == "text":
                    feat = 768
                elif task == "vit":
                    feat = 768
                elif task == "audio":
                    feat = 1027
                elif task == "text+audio":
                    feat = 1795
                elif task == "text+vit":
                    feat = 1536
                elif task == "text+vit+audio":
                    feat = 2563


                self.fusion_model = nn.Sequential(nn.Linear(feat, 1027),
                                              nn.Tanh(),
                                              nn.Linear(1027, 6),
                                              )
                self.lstm_audio = nn.LSTM(1027, 1027, num_layers=2, batch_first=True, bidirectional=False)
                self.lstm_vis = nn.LSTM(1280, 768, num_layers=2, batch_first=True, bidirectional=False)
                               
    def forward(self, audio, vision, text, length):
        if self.linear:
            # For the linear case, simply use the mean of vision and audio features.
            return self.model(torch.cat([torch.mean(vision, dim=0), torch.mean(audio, dim=0)], dim=1))
        else:
            raw_lengths = audio['attention_mask'].sum(dim=1)  # Tensor of shape [batch_size]
            max_padded_length = 12 * 16000
            if "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim" in self.model_name[1]:
                audio_output = self.audio_model(audio['input_values'])
                audio_cat = torch.cat((audio_output[0], audio_output[1]), dim=2)
                transformer_output_length = audio_cat.size(1)
                downsampling_factor = max_padded_length / transformer_output_length

                effective_lengths = torch.floor(raw_lengths.float() / downsampling_factor).long()
                effective_lengths = torch.clamp(effective_lengths, min=1)
                features = []
                if "audio" in self.task:
                    lstm_audio, _ = self.lstm_audio(audio_cat)
                    batch_indices = torch.arange(lstm_audio.size(0), device=lstm_audio.device)
                    audio_input = lstm_audio[batch_indices, effective_lengths - 1, :]
                    features.append(audio_input)
                if "vit" in self.task:
                    lstm_output,_ = self.lstm_vis(vision)  # assuming shape: [batch_size, seq_len, features]
                    batch_indices = torch.arange(lstm_output.size(0))
                    vision = lstm_output[batch_indices, length.cpu() - 1, :]
                    features.append(vision)
                if "text" in self.task:
                    text_feat = self.text_model(**text).last_hidden_state[:, 0, :]
                    features.append(text_feat)

                if len(features) > 1:
                    fusion_input = torch.cat(features, dim=1)
                else:
                    fusion_input = features[0]
                pred = self.fusion_model(fusion_input)
                return pred

