import torch
import timm
import numpy as np
import torch.nn as nn
from transformers import Wav2Vec2BertModel, Wav2Vec2Model, ViTForImageClassification, AutoModel
from torch.nn.utils.rnn import unpack_sequence, pack_sequence
from abaw.audeer import EmotionModel

from transformers import AutoProcessor

class Model(nn.Module):

    def __init__(self,
                 model_name,
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
                self.text_model = AutoModel.from_pretrained(model_name[2])
                self.fusion_model = nn.Sequential(nn.Linear(1795, 1027),
                                              nn.Tanh(),
                                              nn.Linear(1027, 6),
                                              #nn.Sigmoid()
                                              )
                self.lstm_audio = nn.LSTM(1027, 1027, num_layers=2, batch_first=True, bidirectional=False)
                
            else:
                if 'large' in model_name[1]:
                    feat = 1024
                else:
                    feat = 768
                self.text_model = AutoModel.from_pretrained(model_name[2])
                self.audio_model = AutoModel.from_pretrained(model_name[1])
                self.fusion_model = nn.Sequential(nn.Linear(feat+768, feat), 
                                              nn.Tanh(),  
                                              nn.Linear(feat, 6),
                                              #nn.Sigmoid()
                                              )
                self.lstm_audio = nn.LSTM(1280, feat, num_layers=2, batch_first=True, bidirectional=False)

                self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

    def forward(self, audio, vision, text):
        if self.linear:
            # For the linear case, simply use the mean of vision and audio features.
            return self.model(torch.cat([torch.mean(vision, dim=0), torch.mean(audio, dim=0)], dim=1))
        else:
            # raw_lengths = audio['attention_mask'].sum(dim=1)  # Tensor of shape [batch_size]
            max_padded_length = 30 * 16000

            if "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim" in self.model_name[1]:
                audio_output = self.audio_model(audio['input_values'])
                audio_cat = torch.cat((audio_output[0], audio_output[1]), dim=2)
                transformer_output_length = audio_cat.size(1)
                downsampling_factor = max_padded_length / transformer_output_length

                #effective_lengths = torch.floor(raw_lengths.float() / downsampling_factor).long()
                #effective_lengths = torch.clamp(effective_lengths, min=1)

                lstm_audio, _ = self.lstm_audio(audio_cat)
                batch_indices = torch.arange(lstm_audio.size(0), device=lstm_audio.device)
                fusion_input = lstm_audio[batch_indices, effective_lengths - 1, :]

                text_feat = self.text_model(**text).last_hidden_state[:, 0, :]
                pred = self.fusion_model(torch.cat([fusion_input, text_feat], dim=1))
                return pred

            elif "whisper-large-v3" in self.model_name[1]:
                decoder_input_ids = torch.tensor([[self.processor.tokenizer.bos_token_id]], device=audio['input_features'].device)

                audio_out = self.audio_model(input_features=audio['input_features'],decoder_input_ids=decoder_input_ids).last_hidden_state
                transformer_output_length = audio_out.size(1)
                downsampling_factor = max_padded_length / transformer_output_length

                effective_lengths = torch.floor(raw_lengths.float() / downsampling_factor).long()
                effective_lengths = torch.clamp(effective_lengths, min=1)

                #print("audio_out.shape:", audio_out.shape)
                #print("self.lstm_audio.input_size:", self.lstm_audio.input_size)

                lstm_audio, _ = self.lstm_audio(audio_out)
                batch_indices = torch.arange(lstm_audio.size(0), device=lstm_audio.device)
                fusion_input = lstm_audio[batch_indices, effective_lengths - 1, :]

                text_feat = self.text_model(**text).last_hidden_state[:, 0, :]

                pred = self.fusion_model(torch.cat([fusion_input, text_feat], dim=1))
                return pred

            else:
                audio_out = self.audio_model(audio['input_values']).last_hidden_state
                transformer_output_length = audio_out.size(1)
                downsampling_factor = max_padded_length / transformer_output_length

                effective_lengths = torch.floor(raw_lengths.float() / downsampling_factor).long()
                effective_lengths = torch.clamp(effective_lengths, min=1)

                lstm_audio, _ = self.lstm_audio(audio_out)
                batch_indices = torch.arange(lstm_audio.size(0), device=lstm_audio.device)
                fusion_input = lstm_audio[batch_indices, effective_lengths - 1, :]

                text_feat = self.text_model(**text).last_hidden_state[:, 0, :]
                pred = self.fusion_model(torch.cat([fusion_input, text_feat], dim=1))
                return pred

