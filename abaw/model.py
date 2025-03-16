import torch

import torch.nn as nn

from transformers import AutoModel
from abaw.audeer import EmotionModel

class Model(nn.Module):
    def __init__(self,
                 model_name,
                 sr,
                 wave_length_s,
                 tokenizer_len
                 ):

        super(Model, self).__init__()
        self.model_name = model_name
        self.sr = sr
        self.wave_length_s = wave_length_s

        if model_name[0] == "googlevit":
            googlevit_feat = 768 
            self.lstm_vis = nn.LSTM(1280, googlevit_feat, num_layers=2, batch_first=True, bidirectional=False)
        else:
            googlevit_feat = 0

        if model_name[1] != None:
            if "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim" in model_name[1]:
                self.audio_model = EmotionModel.from_pretrained(model_name[1])
                audio_feat = 1027  
            else:
                self.audio_model = AutoModel.from_pretrained(model_name[1]) 
                if "large" in model_name[1] or "wav2vec2-lg-xlsr" in model_name[1] or "r-f/wav2vec" in model_name[1]:
                    audio_feat = 1024
                elif "base" in model_name[1]:
                    audio_feat = 768
            self.lstm_audio = nn.LSTM(audio_feat, audio_feat, num_layers=2, batch_first=True, bidirectional=False)
        else:
            audio_feat = 0

        if model_name[2] != None:
            self.text_model = AutoModel.from_pretrained(model_name[2], trust_remote_code=True)
            # Resize Token Embeddings (To learn new custom special tokens)
            self.text_model.resize_token_embeddings(tokenizer_len)
            if "large" in model_name[2]:
                text_feat = 1024
            elif "base" in model_name[2]:
                text_feat = 768
        else:
            text_feat = 0        

        self.fusion_model = nn.Sequential(nn.Linear(googlevit_feat + audio_feat + text_feat, 1027),
                                            nn.Tanh(),
                                            nn.Linear(1027, 1),
                                            )

    def forward(self, audio, vision, text, attention_token_pos, length):

        features = []

        ## Generate Audio Features!
        if self.model_name[1] != None:
            raw_lengths = audio['attention_mask'].sum(dim=1) 
            max_padded_length = round(self.wave_length_s * self.sr)

            if "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim" in self.model_name[1]:
                audio_output = self.audio_model(audio['input_values'])
                audio_out = torch.cat((audio_output[0], audio_output[1]), dim=2)
            else:
                audio_out = self.audio_model(audio['input_values']).last_hidden_state

            transformer_output_length = audio_out.size(1)
            downsampling_factor = max_padded_length / transformer_output_length

            effective_lengths = torch.floor(raw_lengths.float() / downsampling_factor).long()
            effective_lengths = torch.clamp(effective_lengths, min=1)

            lstm_audio, _ = self.lstm_audio(audio_out)

            batch_indices = torch.arange(lstm_audio.size(0), device=lstm_audio.device)
            audio_input = lstm_audio[batch_indices, effective_lengths - 1, :]
            features.append(audio_input)

        ## Generate Text Features!
        if self.model_name[2] != None:
            last_hidden_states = self.text_model(**text)["last_hidden_state"]
            batch_indices = torch.arange(last_hidden_states.size(0), device=last_hidden_states.device)
            text_input = last_hidden_states[batch_indices, attention_token_pos, :]
            features.append(text_input)
        
        ## (use) Generate(d) Text Features!
        if self.model_name[0] == "googlevit":
            lstm_output,_ = self.lstm_vis(vision) 
            batch_indices = torch.arange(lstm_output.size(0))
            vision = lstm_output[batch_indices, length.cpu() - 1, :]
            features.append(vision)

        if len(features) > 1:
            fusion_input = torch.cat(features, dim=1)
        else:
            fusion_input = features[0]
            
        pred = self.fusion_model(fusion_input)

        return pred