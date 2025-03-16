import os
import torch

import pandas as pd
import numpy as np

from dataclasses import dataclass
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from abaw.model import Model
from abaw.evaluate import predict
from abaw.abaw_dataset import HumeDatasetEval, SpecialTokens

from sklearn.metrics import f1_score

@dataclass
class InterferenceConfiguration:
    # Model
    model: tuple = ('googlevit',
                    None,#'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim',
                    'Alibaba-NLP/gte-large-en-v1.5',)  # 'bert-base-uncased'

    vit_frame_width: int = 20 * 24 # (12s * 24 FPS  )
    max_googlevit_feat_length: int = 400
    wave_length_s: float = 12.0
    text_context_width_s: float = 20.0

    # Predict
    treshold: float = 0.570 # for preds -> 0 or 1
    batch_size: int = 32
    verbose: bool = True
    gpu_ids: tuple = (0,1,2,3)  # GPU ids for training

    # Test-Dataset
    # TODO: CHANGE TO TEST
    predict_csv_path: str = "/Coding/CVPR2025_abaw_framewise/data_abaw/test_data/test.csv"
    #######################

    # TODO: CHANGE TO TEST
    data_folder: str = "/Coding/CVPR2025_abaw_framewise/data_abaw/test_data"
    #######################

    fps: float = 24
    sr: float = 16 * 1e3

    # Path for model checkpoint to load
    model_checkpint_path: str = "/Coding/CVPR2025_abaw_framewise/data_abaw/test_data/weights/eval_train_trained/005631/weights_e5_0.7255.pth"

    # Path to save predicitons
    save_path_predictions: str = "data_abaw/test_data/output/trial-0.txt"

    # TODO: CHANGE TO TEST
    example_txt_path: str = "/Coding/CVPR2025_abaw_framewise/data_abaw/test_data/submissions (examples)/trial-3.txt" # "/Coding/CVPR2025_abaw_framewise/data_abaw/test_data/submissions (examples)/trial-0.txt"
    #######################

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4

    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = InterferenceConfiguration() 

if __name__ == '__main__':

    output_submission_path = config.save_path_predictions
    print(f'\nModel: {config.model}')
    print(f'Used .csv file for evaluating: {config.predict_csv_path}')

    if config.model[2] != None:
        processor_text = AutoTokenizer.from_pretrained(config.model[2])
        processor_text.add_special_tokens(SpecialTokens.generate_special_token_dict())
        processor_len = len(processor_text)
    else:
        processor_text = None
        processor_len = 0

    model = Model(config.model, config.sr, config.wave_length_s, tokenizer_len=processor_len)
    
    # load Checkpoint    
    print('Start from:', config.model_checkpint_path)
    model_state_dict = torch.load(config.model_checkpint_path)  
    model.load_state_dict(model_state_dict, strict=False)     

    # Data parallel
    print('\nGPUs available:', torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    # Model to device   
    model = model.to(config.device)

    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#
    
    # Predict
    predict_dataset = HumeDatasetEval(data_folder=config.data_folder,
                                      label_file_csv=config.predict_csv_path,
                                      model=config.model,
                                      processor_text=processor_text,
                                      sr=config.sr,
                                      fps=config.fps,
                                      eval_fps=config.fps,
                                      vit_frame_width=config.vit_frame_width,
                                      max_googlevit_feat_length=config.max_googlevit_feat_length,
                                      wave_length_s=config.wave_length_s,
                                      text_context_width_s=
                                      config.text_context_width_s,
                                      )

    predict_dataloader = DataLoader(predict_dataset,
                                 batch_size=config.batch_size,
                                 num_workers=config.num_workers,
                                 shuffle=False,
                                 pin_memory=True,
                                 collate_fn=predict_dataset.collate_fn
                                 )

    print("len(predict_dataset):", len(predict_dataset))

    #-----------------------------------------------------------------------------#
    # Evaluate                                                                    #
    #-----------------------------------------------------------------------------#
    
    print('\n{}[{}]{}'.format(30*'-', 'Predict!!!', 30*'-'))  

    probs, _ = predict(config, model, predict_dataloader)
    test_df = pd.read_csv(config.predict_csv_path)

    result_df = predict_dataset.meta
    sub_df = pd.read_csv(config.example_txt_path, names=["general_path", "class_id"], sep=",", header=None)

    print(len(result_df))
    print(len(sub_df))

    if result_df['general_path'].apply(os.path.dirname).equals(sub_df['general_path'].apply(os.path.dirname)):
        print("The 'general_path' columns are identical and in the same order.")
    else:
        print("The 'general_path' columns differ or are in a different order.")
        print(os.path.dirname(result_df['general_path']))
        print(os.path.dirname(sub_df['general_path']))

    y_true = sub_df["class_id"]

    predictions = (probs > config.treshold).int().cpu().numpy()
    sub_df["class_id"] = predictions
    sub_df.to_csv(output_submission_path, index=False, header=False)

    print("The prediction is complete!🎉🚀✨")

    #  # Testing (take eval.csv)
    #  
    #  y_true = sub_df["class_id"]
    #  
    #  f1 = f1_score(y_true, predictions, average="weighted")
    #  
    #  print("F1_w score:", f1)

