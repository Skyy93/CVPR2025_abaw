import os
import sys
import math
import time
import torch
import shutil
import audiomentations

from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup, AutoTokenizer

from abaw.model import Model
from abaw.trainer import train
from abaw.evaluate import evaluate
from abaw.utils import setup_system, Logger
from abaw.loss import MSE, CCC, MSECCC, CORR
from abaw.abaw_dataset import HumeDatasetEval, HumeDatasetTrain, SpecialTokens
from abaw.transforms import get_transforms_train_wave, get_transforms_train_wave_custom


@dataclass
class TrainingConfiguration:
    '''
    Describes configuration of the training process
    '''

    # Model
    model: tuple = ( 'googlevit', # googlevit or None
                      None, # ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
                     'Alibaba-NLP/gte-large-en-v1.5',) #'Alibaba-NLP/gte-large-en-v1.5',) #'Alibaba-NLP/gte-en-mlm-base',)  # 'bert-base-uncased'
    
    # Alibaba-NLP/gte-large-en-v1.5
    # Alibaba-NLP/gte-en-mlm-large
    # Alibaba-NLP/gte-Qwen2-1.5B-instruct

    vit_frame_width: int = 20 * 24 # (12s * 24 FPS  ) # 40?
    max_googlevit_feat_length: int = 400
    wave_length_s: float = 12.0
    text_context_width_s: float = 20.0  # 40?
    eval_fps: float = 0.3               # An evaluation is conducted every X seconds during the given samples. X = eval_fps

    # Training 
    balanced_sampling: bool = True
    mixed_precision: bool = True
    seed = 10
    epochs: int = 30
    freeze_epoch_text_model:int = 30
    freeze_epoch_audio_model:int = 30
    batch_size: int = 32
    verbose: bool = True
    gpu_ids: tuple = (0,1,2,3)  # GPU ids for training

    # Text-Augmentations
    enable_text_augs = False
    prob_synonym = 0.01                 # 0.01  # prob_synonym
    prob_deletion = 0.01                   # 0.01  # prob_deletion  
    prob_swap = 0.00                       # 0.00  # prob_swap
    prob_bert = 0.025                    # 0.025 # prob_bert

    # Sound-Augmentations
    # default_sound_augs = [
    #     audiomentations.AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.15, p=0.5),
    #     audiomentations.TimeStretch(min_rate=0.95, max_rate=1.05, p=0.5),
    #     audiomentations.PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    #     audiomentations.Shift(min_shift=-0.1, max_shift=0.1, p=0.5),

    # Sound-Augmentations (best choose)
    # custom_sound_augs = [
    #     audiomentations.AddGaussianNoise(min_amplitude=0.025, max_amplitude=0.25, p=0.6), # modified -> 0.507 / 0.506
    #     audiomentations.TimeStretch(min_rate=0.95, max_rate=1.05, p=0.5),
    #     audiomentations.PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    #     audiomentations.Shift(min_shift=-0.1, max_shift=0.1, p=0.5),

    enable_custom_sound_augs = False
    custom_sound_augs = [
        audiomentations.AddGaussianNoise(min_amplitude=0.025, max_amplitude=0.25, p=0.6),  # BEST -> 0.5065
        audiomentations.TimeStretch(min_rate=0.95, max_rate=1.05, p=0.5),                  # BEST -> 0.5065 (unverändert)
        audiomentations.PitchShift(min_semitones=-2, max_semitones=2, p=0.5),              # BEST -> 0.5065 (unverändert)
        audiomentations.Shift(min_shift=-0.1, max_shift=0.1, p=0.5),                       # BEST -> 0.5065 (unverändert)
    ]

    # Eval
    batch_size_eval: int = 32
    eval_every_n_epoch: int = 1  # eval every n Epoch

    # Optimizer 
    clip_grad = 100.  # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False  # Gradient Checkpointing

    # Loss
    loss: str = 'BCEWithLogitsLoss'  # BCEWithLogitsLoss, Focal, Dice choice wise  ##  (MSE, CCC, MSECCC) <- meh
    pos_weight: float = torch.tensor(1)

    # Learning Rate
    fact = 0.075
    lr: float = fact * 1e-4 # 7 * 1e-4      # 7 * 1e-4 
    lr_text: float = fact * 1e-4 # 0.1 * 1e-4
    lr_audio: float = fact * 1e-4
    lr_lstm: float = fact * 1e-4 # 0.7 * 1e-4  # 0.7 * 1e-4 -> Verbesserung auf 0.505
    # * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "cosine"  # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float = 1e-6  # only for "polynomial"
    gradient_accumulation: int = 1

    # Dataset
    data_folder:str = "./data_abaw/"
    fps:float = 24
    sr:float = 16 * 1e3
    # -> within mp4 => 24 FPS

    # Savepath for model checkpoints
    model_path: str = "./abaw_model"

    # Checkpoint to start from
    checkpoint_start = None

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4

    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for better performance
    cudnn_benchmark: bool = True

    # make cudnn deterministic
    cudnn_deterministic: bool = False


# -----------------------------------------------------------------------------#
# Train Config                                                                 #
# -----------------------------------------------------------------------------#

config = TrainingConfiguration()

####

if __name__ == '__main__':

    info_env = os.getenv("INFO")

    model_path = "{}/{}/{}/{}".format(config.model_path,
                                      info_env,
                                      config.model,
                                      time.strftime("%H%M%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    # -----------------------------------------------------------------------------#
    # Model                                                                       #
    # -----------------------------------------------------------------------------#

    print("\nModel: {}".format(config.model))

    if config.model[2] != None:
        processor_text = AutoTokenizer.from_pretrained(config.model[2])
        processor_text.add_special_tokens(SpecialTokens.generate_special_token_dict())
        processor_len = len(processor_text)
    else:
        processor_text = None
        processor_len = 0

    model = Model(config.model, config.sr, config.wave_length_s, tokenizer_len=processor_len)

    # load pretrained Checkpoint    
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

        # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    #if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    # Model to device   
    model = model.to(config.device)

    # -----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    # -----------------------------------------------------------------------------#

    # Train
    train_dataset = HumeDatasetTrain(data_folder=config.data_folder,
                                     label_file_csv=f"data_abaw/splits/train_and_eval.csv",
                                     model=config.model,
                                     processor_text=processor_text,
                                     sr=config.sr,
                                     fps=config.fps,
                                     vit_frame_width=config.vit_frame_width,
                                     max_googlevit_feat_length=config.max_googlevit_feat_length,
                                     wave_length_s=config.wave_length_s,
                                     wave_transforms=get_transforms_train_wave() if not config.enable_custom_sound_augs else get_transforms_train_wave_custom(config.custom_sound_augs),
                                     enable_text_augs=config.enable_text_augs,
                                     text_context_width_s=config.text_context_width_s,
                                     config=config
                                     )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.balanced_sampling,
                                  pin_memory=True,
                                  collate_fn=train_dataset.collate_fn
                                  )

    # Eval
    eval_dataset = HumeDatasetEval(data_folder=config.data_folder,
                                   label_file_csv=f"data_abaw/splits/val.csv",
                                   model=config.model,
                                   processor_text=processor_text,
                                   sr=config.sr,
                                   fps=config.fps,
                                   eval_fps=config.eval_fps, 
                                   vit_frame_width=config.vit_frame_width,
                                   max_googlevit_feat_length=config.max_googlevit_feat_length,
                                   wave_length_s=config.wave_length_s,
                                   text_context_width_s=config.text_context_width_s,
                                   )

    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=config.batch_size_eval,
                                 num_workers=config.num_workers,
                                 shuffle=False,
                                 pin_memory=True,
                                 collate_fn=eval_dataset.collate_fn
                                 )

    print("Train Length:", len(train_dataset))
    print("Val Length:", len(eval_dataset))

    # -----------------------------------------------------------------------------#
    # Loss                                                                        #
    # -----------------------------------------------------------------------------#

    def dice_loss(pred, target, smooth=1e-6):
        intersection = (pred * target).sum(dim=0)
        union = pred.sum(dim=0) + target.sum(dim=0)
        dice_coeff = (2. * intersection + smooth) / (union + smooth)
        loss = 1 - dice_coeff
        return loss
    
    class FocalLoss(torch.nn.Module):
        def __init__(self, alpha=1, gamma=4, reduction='mean'):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction
            self.smoothing = 0.10
        
        def forward(self, inputs, targets):
            targets_smoothed = targets  * (1 - self.smoothing) + self.smoothing / 2
            # Binary Cross Entropy Loss ohne Reduktion
            BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets_smoothed, reduction='none')
            pt = torch.exp(-BCE_loss)  # pt entspricht der Wahrscheinlichkeit der korrekten Klassifizierung
            focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
            
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss


    if config.loss == 'MSE':
        loss_function = MSE()
    elif config.loss == 'CCC':
        loss_function = CCC()
    elif config.loss == 'MSECCC':
        loss_function = MSECCC()
    elif config.loss == 'CORR':
        loss_function = CORR()
    elif config.loss == 'BCEWithLogitsLoss':
        loss_function = BCEWithLogitsLoss() #pos_weight=config.pos_weight)
    elif config.loss == 'Dice':
        loss_function = dice_loss
    elif config.loss == 'Focal':
        loss_function = FocalLoss()
    else:
        raise ReferenceError("Loss function does not exist.")

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2. ** 10)
    else:
        scaler = None

    # -----------------------------------------------------------------------------#
    # optimizer                                                                   #
    # -----------------------------------------------------------------------------#
    named_params = list(model.named_parameters())

    linear_params = [(n, p) for n, p in named_params if not ("text_model" in n or "audio_model" in n or "lstm_audio" in n)]
    text_params = [(n, p) for n, p in named_params if ("text_model" in n)]
    audio_params = [(n, p) for n, p in named_params if ("audio_model" in n)]
    lstm_params = [(n, p) for n, p in named_params if "lstm_audio" in n]

    if config.decay_exclue_bias:  
        no_decay = ['bias', 'LayerNorm.bias']
        params_for_AdamW = [
            { 
                'params': [p for n, p in linear_params if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
                'lr': config.lr
            },
            { 
                'params': [p for n, p in linear_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': config.lr
            },
            { 
                'params': [p for n, p in text_params if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
                'lr': config.lr_text
            },
            { 
                'params': [p for n, p in text_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': config.lr_text
            },
            { 
                'params': [p for n, p in audio_params if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
                'lr': config.lr_audio
            },
            { 
                'params': [p for n, p in audio_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': config.lr_audio
            },
            { 
                'params': [p for n, p in lstm_params if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
                'lr': config.lr_lstm
            },
            { 
                'params': [p for n, p in lstm_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': config.lr_lstm
            },
        ]

    else:
        params_for_AdamW = [
            { 
                'params': [p for n, p in linear_params],
                'lr': config.lr
            },
            { 
                'params': [p for n, p in text_params],
                'lr': config.lr_text
            },
            { 
                'params': [p for n, p in audio_params],
                'lr': config.lr_audio
            },
            { 
                'params': [p for n, p in lstm_params],
                'lr': config.lr_lstm
            },
        ]

    optimizer = torch.optim.AdamW(params_for_AdamW)

    # -----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    # -----------------------------------------------------------------------------#

    train_steps = math.floor((len(train_dataloader) * config.epochs) / config.gradient_accumulation)
    warmup_steps = len(train_dataloader) * config.warmup_epochs

    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end=config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)

    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)

    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=warmup_steps)

    else:
        scheduler = None

    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))

    # -----------------------------------------------------------------------------#
    # Train                                                                       #
    # -----------------------------------------------------------------------------#
    start_epoch = 0
    best_score = 0

    for epoch in range(1, config.epochs + 1):
        model.train()

        if config.model[2] and epoch >= config.freeze_epoch_text_model:  
            for param in model.module.text_model.parameters():
                param.requires_grad = False
            model.module.text_model.eval()

        if config.model[1] and epoch >= config.freeze_epoch_audio_model:  
            for param in model.module.audio_model.parameters():
                param.requires_grad = False
            model.module.audio_model.eval()

        print("\n{}[Epoch: {}]{}".format(30 * "-", epoch, 30 * "-"))

        if config.balanced_sampling:
            train_dataset.sample_balanced()

        train_loss = train(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler)

        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))

        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
            model.eval()
            print("\n{}[{}]{}".format(30 * "-", "Evaluate", 30 * "-"))

            p1, _ = evaluate(config=config,
                             model=model,
                             eval_dataloader=eval_dataloader
                             )

            if p1 > best_score:

                best_score = p1

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, p1))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, p1))
            print("Epoch: {}, Eval F1 = {:.3f},".format(epoch, p1))
        if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
            torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, p1))
        else:
            torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, p1))

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))
