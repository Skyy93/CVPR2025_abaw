import os
import time
import math
import shutil
import sys
import torch
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from abaw.abaw_dataset import HumeDatasetEval, HumeDatasetTrain
from abaw.transforms import get_transforms_train, get_transforms_val
from abaw.utils import setup_system, Logger
from abaw.trainer import train
from abaw.evaluate import evaluate
from abaw.loss import MSE, CCC, MSECCC, CORR, MTLoss
from abaw.model import Model
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, \
    get_cosine_schedule_with_warmup
import pickle


##### Audio standard with 30 eps


@dataclass
class TrainingConfiguration:
    '''
    Describes configuration of the training process
    '''

    # Model
    #model: tuple = ('timm/convnext_base.fb_in22k_ft_in1k', 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim')#'facebook/wav2vec2-large-960h') # ('facebook/dinov2-small', 'hf-audio/wav2vec2-bert-CV16-en') or ('linear', 'linear')
    model: tuple = ('linear',
                    'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim',#'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim',#'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim')  # 'facebook/wav2vec2-large-960h') # ('facebook/dinov2-small', 'hf-audio/wav2vec2-bert-CV16-en') or ('linear', 'linear')
                    'Alibaba-NLP/gte-en-mlm-base',)
    # Training 
    mixed_precision: bool = True
    seed = 3407
    epochs: int = 30
    batch_size: int = 32 
    verbose: bool = True
    gpu_ids: tuple = (0,1,2,3)#,3)  # GPU ids for training
    task: str = "text+vit+audio"
    # Eval
    batch_size_eval: int = 32
    eval_every_n_epoch: int = 1  # eval every n Epoch

    # Optimizer 
    clip_grad = 100.  # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False  # Gradient Checkpointing

    # Loss
    loss: str = 'MSE'  # MSE, CCC, MSECCC choice wise
    mtl = False

    # Learning Rate
    lr: float = 1e-4  # 1
    # * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "cosine"  # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float = 0.0001  # only for "polynomial"
    gradient_accumulation: int = 1

    # Dataset
    data_folder = "./data/"

    # Savepath for model checkpoints
    model_path: str = "./hume_model"

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
# Train Config                                                                #
# -----------------------------------------------------------------------------#

config = TrainingConfiguration()

# %%

if __name__ == '__main__':

    model_path = "{}/{}/{}".format(config.model_path,
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

    model = Model(config.model, config.task)

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
                                     label_file='data/train_split.csv',
                                     #label_file='data/valid_split.csv',
                                     model=config.model,
                                     )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=True,
                                  pin_memory=True,
                                  collate_fn=train_dataset.collate_fn)

    # Reference Satellite Images
    eval_dataset = HumeDatasetEval(data_folder=config.data_folder,
                                   label_file='data/valid_split.csv',
                                   #label_file='data/train_split.csv',
                                   model=config.model,
                                   )

    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=config.batch_size_eval,
                                 num_workers=config.num_workers,
                                 shuffle=False,
                                 pin_memory=True,
                                 collate_fn=eval_dataset.collate_fn)

    print("Train Length:", len(train_dataset))
    print("Val Length:", len(eval_dataset))

    # -----------------------------------------------------------------------------#
    # Loss                                                                        #
    # -----------------------------------------------------------------------------#
    if config.loss == 'MSE':
        loss_function = MSE()
    elif config.loss == 'CCC':
        loss_function = CCC()
    elif config.loss == 'MSECCC':
        loss_function = MSECCC()
    elif config.loss == 'CORR':
        loss_function = CORR()
    else:
        raise ReferenceError("Loss function does not exist.")

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2. ** 10)
    else:
        scaler = None

    # -----------------------------------------------------------------------------#
    # optimizer                                                                   #
    # -----------------------------------------------------------------------------#

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

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

    if config.mtl:
        mtloss = MTLoss(num_task=2, loss_fn=loss_function)
        mtloss = torch.nn.DataParallel(mtloss, device_ids=config.gpu_ids)
        mtloss.train()
        mtloss.to(config.device)
        optimizer.add_param_group({'params': mtloss.parameters()})

    for epoch in range(1, config.epochs + 1):
        model.train()
        print("\n{}[Epoch: {}]{}".format(30 * "-", epoch, 30 * "-"))

        train_loss = train(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=mtloss if config.mtl else loss_function,
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

            p1, _, _ = evaluate(config=config,
                          model=model,
                          eval_dataloader=eval_dataloader,
                          weight=mtloss.module.weight if config.mtl else None)

            if p1 > best_score:

                best_score = p1

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, p1))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, p1))
            print("Epoch: {}, Eval Pearson = {:.3f},".format(epoch, p1))
        if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
            torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, p1))
        else:
            torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, p1))

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))
