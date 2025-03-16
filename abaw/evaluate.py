import time
import torch

import numpy as np

from tqdm import tqdm
from torch.cuda.amp import autocast
from sklearn.metrics import f1_score
from torchmetrics.classification import BinaryF1Score

def evaluate(config, model, eval_dataloader):
    with torch.no_grad():
        predictions, labels = predict(config, model, eval_dataloader)
        predictions = predictions.squeeze()
        
        print(f"max prediction value: {predictions.float().max()}")
        print(f"min prediction value: {predictions.float().min()}")

        f1 = f1_score(labels, (predictions > 0.50).int(), average="weighted")
        print(f"f1 {f1} at treshold 0.50")
        #f1_max = f1

        f1 = f1_score(labels, (predictions > 0.55).int(), average="weighted")
        print(f"f1 {f1} at treshold 0.55")

        f1 = f1_score(labels, (predictions > 0.60).int(), average="weighted")
        print(f"f1 {f1} at treshold 0.60")

        f1 = f1_score(labels, (predictions > 0.65).int(), average="weighted")
        print(f"f1 {f1} at treshold 0.65")

        threshold_max = 0.5
        f1_max = 0
        # Iterate from 0.2 to 0.6 (exclusive) in steps of 0.025
        for threshold in np.arange(0.40, 0.75, 0.001):
            # Compute F1 score with weighted average; ensure predictions are converted to int
            f1 = f1_score(labels, (predictions > threshold).int().squeeze(), average="weighted")
            if f1 > f1_max:
                f1_max = f1
                threshold_max = threshold
        
        print(f"max f1 {f1_max} at treshold {threshold_max}")

    return f1_max.item(), predictions

def predict(train_config, model, dataloader):
    model.eval()
    f1_metric = BinaryF1Score().to(train_config.device)
    time.sleep(0.1)

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    predictions = []
    labels = []
    with torch.no_grad():
        for audio, vision, length, text, att_pos, label_id in bar:
            with autocast():
                # data (batches) to device   
                audio = {key: val.to(train_config.device) for key, val in audio.items()}
                vision = vision.to(train_config.device)
                label_id = label_id.to(train_config.device)
                text = {key: val.to(train_config.device) for key, val in text.items()}
                att_pos = att_pos.to(train_config.device)
                # Forward pass
                logit = model(audio, vision, text, att_pos, length)
                # Calculate prediciton for F1 calculation
                label_id = label_id.reshape(-1)
                prediction = torch.sigmoid(logit).reshape(-1)

            # save features in fp32 for sim calculation
            labels.append(label_id.detach().cpu())
            predictions.append(prediction.to(torch.float32).detach().cpu())
            f1_metric.update((prediction > 0.5).int(), label_id.int())
            current_f1 = f1_metric.compute()
            bar.set_postfix(f1=current_f1.cpu().numpy())
        # keep Features on GPU
        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)

    if train_config.verbose:
        bar.close()

    return predictions, labels