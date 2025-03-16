import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torchmetrics.classification import BinaryF1Score


def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):
    model.train()
    losses = AverageMeter()
    blackimgs = AverageMeter()
    f1_metric = BinaryF1Score().to(train_config.device)    

    time.sleep(0.1)    
    optimizer.zero_grad(set_to_none=True)
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for audio, vision, length, text, att_pos, label_id, avg in bar:
        if scaler:
            with autocast():
                # data (batches) to device   
                audio = {key: val.to(train_config.device) for key, val in audio.items()}
                vision = vision.to(train_config.device)
                label_id = label_id.to(train_config.device)
                text = {key: val.to(train_config.device) for key, val in text.items()}
                att_pos = att_pos.to(train_config.device)
                # Forward pass

                logit = model(audio, vision, text, att_pos, length)

                loss = loss_function(logit.squeeze(), label_id)
                losses.update(loss.item())
                # Calculate prediciton for F1 calculation
                label_id = label_id.reshape(-1)
                prediction = torch.sigmoid(logit).reshape(-1)
                
            scaler.scale(loss).backward()
            
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            if step % train_config.gradient_accumulation == 0:
                # Update model parameters (weights)
                scaler.step(optimizer)
                scaler.update()
                # Zero gradients for next step
                optimizer.zero_grad()
                # Scheduler
                if train_config.scheduler in ["polynomial", "cosine", "constant"]:
                    scheduler.step()
   
        else:
            # data (batches) to device   
            audio = {key: val.to(train_config.device) for key, val in audio.items()}
            vision = vision.to(train_config.device)
            label_id = label_id.to(train_config.device)
            text = {key: val.to(train_config.device) for key, val in text.items()}
            att_pos = att_pos.to(train_config.device)
            # Forward pass
            logit = model(audio, vision, text, att_pos, length)
            loss = loss_function(logit.squeeze(), label_id)
            losses.update(loss.item())
            # Calculate gradient using backward pass
            loss.backward()
            # Calculate prediciton for F1 calculation
            label_id = label_id.reshape(-1)
            prediction = torch.sigmoid(logit).reshape(-1)
                       
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)        
                        
            if step % train_config.gradient_accumulation == 0:
                # Update model parameters (weights)
                optimizer.step()
                # Zero gradients for next step
                optimizer.zero_grad()
                # Scheduler
                if train_config.scheduler in ["polynomial", "cosine", "constant"]:
                    scheduler.step()

        if train_config.verbose:
            blackimgs.update(avg)
            f1_metric.update((prediction > 0.5).int(), label_id.int())
            current_f1 = f1_metric.compute()
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr": "{:.6f}".format(optimizer.param_groups[0]['lr']),
                       "cutoff": "{:.4f}".format(blackimgs.avg),
                       "f1": "{:.4f}".format(current_f1)
            }
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


