import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torchmetrics.regression import PearsonCorrCoef

def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):
    # set model train mode
    model.train()
    
    losses = AverageMeter()

    blackimgs = AverageMeter()

    corr = PearsonCorrCoef(num_outputs=6).to(train_config.device)
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for audio, vision, length,  text, label, avg in bar:
        if scaler:
            with autocast():
            
                # data (batches) to device   
                audio = {key: val.to(train_config.device) for key, val in audio.items()}
                vision = vision.to(train_config.device)
                label = label.to(train_config.device)
                text = {key: val.to(train_config.device) for key, val in text.items()}
                # Forward pass
                features = model(audio, vision, text, length)
                if train_config.mtl:
                    loss_mtl = loss_function(features, label)
                    loss = loss_mtl['loss']
                else:
                    loss = loss_function(features, label)

                losses.update(loss.item())
                
                  
            scaler.scale(loss).backward()
            
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                if train_config.mtl:
                    torch.nn.utils.clip_grad_value_(list(model.parameters()) + list(loss_function.parameters()), train_config.clip_grad)
                else:
                    torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            if step % train_config.gradient_accumulation == 0:
                # Update model parameters (weights)
                scaler.step(optimizer)
                scaler.update()

                # Zero gradients for next step
                optimizer.zero_grad()
                
                # Scheduler
                if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                    scheduler.step()
   
        else:

            # data (batches) to device   
            audio = {key: val.to(train_config.device) for key, val in audio.items()}
            text = {key: val.to(train_config.device) for key, val in text.items()}

            vision = vision.to(train_config.device)
            label = label.to(train_config.device)

            # Forward pass
            features = model(audio, vision, text)
            if train_config.mtl:
                loss_mtl = loss_function(features, label)
                loss = loss_mtl['loss']
            else:
                loss = loss_function(features, label)
            
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
                       
            if train_config.clip_grad:
                if train_config.mtl:
                    torch.nn.utils.clip_grad_value_(list(model.parameters()) + list(loss_function.parameters()), train_config.clip_grad)
                else:
                    torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)

            if step % train_config.gradient_accumulation == 0:
                # Update model parameters (weights)
                optimizer.step()
                # Zero gradients for next step
                optimizer.zero_grad()
                
                # Scheduler
                if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                    scheduler.step()
        
        
        
        if train_config.verbose:
            blackimgs.update(avg)
            if train_config.mtl:
                corr.update((loss_mtl['task_weight']/loss_mtl['task_weight'].sum())[0] * features[0] + (loss_mtl['task_weight']/loss_mtl['task_weight'].sum())[1] * features[1], label)
            else:
                corr.update(features, label)
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr": "{:.6f}".format(optimizer.param_groups[0]['lr']),
                       "cutoff": "{:.4f}".format(blackimgs.avg),
                       "corr": "{:.4f}".format(corr.compute().cpu().numpy().mean())}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    if train_config.mtl:
        print('task_loss:        {:.3f}, {:.3f}'.format(*list(loss_mtl['task_loss'])))
        print('task_weight:      {:.3f}, {:.3f}'.format(*list(loss_mtl['task_weight'])))
        print('task_uncertainty: {:.3f}, {:.3f}'.format(*list(loss_mtl['task_uncertainty'])))

    return losses.avg


