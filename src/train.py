import pickle
from models.diff_model import diff_model
from model_trainer import model_trainer
import os
import click
from typing import List





def train():
    totalSteps = 250_000
    batchSize = 512
    input_dim = 1024
    num_classes = 5
    num_blocks = 17
    model_max_length = 256
    dim = int(64*num_blocks)
    c_dim = 1024
    hidden_scale = 4.0
    num_heads = num_blocks
    attn_type = "softmax"
    device = "gpu"
    wandb_name = "attempt1"
    log_steps = 10
    p_uncond = 0.1
    lr = 1e-4
    use_lr_scheduler = False
    ema_update_freq = 100
    ema_decay = 0.99
    warmup_steps = 1000
    positional_encoding = "RoPE" # "absolute" or "RoPE"

    numSaveSteps = 10_000
    saveDir = "models/attempt1"

    loadModel = False
    loadDir = "models/attempt1"
    loadFile = "model_220000s.pkl"
    loadDefFile = "model_params_220000s.json"
    optimFile = "optim_220000s.pkl"
    schedulerFile = "scheduler_220000s.pkl"
    scalerFile = "scaler_220000s.pkl"

    # VAE_loadDir = "models/causal_45BS_KL1e-6"
    VAE_loadDir = "models/causal_45BS_KL1e-6_nosmallseq_maskfix"
    VAE_loadFile = "model_170000s.pkl"
    VAE_loadDefFile = "model_params_170000s.json"
    
    
    
    ### Model Creation
    model = diff_model(
        input_dim=input_dim,
        num_classes=num_classes,
        dim=dim,
        c_dim=c_dim,
        hidden_scale=hidden_scale,
        num_heads=num_heads,
        attn_type=attn_type,
        num_blocks=num_blocks,
        model_max_length=model_max_length,
        positional_encoding=positional_encoding,
        VAE_loadDir=VAE_loadDir,
        VAE_loadFile=VAE_loadFile,
        VAE_loadDefFile=VAE_loadDefFile,
        device=device,
    )
    
    # Optional model loading
    if loadModel == True:
        model.loadModel(loadDir, loadFile, loadDefFile)
    
    # Train the model
    trainer = model_trainer(
        diff_model=model,
        batchSize=batchSize, 
        numSteps=1,
        totalSteps=totalSteps, 
        lr=lr, 
        ema_update_freq=ema_update_freq,
        ema_decay=ema_decay,
        warmup_steps=warmup_steps,
        use_lr_scheduler=use_lr_scheduler,
        saveDir=saveDir,
        numSaveSteps=numSaveSteps,
        p_uncond=p_uncond,
        optimFile=None if loadModel==False or optimFile==None else loadDir+os.sep+optimFile,
        schedulerFile=None if loadModel==False or schedulerFile==None else loadDir+os.sep+schedulerFile,
        scalerFile=None if loadModel==False or scalerFile==None else loadDir+os.sep+scalerFile,
        use_amp=True,
        wandb_name=wandb_name,
        log_steps=log_steps,
        device=device,
    )
    trainer.train()
    
    
    
    
    
if __name__ == '__main__':
    train()
