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
    num_blocks = 21
    model_max_length = 256
    dim = int(64*num_blocks)
    c_dim = 1024
    hidden_scale = 4.0
    num_heads = num_blocks
    attn_type = "flash"
    device = "gpu"
    wandb_name = "attempt7_causal_epspred"
    log_steps = 10
    p_uncond = 0.1
    lr = 1e-4
    use_lr_scheduler = False
    ema_update_freq = 100
    ema_decay = 0.99
    warmup_steps = 1000
    positional_encoding = "RoPE" # "absolute" or "RoPE"
    output_type = "eps" # "velocity", "x", or "eps"

    numSaveSteps = 10_000
    saveDir = "models_Diff/attempt7_causal_epspred"

    loadModel = False
    loadDir = "models_Diff/attempt7_causal_epspred"
    loadFile = "model_150000s.pkl"
    loadDefFile = "model_params_150000s.json"
    optimFile = "optim_150000s.pkl"
    schedulerFile = "scheduler_150000s.pkl"
    scalerFile = "scaler_150000s.pkl"

    # VAE_loadDir = "models/causal_45BS_KL1e-6"
    VAE_loadDir = "models_VAE/causal_60BS_KL1e-6_128_latent_space"
    VAE_loadFile = "model_150000s.pkl"
    VAE_loadDefFile = "model_params_150000s.json"
    
    
    
    ### Model Creation
    model = diff_model(
        num_classes=num_classes,
        dim=dim,
        c_dim=c_dim,
        hidden_scale=hidden_scale,
        num_heads=num_heads,
        attn_type=attn_type,
        num_blocks=num_blocks,
        model_max_length=model_max_length,
        positional_encoding=positional_encoding,
        output_type=output_type,
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
