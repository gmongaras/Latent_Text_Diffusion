import pickle
from models.Autoencoder import Autoencoder
from model_trainer import model_trainer
import os
import click
from typing import List





def train():
    totalSteps = 250_000
    batchSize = 62
    dim_encoder = 512
    dim_decoder = 1024
    num_blocks_enc = 3
    num_blocks_dec = 7
    encoder_seq_downscale_factor = 2
    encoder_out_dims = [512, 768, 1024]
    encoder_num_inner_blocks = 2
    hidden_scale = 4.0
    num_heads = 8
    attn_type = "cosine"
    device = "gpu"
    wandb_name = "nocausal_62BS_KL1e-6_V2"
    log_steps = 10
    lr = 1e-4
    use_lr_scheduler = False
    ema_update_freq = 100
    ema_decay = 0.99
    warmup_steps = 1000
    positional_encoding = "RoPE" # "absolute" or "RoPE"
    # dataset = "TrevorDohm/Stack_Tokenized"
    # dataset = "gmongaras/dummy_text_dataset"
    dataset = "gmongaras/EleutherAI_the_pile_deduplicated"
    max_seq_len = 1024
    KL_penalty_weight = 1e-6
    use_cross_attn_decoder = False
    causal_VAE = False
    causal_cheat = False

    numSaveSteps = 10_000
    saveDir = "models_VAE/nocausal_62BS_KL1e-6_V2"

    loadModel = False
    loadDir = "models_VAE/nocausal_62BS_KL1e-6_V2"
    loadFile = "model_90000s.pkl"
    loadDefFile = "model_params_90000s.json"
    optimFile = "optim_90000s.pkl"
    schedulerFile = "scheduler_90000s.pkl"
    scalerFile = "scaler_90000s.pkl"
    
    
    
    ### Model Creation
    model = Autoencoder(
        dim_encoder=dim_encoder,
        dim_decoder=dim_decoder, 
        hidden_scale=hidden_scale, 
        num_heads=num_heads, 
        attn_type=attn_type, 
        num_blocks_enc=num_blocks_enc, 
        num_blocks_dec=num_blocks_dec, 
        encoder_seq_downscale_factor=encoder_seq_downscale_factor, 
        encoder_out_dims=encoder_out_dims,
        encoder_num_inner_blocks=encoder_num_inner_blocks,
        positional_encoding=positional_encoding, 
        max_seq_len=max_seq_len,
        use_cross_attn_decoder=use_cross_attn_decoder,
        causal_VAE=causal_VAE,
        causal_cheat=causal_cheat,
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
        KL_penalty_weight=KL_penalty_weight,
        optimFile=None if loadModel==False or optimFile==None else loadDir+os.sep+optimFile,
        schedulerFile=None if loadModel==False or schedulerFile==None else loadDir+os.sep+schedulerFile,
        scalerFile=None if loadModel==False or scalerFile==None else loadDir+os.sep+scalerFile,
        use_amp=True,
        wandb_name=wandb_name,
        log_steps=log_steps,
        device=device,
    )
    trainer.train(dataset)
    
    
    
    
    
if __name__ == '__main__':
    train()
