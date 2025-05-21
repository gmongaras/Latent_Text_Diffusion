import torch
from models.Autoencoder import Autoencoder
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import click
import random
import pickle







def infer():
    # loadDir = "models_VAE/nocausal_60BS_KL1e-6_V2"
    # loadFile = "model_180000s.pkl"
    # loadDefFile = "model_params_180000s.json"

    loadDir = "models_VAE/causal_60BS_KL1e-6_128_latent_space"
    loadFile = "model_150000s.pkl"
    loadDefFile = "model_params_150000s.json"

    
    
    
    
    
    ### Model Creation

    # Create a dummy model
    dim_encoder = 256
    dim_decoder = 256
    latent_dim = 128
    hidden_scale = 2.0
    num_heads = 8
    attn_type = "cosine"
    num_blocks_enc = 4
    num_blocks_dec = 4
    encoder_seq_downscale_factor = 2
    encoder_out_dims = [256, 512, 512, 1024]
    encoder_num_inner_blocks = 4
    positional_encoding = "absolute"
    max_seq_len = 1024
    use_cross_attn_decoder = True
    device = "gpu"
    causal_VAE=True
    seed = -1
    text = [
        "Mario is a game about a plumber who goes on an adventure to save a princess. He has to jump over obstacles, collect coins, and defeat enemies. The game is known for its simple gameplay and iconic characters. Mario has become one of the most popular video game characters of all time. He has appeared in over 200 video games and has sold over 600 million copies worldwide. Mario has also appeared in movies, TV shows, and comic books. He is a cultural icon and a symbol of the video game industry. Mario has inspired many other video game characters and has had a huge impact on popular culture. He is loved by people of all ages and is considered one of the greatest video game characters of all time.",
        # "I'm definitely going to look into this a bit more. Where can I find more information about how synchronization works? Having trouble locating this. Can it get through firewalls? Is the entire tree always synchronized?"
        # "“Jargon is part of office life and while it can often be regarded as baffling and frustrating, there are advantages to speaking the office lingo, the Telegraph quoted David Clubb, the managing director of Office Angels, as saying. “These can range from bonding with a team to understanding mind-boggling conference calls, he added. (ANI)",
        # "The quick brown fox jumped over the lazy dog",
        # "I like cats."
        # "The number 01010011 as an integer is 82."
        # "01010011001000438189248926"
        # "Thoughts of summer\n\nSummer is in full swing and with it the urge to be outside. Many people these days are unemployed and get to be outside, but for those who are still employed and have the fantasy of quitting their job and being their own boss, this is a rough time of year.\n\nWarm summer days and free concerts at the Santa Monica Pier breed an urgency in one to get out of the office, to begin to play in the sand, to find a summer lover and enjoy the warm days ahead with a sandwich from Bay Cities Deli on the beach.\n\nThe heat of the summer, and the lack of need for clothing motivates some of us to renew those January resolutions to lose weight, and to take a look at our life and re-evaluate where we are physically and professionally."
    ]

    model = Autoencoder(
        dim_encoder=dim_encoder,
        dim_decoder=dim_decoder, 
        latent_dim=latent_dim,
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
        causal_cheat=False,
        device=device, 
    )
    
    # Load in the model weights
    model.loadModel(loadDir, loadFile, loadDefFile)



    # Load on device
    if device == "gpu":
        model = model.cuda()
    else:
        model = model
    model.device = model.mean_proj.weight.device

    
    def enocde_text(text):
        batch = model.tokenizer(text, 
                                return_tensors="pt", 
                                padding=True,
                                truncation=True, 
                                max_length=model.tokenizer.model_max_length+1)
        # Remove short sequences
        batch["input_ids"] = batch["input_ids"]#[batch["attention_mask"].sum(-1) > 4 + model.total_downscale_factor]
        batch["attention_mask"] = batch["attention_mask"]#[batch["attention_mask"].sum(-1) > 4 + model.total_downscale_factor]
        return batch
    
    batch = enocde_text(text).to(model.device)

    # Send the batch to the model
    model.eval()
    output = model.infer(x_t=batch["input_ids"], mask=batch["attention_mask"], sample=True)
    print(model.tokenizer.decode(output[0]))
    # output.softmax(-1)[0].index_select(-1, batch["input_ids"][0])
    
    
    
    
    
if __name__ == '__main__':
    infer()
