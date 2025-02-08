import torch
from torch import nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import wandb
from tqdm import tqdm
import copy
import os
os.environ["HF_HOME"] = "data/cache"

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader

import datasets

from helpers.multi_gpu_helpers import is_main_process

from cut_cross_entropy import linear_cross_entropy


cpu = torch.device('cpu')



from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup
from transformers import get_constant_schedule_with_warmup

def get_scheduler(optimizer, num_warmup_steps, num_training_steps, use_lr_scheduler):
    """
    Creates a cosine scheduler with warmup
    """
    if use_lr_scheduler:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=0.5  # Standard half-cycle cosine
        )
    else:
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
        )
    return scheduler




def init_distributed():

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # Try the nccl backend
    try:
        dist.init_process_group(
                backend="nccl",
                init_method=dist_url,
                world_size=world_size,
                device_id=torch.device(f"cuda:{local_rank}"),
                rank=rank)
    # Use the gloo backend if nccl isn't supported
    except RuntimeError:
        dist.init_process_group(
                backend="gloo",
                init_method=dist_url,
                world_size=world_size,
                device_id=torch.device(f"cuda:{local_rank}"),
                rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()




# Trains a diffusion model
class model_trainer():
    # diff_model - A diffusion model to train
    # batchSize - Batch size to train the model with
    # numSteps - Number of steps to breakup the batchSize into. Instead
    #            of taking 1 massive step where the whole batch is loaded into
    #            memory, the batchSize is broken up into sizes of
    #            batchSize//numSteps so that it can fit into memory. Mathematically,
    #            the update will be the same, as a single batch update, but
    #            the update is distributed across smaller updates to fit into memory.
    # totalSteps - Number of steps to train the model for
    # lr - Learning rate of the model optimizer
    # device - Device to put the model and data on (gpu or cpu)
    # saveDir - Directory to save the model to
    # numSaveSteps - Number of steps until saving the models
    # use_importance - True to use importance sampling to sample values of t,
    #                  False to use uniform sampling.
    # p_uncond - Probability of training on a null class (only used if class info is used)
    # load_into_mem - True to load all data into memory first, False to load from disk as needed
    # optimFile - Optional name of optimizer to load in
    # schedulerFile - Optional name of scheduler to load in
    def __init__(self, 
            diff_model, 
            batchSize, 
            numSteps, 
            totalSteps, 
            lr, 
            ema_update_freq, 
            ema_decay, 
            warmup_steps,
            use_lr_scheduler,
            device, 
            saveDir, 
            numSaveSteps, 
            p_uncond=None, 
            KL_penalty_weight=1e-6,
            optimFile=None, 
            schedulerFile=None, 
            scalerFile=None, 
            use_amp=True, 
            wandb_name=None, 
            log_steps=10):
        # Saved info
        self.batchSize = batchSize//numSteps
        self.numSteps = numSteps
        self.totalSteps = totalSteps
        self.ema_update_freq = ema_update_freq
        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps
        self.use_lr_scheduler = use_lr_scheduler
        self.saveDir = saveDir
        self.numSaveSteps = numSaveSteps
        self.p_uncond = p_uncond
        self.KL_penalty_weight = KL_penalty_weight
        self.use_amp = use_amp
        self.wandb_name = wandb_name
        self.log_steps = log_steps
        
        # Convert the device to a torch device
        if device.lower() == "gpu":
            if torch.cuda.is_available():
                dev = device.lower()
                local_rank = int(os.environ['LOCAL_RANK'])
                device = torch.device(f"cuda:{local_rank}")
            else:
                dev = "cpu"
                print("GPU not available, defaulting to CPU. Please ignore this message if you do not wish to use a GPU\n")
                device = torch.device('cpu')
        else:
            dev = "cpu"
            device = torch.device('cpu')
        self.device = device
        self.dev = dev
        
        # Put the model on the desired device
        if dev != "cpu":
            # Initialize the environment
            init_distributed()

            self.model = DDP(diff_model.cuda(local_rank), device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
        else:
            self.model = diff_model.cpu()


        # EMA model on CPU to save GPU memory
        self.ema_model_cpu = copy.deepcopy(self.model.module).cpu()
        self.ema_model_cpu.eval()
        
        # Optimizer
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=lr, eps=1e-8, weight_decay=0.01, betas=(0.9, 0.999))

        # LR Scheduler
        self.scheduler = get_scheduler(self.optim, num_warmup_steps=warmup_steps, num_training_steps=totalSteps, use_lr_scheduler=use_lr_scheduler)

        # Automatic mixed precision gradient scalar
        if self.use_amp:
            self.grad_scaler = torch.amp.GradScaler("cuda")
        else:
            self.grad_scaler = None

        # Load in optimizer paramters if they exist
        if optimFile:
            self.optim.load_state_dict(torch.load(optimFile, map_location=self.device))

        # Load in scheduler paramters if they exist
        if schedulerFile:
            self.scheduler.load_state_dict(torch.load(schedulerFile, map_location=self.device))

        # Load in scalar paramters if they exist
        if scalerFile:
            self.grad_scaler.load_state_dict(torch.load(scalerFile, map_location=self.device))

        # Load in states from the pretrained diffusion model
        self.wandb_id = self.model.wandb_id if dev == "cpu" else self.model.module.wandb_id
        self.start_step = self.model.start_step if dev == "cpu" else self.model.module.start_step

        # Total params of the model
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Number of parameters in the model: {total_params:.2f}M")















    # Trains the model
    def train(self, dataset):
        # Put the model is train mode
        self.model.train()

        # Number of steps taken so far
        num_steps = self.start_step
        






        # Load in datasets
        os.environ["HF_HOME"] = "data/cache"
        os.environ["HF_DATASETS_CACHE"] = "data/cache"
        self.dataset = datasets.load_dataset(dataset, num_proc=16, split="train", cache_dir="data/cache")
        
        # # Tokenize the dummy dataset
        # if dataset == "gmongaras/dummy_text_dataset":
        #     def tokenize_function(examples):
        #         return self.model.tokenizer(examples["text"], truncation=False)
        #     self.tokenized_dataset = self.tokenized_dataset.map(
        #         tokenize_function,
        #         remove_columns=["text"],
        #         cache_file_name="dummy_tokenized_dataset",
        #     )
        
        def collate_fn(batch):
            batch = self.model.module.tokenizer([i["text"] for i in batch], 
                                               return_tensors="pt", 
                                               padding=True, #"max_length",
                                               truncation=True, 
                                               max_length=self.model.module.tokenizer.model_max_length)
            # Remove short sequences
            batch["input_ids"] = batch["input_ids"][batch["attention_mask"].sum(-1) > 4 + self.model.module.total_downscale_factor]
            batch["attention_mask"] = batch["attention_mask"][batch["attention_mask"].sum(-1) > 4 + self.model.module.total_downscale_factor]
            """ Causal
            # Labels are the input_ids shifted by one
            batch["labels"] = batch["input_ids"][:, 1:].to(batch["input_ids"].device)
            # Cutoff input_ids and attention_mask by 1
            batch["input_ids"] = batch["input_ids"][:, :-1]
            batch["attention_mask"] = batch["attention_mask"][:, :-1].bool()
            """
            #""" Bidirectional
            batch["attention_mask"] = batch["attention_mask"].bool()
            # Labels are the input_ids
            batch["labels"] = batch["input_ids"].to(batch["input_ids"].device)
            #"""

            # Mask labels with -100 where the attention mask is 0.
            batch["labels"] = torch.where(batch["attention_mask"], batch["labels"], torch.tensor(-100).to(batch["labels"].device))

            return batch

        if self.dev == "cpu":
            data_loader = DataLoader(self.dataset, batch_size=self.batchSize,
                pin_memory=True,
                drop_last=False, 
                shuffle=True,

                num_workers=5,
                prefetch_factor=5,
                persistent_workers=True,
                collate_fn=collate_fn,
            )
        else:
            data_loader = DataLoader(self.dataset, batch_size=self.batchSize,
                pin_memory=True,
                drop_last=False, 
                # sampler=DistributedSampler(dataset, shuffle=True, num_replicas=torch.distributed.get_world_size(), rank=torch.distributed.get_rank()),
                sampler=torch.utils.data.RandomSampler(self.dataset, replacement=True, num_samples=(self.totalSteps-num_steps)*self.batchSize),

                num_workers=5,
                prefetch_factor=5,
                persistent_workers=True,
                collate_fn=collate_fn,
            )

        

        # Losses over steps
        self.losses_comb = np.array([])
        self.steps_list = np.array([])

        # Cumulative loss over the batch over each set of steps
        losses_comb_s = torch.tensor(0.0, requires_grad=False)

        batch_loss = 0
        batch_recon_loss = 0
        batch_KL_loss = 0

        # Initialize wandb run
        if is_main_process():
            wandb.init(
                project="Latent_Test_Diffusion",
                name=self.wandb_name,
                notes=None, # May add notes later
                
                # Resume training if checkpoint exists
                resume="must" if self.wandb_id is not None else None,
                id=self.wandb_id,
            )
            wandb.watch(self.model, log_freq=self.log_steps)
            
            # Save wandb run id
            self.wandb_id = wandb.run.id
            self.model.wandb_id = self.wandb_id 
            if self.dev != "cpu":
                self.model.module.wandb_id = self.wandb_id

        # Sync all processes
        dist.barrier()

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Iterate over the desiered number of steps
        for step, data in enumerate(tqdm(data_loader, initial=num_steps, total=self.totalSteps)):
            # Get the batch
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            labels = data["labels"].to(self.device)

            if input_ids.shape[0] == 0:
                print("Batch too small.. continuing")
                continue
            
            # Increate the number of steps taken
            num_steps += 1

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if self.use_amp else nullcontext():
                # Logit prediction
                outputs, KL_loss = self.model(input_ids, mask=attention_mask, get_KL_loss=True, unembed=True)

                # Mask labels with -100 where the attention mask is 0.
                # labels = torch.where(attention_mask, labels, torch.tensor(-100).to(labels.device))
                
                # Loss 
                # loss_recon = linear_cross_entropy(outputs.to(torch.bfloat16).view(-1, self.model.module.defaults["dim_decoder"]), self.model.module.unembedding.weight.to(torch.bfloat16), labels.view(-1).to(outputs.device))
                loss_recon = loss_fct(outputs.view(-1, self.model.module.vocab_size), labels.view(-1).to(outputs.device))
                loss = loss_recon + (self.KL_penalty_weight*KL_loss)

            # Scale the loss to be consistent with the batch size. If the loss
            # isn't scaled, then the loss will be treated as an independent
            # batch for each step. If it is scaled by the step size, then the loss will
            # be treated as a part of a larger batchsize which is what we want
            # to acheive when using steps.
            loss = loss/self.numSteps

            # Backpropagate loss
            if self.use_amp:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            # Save the loss values
            losses_comb_s += loss.cpu().detach()
            batch_loss += loss.cpu().detach().item()
            batch_recon_loss += loss_recon.cpu().detach().item()
            batch_KL_loss += KL_loss.cpu().detach().item()

            # If the number of steps taken is a multiple of the number
            # of desired steps, update the models
            if num_steps%self.numSteps == 0:
                # Unscale gradients
                if self.use_amp:
                    self.grad_scaler.unscale_(self.optim, )

                # Clip gradients
                if self.use_amp:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update the model using all losses over the steps
                if self.use_amp:
                    self.grad_scaler.step(self.optim)
                else:
                    self.optim.step()

                # Update scheduler
                self.scheduler.step(step)
                    
                # Step the gradient scaler
                if self.use_amp:
                    self.grad_scaler.update()
                            
                # Zero gradients 
                self.optim.zero_grad()

                # if is_main_process():
                #     print(f"step #{num_steps}   Latest loss estimate: {round(losses_comb_s.cpu().detach().item(), 6)}")

                # Log wandb
                if num_steps % self.log_steps == 0:
                    batch_loss = batch_loss/self.log_steps
                    batch_recon_loss = batch_recon_loss/self.log_steps
                    batch_KL_loss = batch_KL_loss/self.log_steps
                    
                    if is_main_process():
                        wandb.log({
                            "loss": batch_loss,
                            "perplexity": torch.exp(torch.tensor(batch_loss)).item(),
                            "recon loss": batch_recon_loss,
                            "KL loss": batch_KL_loss,
                            "lr": self.optim.param_groups[0]['lr'],
                        },
                        step=num_steps)
                    
                    batch_loss = 0
                    batch_recon_loss = 0
                    batch_KL_loss = 0

                # Save the loss values
                self.losses_comb = np.append(self.losses_comb, losses_comb_s.item())
                self.steps_list = np.append(self.steps_list, num_steps)

                # Reset the cumulative step loss
                losses_comb_s *= 0


            # Update EMA on CPU every `update_frequency` batches
            if num_steps % self.ema_update_freq == 0:
                with torch.no_grad():
                    for ema_param, param in zip(self.ema_model_cpu.parameters(), self.model.module.parameters()):
                        if param.requires_grad:
                            ema_param.data.mul_(self.ema_decay).add_(param.cpu().data, alpha=(1.0 - self.ema_decay))


            # Save the EMA model and graph every number of desired steps
            if num_steps%self.numSaveSteps == 0 and is_main_process():
                self.ema_model_cpu.wandb_id = self.wandb_id
                self.ema_model_cpu.saveModel(self.saveDir, self.optim, self.scheduler, self.grad_scaler, num_steps)
                # self.graph_losses()

                print("Saving model")
        
        # if is_main_process():
        #     print(f"Loss at step #{num_steps}, update #{num_steps/self.numSteps}\n"+\
        #             f"Combined: {round(self.losses_comb[-10:].mean(), 4)}\n\n")


