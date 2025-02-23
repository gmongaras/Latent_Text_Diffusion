import os
os.environ["HF_HOME"] = "/users/gmongaras/gmongaras_diffusion_models/Text_Diffusion/data/cache"
os.environ["HF_DATASETS_CACHE"] = "/users/gmongaras/gmongaras_diffusion_models/Text_Diffusion/data/cache"

from datasets import load_dataset
#ds = load_dataset("gmongaras/EleutherAI_the_pile_deduplicated", cache_dir="/users/gmongaras/gmongaras_diffusion_models/Text_Diffusion/data/cache", num_proc=16, split="train")
#del ds
ds = load_dataset("TrevorDohm/Stack_Tokenized", cache_dir="/users/gmongaras/gmongaras_diffusion_models/Text_Diffusion/data/cache", num_proc=16, split="train")
# Only get the text column
ds = ds.remove_columns(["input_ids", "attention_mask"])
# Rename the column to text
ds = ds.rename_column("content", "text")
ds.push_to_hub("gmongaras/Stack", token="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")
