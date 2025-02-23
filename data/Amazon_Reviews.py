import os
os.environ["HF_HOME"] = "data/cache"
os.environ["HF_DATASETS_CACHE"] = "data/cache"
import datasets
from datasets import load_dataset

dataset_names = ['raw_review_All_Beauty', 'raw_review_Toys_and_Games', 'raw_review_Cell_Phones_and_Accessories', 'raw_review_Industrial_and_Scientific', 'raw_review_Gift_Cards', 'raw_review_Musical_Instruments', 'raw_review_Electronics', 'raw_review_Handmade_Products', 'raw_review_Arts_Crafts_and_Sewing', 'raw_review_Baby_Products', 'raw_review_Health_and_Household', 'raw_review_Office_Products', 'raw_review_Digital_Music', 'raw_review_Grocery_and_Gourmet_Food', 'raw_review_Sports_and_Outdoors', 'raw_review_Home_and_Kitchen', 'raw_review_Subscription_Boxes', 'raw_review_Tools_and_Home_Improvement', 'raw_review_Pet_Supplies', 'raw_review_Video_Games', 'raw_review_Kindle_Store', 'raw_review_Clothing_Shoes_and_Jewelry', 'raw_review_Patio_Lawn_and_Garden', 'raw_review_Unknown', 'raw_review_Books', 'raw_review_Automotive', 'raw_review_CDs_and_Vinyl', 'raw_review_Beauty_and_Personal_Care', 'raw_review_Amazon_Fashion', 'raw_review_Magazine_Subscriptions', 'raw_review_Software', 'raw_review_Health_and_Personal_Care', 'raw_review_Appliances', 'raw_review_Movies_and_TV']

with open(".env", "r") as f:
    token = f.read()

# Load in the DatasetDict
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", dataset_names[0], trust_remote_code=True, split="full", cache_dir="data/Amazon_Reviews", num_proc=128)

# Load in all other datasets and concatenate
from datasets import concatenate_datasets
for i in range(1, len(dataset_names)):
    dataset = concatenate_datasets([
        dataset, 
        load_dataset("McAuley-Lab/Amazon-Reviews-2023", dataset_names[i], trust_remote_code=True, split="full", cache_dir="data/Amazon_Reviews", num_proc=128)
    ])

# Save the dataset
dataset.push_to_hub("gmongaras/Amazon-Reviews-2023", token=token, num_shards=150)