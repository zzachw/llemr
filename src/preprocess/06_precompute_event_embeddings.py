import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.utils import create_directory, processed_data_path, set_seed

DEVICE = "cuda:0"
BATCH_SIZE = 4096


def get_event_list(output_path, hadm_id):
    df = pd.read_csv(os.path.join(output_path, f"event_selected/event_{hadm_id}.csv"))
    text = []
    for i, row in df.iterrows():
        text.append(row.event_value)
    return text


class Data(Dataset):
    def __init__(self, output_path, hadm_id):
        self.event_list = get_event_list(output_path, hadm_id)

    def __getitem__(self, index):
        return self.event_list[index]

    def __len__(self):
        return len(self.event_list)


def get_embeddings(model, tokenizer, loader):
    all_embeddings = []
    for text in loader:
        with torch.no_grad():
            text_tokenized = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            text_tokenized = text_tokenized.to(DEVICE)
            embeddings = model(**text_tokenized).last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings.cpu())
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings


def save_embeddings(emb, output_path, hadm_id):
    torch.save(emb, os.path.join(output_path, f"pt_event_selected_no_time_type/event_{hadm_id}.pt"))


def main():
    set_seed(seed=42)
    output_path = os.path.join(processed_data_path, "mimic4")

    cohort = pd.read_csv(os.path.join(output_path, "cohort.csv"))
    print(f"Cohort read: {cohort.shape}")
    hadm_ids = set(cohort.hadm_id.unique().tolist())
    print(f"Unique hadm_ids: {len(hadm_ids)}")

    model = AutoModel.from_pretrained("microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract")
    model.eval()
    model.to(DEVICE)

    create_directory(os.path.join(output_path, f"pt_event_selected_no_time_type"))
    for hadm_id in tqdm(hadm_ids):
        if os.path.exists(os.path.join(output_path, f"pt_event_selected_no_time_type/event_{hadm_id}.pt")):
            continue
        data = Data(output_path, hadm_id)
        loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)
        emb = get_embeddings(model, tokenizer, loader)
        save_embeddings(emb, output_path, hadm_id)


if __name__ == '__main__':
    main()
