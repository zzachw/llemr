import logging
import os
import re

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils import processed_data_path


class InstructionTuningDataset(Dataset):
    def __init__(self, split, source):
        assert split in ["train", "val", "test", "test_subset"]
        assert source in ["event", "note", "joint", "joint_all"]
        self.split = split
        self.source = source
        self.data_path = os.path.join(processed_data_path, f"mimic4")
        self.cohort = pd.read_csv(os.path.join(self.data_path, f"cohort_{split}.csv"))

        qa_note = pd.read_json(os.path.join(self.data_path, "qa_note.jsonl"), lines=True)
        qa_note = qa_note[qa_note.hadm_id.isin(self.cohort.hadm_id.unique())]

        qa_event = pd.read_json(os.path.join(self.data_path, f"qa_event.jsonl"), lines=True)
        qa_event = qa_event[qa_event.hadm_id.isin(self.cohort.hadm_id.unique())]

        if source == "note":
            qa = qa_note
        elif source == "event":
            qa = qa_event
        else:
            if source == "joint":
                logging.warning(f"Subsample event QA to {len(qa_note)}")
                qa_event = qa_event.sample(n=len(qa_note), replace=False, random_state=42)
            else:
                logging.warning(f"Use all event QA")
            qa = pd.concat([qa_note, qa_event], ignore_index=True)

        self.qa = qa
        logging.warning(f"Loaded {len(qa)} {source} QA samples for {split} on {event}")

    def _get_event_list(self, hadm_id):
        df = pd.read_csv(os.path.join(self.data_path, f"event_selected/event_{hadm_id}.csv"))
        event_list = []
        for i, row in df.iterrows():
            event_list.append((row.timestamp, row.event_type, row.event_value))
        return event_list

    def _get_event_emb(self, hadm_id):
        return torch.load(os.path.join(self.data_path, f"pt_event_selected_no_time_type/event_{hadm_id}.pt"))

    def __len__(self):
        return len(self.qa)

    @staticmethod
    def _extract_digits(event_tuple):
        timestamp, event_type, event_value = event_tuple
        try:
            if event_type == "patient demographics" or event_type == "patient_demographics":
                value_match = re.search(r"age:\s*([\d.]+)", event_value)
                if value_match:
                    value = float(value_match.group(1))
                else:
                    value = 0
                duration = 0
            elif event_type == "admission info" or event_type == "admission_info":
                value, duration = 0, 0
            elif event_type == "diagnoses_icd":
                value, duration = 0, 0
            elif event_type == "labevents":
                value_match = re.search(r":\s*([\d.]+)", event_value)
                if value_match:
                    value = float(value_match.group(1))
                else:
                    value = 0
                duration = 0
            elif event_type == "microbiologyevents":
                value, duration = 0, 0
            elif event_type == "prescriptions":
                value_match = re.search(r"prescribed dose:\s*([\d.]+)", event_value)
                if value_match:
                    value = float(value_match.group(1))
                else:
                    value = 0
                duration_match = re.search(r"duration:\s*([\d.]+)", event_value)
                if duration_match:
                    duration = float(duration_match.group(1))
                else:
                    duration = 0
            elif event_type == "transfers":
                value, duration = 0, 0
            elif event_type == "procedureevents":
                value = 0
                duration_match = re.search(r"for\s*([\d.]+)\s*hour", event_value)
                if duration_match:
                    duration = float(duration_match.group(1))
                else:
                    duration = 0
            else:
                raise ValueError(f"Unknown event type: {event_type}")
        except Exception as e:
            value, duration = 0, 0
            logging.warning(f"Error {e} in extracting digits from event tuple: {event_tuple}")
        return value, duration

    def __getitem__(self, index):
        data = self.qa.iloc[index]
        q = data["q"]
        a = data["a"]
        event_emb = self._get_event_emb(data["hadm_id"])
        num_events = event_emb.shape[0]

        event_list = self._get_event_list(data["hadm_id"])
        assert len(event_list) == num_events
        time_tensor = torch.tensor([[e[0]] for e in event_list], dtype=torch.float32)
        value_duration_tensor = torch.tensor([self._extract_digits(e) for e in event_list], dtype=torch.float32)
        event_emb = torch.cat(
            [
                event_emb,
                time_tensor,
                value_duration_tensor,
            ],
            dim=1
        )
        final_q = "\n".join(["<image>" * num_events, q])

        return final_q, a, event_emb
