"""Byte-level blocks and batches for the fixed CS 489 A2 protocol."""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from evaluator import BLOCK_LENGTH, BOS_ID, PAD_ID

IGNORE_INDEX = -100  # target value skipped by F.cross_entropy


def load_documents(data_cfg, split):
    """Return the documents (strings) of one split of the pinned HF dataset."""
    dataset = load_dataset(
        data_cfg["dataset_id"], revision=data_cfg["revision"], split=data_cfg[split]
    )
    return dataset[data_cfg["text_field"]]


def make_blocks(documents):
    """Cut each document's UTF-8 bytes into consecutive blocks of <= 256 bytes."""
    blocks = []
    for document in documents:
        encoded = document.encode("utf-8")
        for start in range(0, len(encoded), BLOCK_LENGTH):
            blocks.append(encoded[start : start + BLOCK_LENGTH])
    return blocks


def collate(blocks):
    """Build right-padded (input_ids, targets) tensors for a list of byte blocks.

    input = [BOS, b0..b(m-2)], target = [b0..b(m-1)]; padded targets are IGNORE_INDEX.
    """
    length = max(map(len, blocks))
    input_ids = torch.full((len(blocks), length), PAD_ID, dtype=torch.long)
    targets = torch.full((len(blocks), length), IGNORE_INDEX, dtype=torch.long)
    for row, block in enumerate(blocks):
        byte_ids = torch.tensor(list(block), dtype=torch.long)
        input_ids[row, 0] = BOS_ID
        input_ids[row, 1 : len(block)] = byte_ids[:-1]
        targets[row, : len(block)] = byte_ids
    return input_ids, targets


def make_loader(blocks, batch_size, shuffle, drop_last=False, **loader_kwargs):
    return DataLoader(
        blocks,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate,
        **loader_kwargs,
    )
