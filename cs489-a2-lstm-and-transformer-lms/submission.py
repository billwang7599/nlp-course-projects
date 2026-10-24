"""Load a trained checkpoint and evaluate it with the course evaluator.

Usage: uv run python submission.py --config artifacts/part1/config.yaml [--split dev_split]
"""

import argparse
from pathlib import Path

import torch
import yaml

from data import load_documents
from evaluator import evaluate
from run_part1 import build_model as build_lstm
from run_part1 import pick_device, resolve_paths
from run_part2 import build_model as build_transformer


def load_model(config_path, device=None):
    """Rebuild the model from a config.yaml and load its best checkpoint, in eval mode."""
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    resolve_paths(cfg, config_path.parent)
    builders = {"lstm": build_lstm, "transformer": build_transformer}
    model_type = cfg["model"]["type"]
    if model_type not in builders:
        raise ValueError(f"unsupported model type: {model_type}")
    device = device or pick_device(cfg["runtime"]["device"])
    model = builders[model_type](cfg["model"]).to(device)
    checkpoint = Path(cfg["output"]["artifact_dir"]) / "checkpoint" / "model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model, cfg, device


def make_predict_logits(model, device):
    """Wrap the model as the NumPy callback that evaluator.evaluate expects."""

    def predict_logits(input_ids, attention_mask):
        with torch.no_grad():
            logits = model(torch.from_numpy(input_ids).to(device))
        return logits.cpu().numpy()

    return predict_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="dev_split", help="a split key from the config's data section")
    args = parser.parse_args()

    model, cfg, device = load_model(args.config)
    documents = load_documents(cfg["data"], args.split)
    result = evaluate(
        make_predict_logits(model, device), documents, batch_size=cfg["evaluation"]["batch_size"]
    )
    print(f"{args.split}: loss {result['loss']:.4f}, ppl {result['ppl']:.3f}")


if __name__ == "__main__":
    main()
