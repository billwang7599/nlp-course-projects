"""Train and evaluate the Part 1 LSTM byte-level language model."""

import argparse
import copy
import csv
import os
from pathlib import Path

import matplotlib
import torch
import wandb
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_documents, make_blocks, make_loader
from evaluator import VOCAB_SIZE, evaluate
from model import LSTMLanguageModel, next_token_loss


def resolve_paths(cfg, config_dir):
    """Rewrite the config's relative local paths to be relative to the YAML's directory."""
    cfg["wandb"]["dir"] = str(config_dir / cfg["wandb"]["dir"])
    for key in ("artifact_dir", "metrics_file", "summary_file", "plot_file"):
        cfg["output"][key] = str(config_dir / cfg["output"][key])


def build_model(model_cfg):
    """Build the (untrained) LSTM language model from the config's model section."""
    return LSTMLanguageModel(
        VOCAB_SIZE,
        model_cfg["embedding_dim"],
        model_cfg["dropout"],
        model_cfg["hidden_dim"],
        model_cfg["num_layers"],
    )


def export_config(cfg, artifact_dir):
    """Save the settings used to artifact_dir/config.yaml, with paths relative to that file."""
    exported = copy.deepcopy(cfg)
    exported["wandb"]["dir"] = os.path.relpath(cfg["wandb"]["dir"], artifact_dir)
    for key in ("artifact_dir", "metrics_file", "summary_file", "plot_file"):
        exported["output"][key] = os.path.relpath(cfg["output"][key], artifact_dir)
    with open(artifact_dir / "config.yaml", "w") as f:
        yaml.safe_dump(exported, f, sort_keys=False)


def pick_device(requested):
    """Use the requested device if available, otherwise fall back to mps, then cpu."""
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    # Step 1: parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # Step 2: load the YAML and resolve its relative paths
    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    resolve_paths(cfg, config_path.parent)

    # Step 3: seed
    torch.manual_seed(cfg["seed"])

    # Step 4: device
    device = pick_device(cfg["runtime"]["device"])
    print(f"seed={cfg['seed']}, device={device}")

    # Step 5: data (dev stays as raw strings, since evaluator.evaluate does its own blocking)
    train_docs = load_documents(cfg["data"], "train_split")
    dev_docs = load_documents(cfg["data"], "dev_split")
    train_loader = make_loader(
        make_blocks(train_docs),
        cfg["training"]["batch_size"],
        shuffle=cfg["training"]["shuffle"],
        drop_last=cfg["training"]["drop_last"],
    )

    # Step 6: model
    model = build_model(cfg["model"]).to(device)

    # Step 7: optimizer
    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt_cfg["learning_rate"],
        betas=tuple(opt_cfg["betas"]),
        eps=opt_cfg["epsilon"],
        weight_decay=opt_cfg["weight_decay"],
        amsgrad=opt_cfg["amsgrad"],
    )

    # Step 8: W&B (WANDB_MODE in the environment overrides the YAML mode, e.g. to go online)
    wandb_cfg = cfg["wandb"]
    os.makedirs(wandb_cfg["dir"], exist_ok=True)
    wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg["entity"],
        name=wandb_cfg["name"],
        mode=os.environ.get("WANDB_MODE", wandb_cfg["mode"]),
        dir=wandb_cfg["dir"],
        config=cfg,
    )
    wandb.define_metric("train/step")
    wandb.define_metric("train/loss", step_metric="train/step")
    wandb.define_metric("epoch")
    wandb.define_metric("dev/*", step_metric="epoch")

    step = 0
    best_ppl = float("inf")
    train_rows = []  # (epoch, step, loss)
    dev_rows = []  # (epoch, loss, ppl)
    max_norm = cfg["training"]["gradient_clip_norm"]
    checkpoint_path = Path(cfg["output"]["artifact_dir"]) / "checkpoint" / "model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def predict_logits(input_ids, attention_mask):
        with torch.no_grad():
            logits = model(torch.from_numpy(input_ids).to(device))
        return logits.cpu().numpy()

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        for input_ids, targets in train_loader:
            input_ids, targets = input_ids.to(device), targets.to(device)
            optimizer.zero_grad() # makes each step independant, dont use stale gradients
            loss = next_token_loss(model(input_ids), targets)
            loss.backward() # stores each steps gradient in each weights' tensor, so its in memory already
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step() # changes the weights in place
            step += 1
            wandb.log({"train/step": step, "train/loss": loss.item()})
            train_rows.append((epoch, step, loss.item()))

        model.eval()
        result = evaluate(predict_logits, dev_docs, batch_size=cfg["evaluation"]["batch_size"])
        wandb.log({"epoch": epoch, "dev/loss": result["loss"], "dev/ppl": result["ppl"]})
        dev_rows.append((epoch, result["loss"], result["ppl"]))
        print(f"epoch {epoch}: dev loss {result['loss']:.4f}, dev ppl {result['ppl']:.3f}")

        if result["ppl"] < best_ppl:
            best_ppl = result["ppl"]
            torch.save(model.state_dict(), checkpoint_path)

    wandb.summary["best_dev_ppl"] = best_ppl
    print(f"best dev ppl {best_ppl:.3f}")

    metrics_file = Path(cfg["output"]["metrics_file"])
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "epoch", "step", "loss", "ppl"])
        for epoch, s, loss in train_rows:
            writer.writerow(["train", epoch, s, loss, ""])
        for epoch, loss, ppl in dev_rows:
            writer.writerow(["dev", epoch, "", loss, ppl])

    fig, (ax_train, ax_dev) = plt.subplots(1, 2, figsize=(10, 4))
    ax_train.plot([r[1] for r in train_rows], [r[2] for r in train_rows])
    ax_train.set(xlabel="training step", ylabel="train loss")
    ax_dev.plot([r[0] for r in dev_rows], [r[1] for r in dev_rows], marker="o")
    ax_dev.set(xlabel="epoch", ylabel="dev loss")
    fig.tight_layout()
    Path(cfg["output"]["plot_file"]).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg["output"]["plot_file"])

    export_config(cfg, Path(cfg["output"]["artifact_dir"]))
    wandb.finish()


if __name__ == "__main__":
    main()
