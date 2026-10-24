"""Talk to a trained byte-level LM (LSTM or Transformer) in a terminal.

Usage: uv run python chat.py [--config artifacts/part1/config.yaml]

These models were trained only on Wikipedia text for next-byte prediction, so they
continue what you type instead of answering it.
"""

import argparse

import torch

from evaluator import BLOCK_LENGTH, BOS_ID
from submission import load_model

SHORTCUTS = {"lstm": "artifacts/part1/config.yaml", "transformer": "artifacts/part2/config.yaml"}
HELP = """commands:
  /model lstm|transformer|<path to config.yaml>   swap the model
  /temp <float>    sampling temperature (lower = safer, higher = wilder)
  /topk <int>      sample only from the k most likely bytes
  /len <int>       number of bytes to generate per turn
  /help, /quit"""


def generate(model, device, prompt, length, temperature, top_k):
    """Continue `prompt` by `length` bytes, sampling one byte at a time."""
    ids = list(prompt.encode("utf-8"))
    generated = []
    for _ in range(length):
        # Models were trained on blocks of <= 256 bytes that start with BOS.
        context = [BOS_ID] + (ids + generated)[-(BLOCK_LENGTH - 1) :]
        with torch.no_grad():
            logits = model(torch.tensor([context], device=device))[0, -1]
        logits = logits[:256] / temperature  # only real bytes; never sample BOS or PAD
        if top_k:
            threshold = torch.topk(logits, min(top_k, 256)).values[-1]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        generated.append(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())
    return bytes(generated).decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=SHORTCUTS["lstm"])
    args = parser.parse_args()

    config = args.config
    model, _, device = load_model(config)
    temperature, top_k, length = 0.8, 40, 200
    print(f"loaded {config} on {device}. Type some text and the model continues it. /help for commands.")

    while True:
        try:
            text = input("\nyou> ")
        except EOFError:
            break
        if not text.strip():
            continue
        if text.startswith("/"):
            command, _, argument = text.partition(" ")
            argument = argument.strip()
            if command == "/quit":
                break
            elif command == "/help":
                print(HELP)
            elif command == "/model":
                path = SHORTCUTS.get(argument, argument)
                try:
                    model, _, device = load_model(path)
                    config = path
                    print(f"loaded {config} on {device}")
                except (OSError, KeyError, ValueError, RuntimeError) as error:
                    print(f"could not load {path}: {error}")
            elif command in ("/temp", "/topk", "/len"):
                try:
                    if command == "/temp":
                        temperature = float(argument)
                    elif command == "/topk":
                        top_k = int(argument)
                    else:
                        length = int(argument)
                except ValueError:
                    print(f"{command} needs a number")
                if temperature <= 0 or length <= 0 or top_k < 0:
                    print("temp and len must be positive, topk non-negative; resetting to defaults")
                    temperature, top_k, length = 0.8, 40, 200
            else:
                print("unknown command, try /help")
            continue
        print("model>", text + generate(model, device, text, length, temperature, top_k))


if __name__ == "__main__":
    main()
