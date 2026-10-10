"""Training loop shared by web_train.py.

Press Ctrl+C during training to stop early; the model is saved before exiting.
"""

import os
import time
from dataclasses import asdict

import torch

from .model import LamConfig, LamLLM
from .tokenizer import CharTokenizer


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"  # Intel GPU
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device, iters=20):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(model, tok, out):
    os.makedirs(out, exist_ok=True)
    torch.save({"config": asdict(model.cfg), "model": model.state_dict()}, os.path.join(out, "lamllm.pt"))
    tok.save(os.path.join(out, "tokenizer.json"))


def train_model(text, out, steps=2000, batch_size=32, block_size=128, n_layer=4, n_head=4,
                n_embd=128, dropout=0.1, lr=3e-4, eval_every=200, seed=1337):
    torch.manual_seed(seed)
    device = get_device()

    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}
    block_size = min(block_size, len(splits["val"]) - 2)

    cfg = LamConfig(
        vocab_size=tok.vocab_size, block_size=block_size,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=dropout,
    )
    model = LamLLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr / 10)

    print(f"device={device} chars={len(text):,} vocab={tok.vocab_size} params={model.num_params():,}")

    def timing(step):
        elapsed = time.time() - t0
        if step == 0:
            return "estimating time left..."
        eta = elapsed / step * (steps - step)
        return f"{elapsed / 60:.1f}m elapsed, ~{eta / 60:.0f}m left"

    t0 = time.time()
    try:
        for step in range(steps + 1):
            if step % eval_every == 0 or step == steps:
                losses = estimate_loss(model, splits, block_size, batch_size, device)
                line = f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {timing(step)}"
                print("\r" + line.ljust(90), flush=True)
                if step:
                    save_checkpoint(model, tok, out)
            if step == steps:
                break
            x, y = get_batch(splits["train"], block_size, batch_size, device)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if (step + 1) % 10 == 0:
                # Progress line, overwritten in place until the next eval line.
                line = f"  step {step + 1}/{steps} | loss {loss.item():.4f} | {timing(step + 1)}"
                print("\r" + line.ljust(90), end="", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted, saving current model...")

    save_checkpoint(model, tok, out)
    print(f"saved checkpoint to {out}/")
    return model, tok
