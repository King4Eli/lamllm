"""Training loop used by websearch/multi_train.py.

Press Ctrl+C during training to stop early; the model is saved before exiting.
"""

import os
import threading
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
    x = torch.stack([data[i:i + block_size] for i in ix]).long()
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).long()
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


def save_checkpoint(model, tok, out, opt=None, trained_steps=0):
    os.makedirs(out, exist_ok=True)
    ckpt = {"config": asdict(model.cfg), "model": model.state_dict(), "trained_steps": trained_steps}
    if opt is not None:
        ckpt["optimizer"] = opt.state_dict()
    torch.save(ckpt, os.path.join(out, "lamllm.pt"))
    tok.save(os.path.join(out, "tokenizer.json"))


def checkpoint_exists(ckpt_dir):
    return os.path.exists(os.path.join(ckpt_dir, "lamllm.pt"))


def load_checkpoint(ckpt_dir, device="cpu"):
    """Returns (model, tokenizer, checkpoint dict)."""
    ckpt = torch.load(os.path.join(ckpt_dir, "lamllm.pt"), map_location=device)
    tok = CharTokenizer.load(os.path.join(ckpt_dir, "tokenizer.json"))
    model = LamLLM(LamConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model, tok, ckpt


class Corpus:
    """Training data that can keep growing while the model trains.

    Other threads call add() with new documents; the training loop calls
    refresh() to fold them in. Every 10th document goes to the validation split.
    """

    def __init__(self, tok, val_every=10):
        self.tok = tok
        self.val_every = val_every
        self.splits = {"train": torch.empty(0, dtype=torch.uint8), "val": torch.empty(0, dtype=torch.uint8)}
        self.n_docs = 0
        self._pending = []
        self._lock = threading.Lock()

    def add(self, text):
        with self._lock:
            self._pending.append(text)

    def refresh(self):
        """Move pending documents into the splits. Returns how many were added."""
        with self._lock:
            pending, self._pending = self._pending, []
        if not pending:
            return 0
        new = {"train": [], "val": []}
        for text in pending:
            self.n_docs += 1
            split = "val" if self.n_docs % self.val_every == 0 else "train"
            new[split].append(torch.tensor(self.tok.encode(text + "\n\n"), dtype=torch.uint8))
        for split, parts in new.items():
            if parts:
                self.splits[split] = torch.cat([self.splits[split], *parts])
        return len(pending)

    def size(self, split):
        return len(self.splits[split])


def fit(corpus, out, steps=2000, batch_size=32, block_size=128, n_layer=4, n_head=4,
        n_embd=128, dropout=0.1, lr=3e-4, eval_every=200, refresh_every=50, seed=1337, resume_from=None):
    """Train on `corpus`, pulling in newly added documents every `refresh_every` steps.

    With `resume_from` (a checkpoint dir), training continues from that model; its
    architecture is kept and the size arguments are ignored. The corpus must use the
    checkpoint's tokenizer (see tokenizer_for).
    """
    torch.manual_seed(seed)
    device = get_device()
    tok = corpus.tok
    corpus.refresh()

    trained_steps = 0
    if resume_from:
        model, ckpt_tok, ckpt = load_checkpoint(resume_from, device)
        assert ckpt_tok.chars == tok.chars, "corpus must use the checkpoint's tokenizer"
        block_size = model.cfg.block_size
        trained_steps = ckpt.get("trained_steps", 0)
    else:
        block_size = min(block_size, corpus.size("val") - 2)
        cfg = LamConfig(
            vocab_size=tok.vocab_size, block_size=block_size,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=dropout,
        )
        model = LamLLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    if resume_from and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] = lr
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr / 10)

    def data_size():
        return f"{(corpus.size('train') + corpus.size('val')) / 1e6:.2f} MB"

    if resume_from:
        so_far = f"{trained_steps:,} steps trained so far" if trained_steps else "earlier steps not recorded"
        print(f"continuing from {resume_from} ({so_far})")
    else:
        print("starting a new model")
    print(f"device={device} data={data_size()} vocab={tok.vocab_size} params={model.num_params():,}")

    def timing(step):
        elapsed = time.time() - t0
        if step == 0:
            return "estimating time left..."
        eta = elapsed / step * (steps - step)
        return f"{elapsed / 60:.1f}m elapsed, ~{eta / 60:.0f}m left"

    t0 = time.time()
    done = 0
    try:
        for step in range(steps + 1):
            if step and (step % refresh_every == 0 or step % eval_every == 0):
                corpus.refresh()
            if step % eval_every == 0 or step == steps:
                losses = estimate_loss(model, corpus.splits, block_size, batch_size, device)
                line = (f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                        f"| data {data_size()} | {timing(step)}")
                print("\r" + line.ljust(100), flush=True)
                if step:
                    save_checkpoint(model, tok, out, opt, trained_steps + step)
            if step == steps:
                break
            x, y = get_batch(corpus.splits["train"], block_size, batch_size, device)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            done = step + 1
            if (step + 1) % 10 == 0:
                # Progress line, overwritten in place until the next eval line.
                line = f"  step {step + 1}/{steps} | loss {loss.item():.4f} | {timing(step + 1)}"
                print("\r" + line.ljust(100), end="", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted, saving current model...")

    save_checkpoint(model, tok, out, opt, trained_steps + done)
    print(f"saved checkpoint to {out}/ ({trained_steps + done:,} steps trained in total)")
    return model, tok


def tokenizer_for(resume_from, default):
    """The checkpoint's tokenizer when resuming, otherwise `default`."""
    if resume_from:
        return CharTokenizer.load(os.path.join(resume_from, "tokenizer.json"))
    return default

