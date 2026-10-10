"""Training loop used by websearch/multi_train.py.

Press Ctrl+C during training (or `kill <pid>` a background run) to stop early; the
model is saved before exiting.
"""

import contextlib
import math
import os
import platform
import signal
import threading
import time
from dataclasses import asdict

import torch

from .model import LamConfig, LamLLM
from .tokenizer import load_tokenizer


DEVICE_CHOICES = ["auto", "gpu", "cpu", "cuda", "xpu", "mps"]


def _available_gpus():
    gpus = []
    if torch.cuda.is_available():
        gpus.append("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        gpus.append("xpu")  # Intel GPU
    if torch.backends.mps.is_available():
        gpus.append("mps")  # Apple Silicon
    return gpus


def get_device(choice="auto"):
    """Pick the device to run on.

    auto: the first available GPU, else the CPU. gpu: the first available GPU, or
    an error if there is none. cpu/cuda/xpu/mps: exactly that, or an error.
    """
    gpus = _available_gpus()
    if choice == "auto":
        return gpus[0] if gpus else "cpu"
    if choice == "cpu":
        return "cpu"
    if choice == "gpu" and gpus:
        return gpus[0]
    if choice in gpus:
        return choice
    raise SystemExit(
        f"--device {choice}: no usable GPU found. PyTorch {torch.__version__} sees "
        f"{', '.join(gpus) or 'no GPU'}. For an Intel GPU, install Intel's GPU compute drivers "
        f"(libze-intel-gpu1, libze1, intel-opencl-icd); for NVIDIA, a CUDA build of PyTorch "
        f"and the NVIDIA driver. Use --device cpu or --device auto to train on the CPU."
    )


def device_name(device):
    """Human-readable name of the hardware behind `device`, e.g. 'GPU: NVIDIA GeForce RTX 4070'."""
    if device == "cuda":
        return f"GPU: {torch.cuda.get_device_name(0)}"
    if device == "xpu":
        return f"GPU: {torch.xpu.get_device_name(0)}"
    if device == "mps":
        return "GPU: Apple Silicon"
    name = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            name = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return f"CPU: {name} ({os.cpu_count()} cores)"




def save_checkpoint(model, tok, out, opt=None, trained_steps=0, trained_tokens=0):
    os.makedirs(out, exist_ok=True)
    ckpt = {"config": asdict(model.cfg), "model": model.state_dict(),
            "trained_steps": trained_steps, "trained_tokens": trained_tokens}
    if opt is not None:
        ckpt["optimizer"] = opt.state_dict()
    tmp = os.path.join(out, "lamllm.pt.tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, os.path.join(out, "lamllm.pt"))  # a crash mid-save can't corrupt the old checkpoint
    tok.save(os.path.join(out, "tokenizer.json"))


def checkpoint_exists(ckpt_dir):
    return os.path.exists(os.path.join(ckpt_dir, "lamllm.pt"))


def load_checkpoint(ckpt_dir, device="cpu"):
    """Returns (model, tokenizer, checkpoint dict)."""
    # Loaded to the CPU first, so the optimizer state stored alongside the model doesn't take GPU memory.
    ckpt = torch.load(os.path.join(ckpt_dir, "lamllm.pt"), map_location="cpu")
    tok = load_tokenizer(os.path.join(ckpt_dir, "tokenizer.json"))
    model = LamLLM(LamConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model, tok, ckpt


class _Buffer:
    """A 1-D token tensor that grows by doubling, so appending doesn't copy everything each time."""

    def __init__(self, dtype):
        self.data = torch.empty(1 << 16, dtype=dtype)
        self.n = 0

    def extend(self, ids):
        end = self.n + len(ids)
        if end > len(self.data):
            grown = torch.empty(max(end, 2 * len(self.data)), dtype=self.data.dtype)
            grown[:self.n] = self.data[:self.n]
            self.data = grown
        self.data[self.n:end] = ids
        self.n = end

    def __len__(self):
        return self.n


class Corpus:
    """Training data from several sources that keeps growing while the model trains.

    Downloader threads call add(text, source); the training loop calls refresh() to
    tokenize new documents and fold them in. Every `val_every`-th document goes to the
    validation split. Each source is stored separately and every batch mixes them:
    a source is picked with probability proportional to size ** mix_power, so with the
    default 0.5 a source with 100x more text is sampled only 10x as often.

    The tokenizer can be set later (set_tokenizer), e.g. once enough text has arrived
    to train a BPE vocabulary on it.

    wants_more() tells downloaders when to pause: when the corpus holds `max_tokens`
    (the memory cap), or is `lookahead` tokens ahead of what training has consumed
    (fit() updates trained_tokens), so fast sources don't download far more than the
    model can read.
    """

    def __init__(self, tok=None, val_every=10, mix_power=0.5, max_tokens=1_500_000_000, lookahead=100_000_000):
        self.tok = tok
        self.val_every = val_every
        self.mix_power = mix_power
        self.max_tokens = max_tokens
        self.lookahead = lookahead
        self.trained_tokens = 0
        self.sources = {}  # name -> {"train": _Buffer, "val": _Buffer}
        self.n_docs = 0
        self.chars = 0  # text added so far, tokenized or not
        self._pending = []  # (source, text)
        self._pending_chars = 0
        self._tokenized = [0, 0]  # chars, tokens: for estimating the size of untokenized text
        self._lock = threading.Lock()

    def set_tokenizer(self, tok):
        self.tok = tok

    def add(self, text, source="text"):
        with self._lock:
            self._pending.append((source, text))
            self.chars += len(text)
            self._pending_chars += len(text)

    def tokens_estimate(self):
        chars, tokens = self._tokenized
        chars_per_token = chars / tokens if tokens else 4.0
        return sum(len(s["train"]) + len(s["val"]) for s in self.sources.values()) + self._pending_chars / chars_per_token

    def wants_more(self):
        n = self.tokens_estimate()
        return n < self.max_tokens and n < self.trained_tokens + self.lookahead

    def pending_texts(self):
        with self._lock:
            return [t for _, t in self._pending]

    def refresh(self):
        """Tokenize pending documents into the splits. Returns how many were added."""
        if self.tok is None:
            return 0
        with self._lock:
            pending, self._pending = self._pending, []
        if not pending:
            return 0
        dtype = torch.int16 if self.tok.vocab_size <= 32768 else torch.int32
        # A few MB at a time: the tokenizer's intermediate results are much bigger than the ids.
        i = 0
        while i < len(pending):
            chunk, chars = [], 0
            while i < len(pending) and (not chunk or chars < 4_000_000):
                chunk.append(pending[i])
                chars += len(pending[i][1])
                i += 1
            ids = self.tok.encode_batch([text + "\n\n" for _, text in chunk])
            for (source, _), doc_ids in zip(chunk, ids):
                self.n_docs += 1
                split = "val" if self.n_docs % self.val_every == 0 else "train"
                if source not in self.sources:
                    self.sources[source] = {"train": _Buffer(dtype), "val": _Buffer(dtype)}
                self.sources[source][split].extend(torch.tensor(doc_ids, dtype=dtype))
            with self._lock:
                self._pending_chars -= chars
            self._tokenized[0] += chars
            self._tokenized[1] += sum(len(d) for d in ids)
        return len(pending)

    def size(self, split):
        return sum(len(s[split]) for s in self.sources.values())

    def eligible(self, split, block_size):
        """Sources with enough tokens in `split` to cut a training example from."""
        return [name for name, s in self.sources.items() if len(s[split]) > block_size + 1]

    def ready(self, block_size):
        return bool(self.eligible("train", block_size)) and bool(self.eligible("val", block_size))

    def get_batch(self, split, block_size, batch_size, device, source=None):
        names = [source] if source else self.eligible(split, block_size)
        bufs = [self.sources[n][split] for n in names]
        weights = torch.tensor([len(b) ** self.mix_power for b in bufs], dtype=torch.float)
        picks = torch.multinomial(weights, batch_size, replacement=True).tolist()
        rows = []
        for p in picks:
            buf = bufs[p]
            i = torch.randint(len(buf) - block_size - 1, (1,)).item()
            rows.append(buf.data[i:i + block_size + 1])
        batch = torch.stack(rows).long()
        if device == "cuda":
            batch = batch.pin_memory()
        batch = batch.to(device, non_blocking=True)
        return batch[:, :-1].contiguous(), batch[:, 1:].contiguous()


def autocast_for(device):
    """bf16 mixed precision where the GPU supports it: about 2x faster and half the activation memory."""
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if device == "xpu":
        return torch.autocast("xpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def estimate_loss(model, corpus, block_size, batch_size, device, amp, iters=20):
    """Mean loss on mixed train and val batches, plus val loss per source."""
    model.eval()

    def mean_loss(split, n, source=None):
        total = 0.0
        for _ in range(n):
            x, y = corpus.get_batch(split, block_size, batch_size, device, source)
            with amp:
                _, loss = model(x, y)
            total += loss.item()
        return total / n

    out = {"train": mean_loss("train", iters), "val": mean_loss("val", iters)}
    per_source = {s: mean_loss("val", max(iters // 4, 1), s) for s in corpus.eligible("val", block_size)}
    model.train()
    return out, per_source


def fit(corpus, out, steps=None, hours=None, batch_size=8, grad_accum=4, block_size=512, n_layer=8, n_head=8,
        n_embd=512, dropout=0.1, lr=6e-4, warmup=200, eval_every=250, refresh_every=50, seed=1337,
        resume_from=None, device=None):
    """Train on `corpus`, pulling in newly added documents every `refresh_every` steps.

    Stops after `steps` optimizer steps or `hours` of training, whichever comes first
    (3000 steps if neither is given). Each step sees batch_size * grad_accum sequences.
    The learning rate warms up over `warmup` steps, then follows a cosine down to lr/10
    over the run, measured by whichever limit is closer.

    With `resume_from` (a checkpoint dir), training continues from that model; its
    architecture is kept and the size arguments are ignored. The corpus must use the
    checkpoint's tokenizer (see tokenizer_for).
    """
    if steps is None and hours is None:
        steps = 3000
    torch.manual_seed(seed)
    device = device or get_device()
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = autocast_for(device)
    tok = corpus.tok
    corpus.refresh()

    trained_steps = trained_tokens = 0
    if resume_from:
        model, ckpt_tok, ckpt = load_checkpoint(resume_from, device)
        assert ckpt_tok.vocab_size == tok.vocab_size, "corpus must use the checkpoint's tokenizer"
        block_size = model.cfg.block_size
        trained_steps = ckpt.get("trained_steps", 0)
        trained_tokens = ckpt.get("trained_tokens", 0)
    else:
        cfg = LamConfig(
            vocab_size=tok.vocab_size, block_size=block_size,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=dropout,
        )
        model = LamLLM(cfg).to(device)
    # Weight decay on matrices only, not on biases and LayerNorm gains.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95), fused=device == "cuda")
    if resume_from and "optimizer" in ckpt:
        try:
            opt.load_state_dict(ckpt["optimizer"])
        except ValueError:
            print("optimizer state from the checkpoint doesn't match; starting it fresh")

    tokens_per_step = batch_size * grad_accum * block_size

    def data_size():
        return f"{corpus.size('train') + corpus.size('val'):,} tokens"

    if resume_from:
        so_far = f"{trained_steps:,} steps trained so far" if trained_steps else "earlier steps not recorded"
        print(f"continuing from {resume_from} ({so_far})")
    else:
        print("starting a new model")
    limits = [f"{steps:,} steps" if steps else None, f"{hours:g} hours" if hours else None]
    print(f"device={device} vocab={tok.vocab_size} params={model.num_params():,} context={block_size} "
          f"tokens/step={tokens_per_step:,} run for {' or '.join(x for x in limits if x)}")

    t0 = time.time()

    def progress(step):
        p = step / steps if steps else 0.0
        if hours:
            p = max(p, (time.time() - t0) / (hours * 3600))
        return min(p, 1.0)

    def lr_at(step, p):
        min_lr = lr / 10
        cosine = min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))
        return cosine * min(1.0, (step + 1) / warmup)

    def timing(p):
        elapsed = time.time() - t0
        if p <= 0 or elapsed < 60:
            return "estimating time left..."
        return f"{elapsed / 60:.1f}m elapsed, ~{elapsed / p * (1 - p) / 60:.0f}m left"

    def stop(signum, frame):
        raise KeyboardInterrupt

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, stop)  # `kill <pid>` saves too, like Ctrl+C

    step = 0
    try:
        while True:
            p = progress(step)
            if step and step % refresh_every == 0:
                corpus.refresh()
            if step % eval_every == 0 or p >= 1:
                losses, per_source = estimate_loss(model, corpus, block_size, batch_size, device, amp)
                by_source = " ".join(f"{s} {v:.2f}" for s, v in sorted(per_source.items()))
                print(f"\rstep {step:6d} | train {losses['train']:.4f} | val {losses['val']:.4f} ({by_source}) "
                      f"| data {data_size()} | {timing(p)}".ljust(110), flush=True)
                if step:
                    save_checkpoint(model, tok, out, opt, trained_steps + step,
                                    trained_tokens + step * tokens_per_step)
            if p >= 1:
                break
            for group in opt.param_groups:
                group["lr"] = lr_at(step, p)
            t_step = time.time()
            for _ in range(grad_accum):
                x, y = corpus.get_batch("train", block_size, batch_size, device)
                with amp:
                    _, loss = model(x, y)
                (loss / grad_accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            corpus.trained_tokens = step * tokens_per_step
            if step % 10 == 0:
                # Progress line, overwritten in place until the next eval line.
                tps = tokens_per_step / (time.time() - t_step)
                line = f"  step {step} | loss {loss.item():.4f} | {tps:,.0f} tok/s | {timing(progress(step))}"
                print("\r" + line.ljust(110), end="", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted, saving current model...")

    total = trained_steps + step
    save_checkpoint(model, tok, out, opt, total, trained_tokens + step * tokens_per_step)
    print(f"saved checkpoint to {out}/ ({total:,} steps trained in total)")
    return model, tok


def tokenizer_for(resume_from, default):
    """The checkpoint's tokenizer when resuming, otherwise `default`."""
    if resume_from:
        return load_tokenizer(os.path.join(resume_from, "tokenizer.json"))
    return default
