# LamLLM

A small GPT-style language model written from scratch in PyTorch. It learns from text downloaded from the web and answers questions.

## Layout

```
lamllm/model.py           transformer (attention, MLP, blocks, generate)
lamllm/tokenizer.py       character-level tokenizer
lamllm/trainer.py         training loop (continues an existing model, takes in new data as it arrives)
websearch/sources.py      downloaders: Wikipedia, Simple English Wikipedia, Stack Exchange, Project Gutenberg
websearch/multi_train.py  download from several sources and train at the same time
ask.py                    ask the trained model questions
```

## Usage

```bash
source ~/.venv/bin/activate
pip install -r requirements.txt

python websearch/multi_train.py                                      # all sources, 20 MB in total, training while downloading
python websearch/multi_train.py --sources wikipedia,stackexchange --mb 10
python websearch/multi_train.py --sources wikipedia --topics "space,biology"
python ask.py                                                        # interactive Q&A
python ask.py --question "What is photosynthesis?"
```

### Sources (`multi_train.py --sources`)

| Name | What it is |
|---|---|
| `wikipedia` | Random English Wikipedia articles (or `--topics` first) |
| `simplewiki` | Simple English Wikipedia: shorter, simpler sentences |
| `stackexchange` | Real questions with their top-voted answers. Pick sites with `--se-sites history,biology` |
| `gutenberg` | Popular public-domain books (from Gutenberg's mirror) |

### How training works

- **Continues by default.** Training keeps improving the model in `checkpoints/wiki/`. Each run adds to it; nothing is lost. Use `--fresh` to start a new model. A continued model keeps its original size and character set, so the size flags (`--n-layer`, `--n-embd`, ...) only apply to new models.
- **Downloads and training run together** in `multi_train.py`. Each source downloads in its own background thread, and new documents join the training data every 50 steps. The `data` column in the log shows how much the model is learning from.
- **Downloads are cached** in `data/sources/`. Later runs reuse them and only fetch more until each source has its share of `--mb`.
- **Q&A format.** Wikipedia articles produce "Q: What is X? / A: X is ..." pairs, and Stack Exchange produces "Q: question / A: top answer" pairs, so the model learns to respond to questions.
- The model is saved every `--eval-every` steps, and Ctrl+C stops training and saves.
- Defaults for a new model: 6 layers, 192-dim, 256-char context (~2.7M params). 3000 steps takes about 75 minutes on CPU; the log prints time remaining.
- Use `--contact you@example.com` to add contact details to the User-Agent, as Wikimedia asks.

### What's stored where

- `checkpoints/wiki/lamllm.pt` + `tokenizer.json` are the model. Everything it has learned is in these two files; they're all `ask.py` needs to answer. Keep them together.
- `data/sources/` is the downloaded text. It's only used for training and for `ask.py`'s passage search. Deleting it doesn't change the model; the next training run just downloads fresh text.

### Asking questions

`ask.py` prints two answers: the best-matching passage from the downloaded Wikipedia and Stack Exchange text (keyword search, reliable), and LamLLM's own generated answer (a model this small writes plausible-sounding but often wrong text). `--model-only` shows only the model's answer.

### GPU

`multi_train.py --device` picks the hardware; the first log line names it (e.g. `training on CPU: Intel(R) Core(TM) Ultra 7 355 (8 cores)`).

| `--device` | Uses |
|---|---|
| `auto` (default) | A GPU if PyTorch can see one, otherwise the CPU |
| `gpu` | Any GPU; stops with an error if none is usable |
| `cpu` | The CPU, even if a GPU is available |
| `cuda` / `xpu` / `mps` | Specifically NVIDIA / Intel / Apple Silicon |
