# LamLLM

A small GPT-style language model written from scratch in PyTorch. It learns from text downloaded from the web and answers questions.

## Layout

```
lamllm/model.py           transformer (attention, MLP, blocks, generate)
lamllm/tokenizer.py       BPE (word-piece) and character-level tokenizers
lamllm/trainer.py         training loop (continues an existing model, takes in new data as it arrives)
websearch/sources.py      downloaders: Wikipedia, Stack Exchange, Gutenberg, Hugging Face datasets, Kaggle
websearch/multi_train.py  download from several sources and train at the same time
ask.py                    ask the trained model questions
```

## Usage

```bash
source ~/.venv/bin/activate
pip install -r requirements.txt

python websearch/multi_train.py --fresh --hours 8                    # new ~30M-param model, all sources, 8 hours
python websearch/multi_train.py --hours 4                            # continue it for 4 more hours
python websearch/multi_train.py --fresh --size large --hours 24      # ~92M params, for very long runs
python websearch/multi_train.py --sources wikipedia,stackexchange --mb 50
python websearch/multi_train.py --sources wikipedia --topics "space,biology"
python websearch/multi_train.py --sources fineweb-edu,squad,hf:roneneldan/TinyStories,kaggle:owner/dataset

# Long run in the background; `kill <pid>` stops it and saves the model
nohup python websearch/multi_train.py --fresh --hours 8 > train.log 2>&1 &
python ask.py                                                        # interactive Q&A
python ask.py --question "What is photosynthesis?"
```

### Sources (`multi_train.py --sources`)

The default is `fineweb-edu,wikipedia-full,cosmopedia,simplewiki,stackexchange,gutenberg,squad,trivia-qa,dolly`.

| Name | What it is |
|---|---|
| `wikipedia` | Random English Wikipedia articles via the API, slow (or `--topics` first) |
| `simplewiki` | Simple English Wikipedia: shorter, simpler sentences |
| `stackexchange` | Real questions with their top-voted answers. Pick sites with `--se-sites history,biology` |
| `gutenberg` | Popular public-domain books (from Gutenberg's mirror) |
| **Hugging Face** | Streamed from the Hub a few MB at a time; fast |
| `fineweb-edu` | Educational web pages, filtered for quality (10B-token sample) |
| `fineweb` | General web pages (10B-token sample) |
| `wikipedia-full` | All 6.4M English Wikipedia articles (Nov 2023 dump) |
| `cosmopedia` | Synthetic textbooks, stories and lessons written by an LLM |
| `tinystories` | Short simple stories; helps small models write fluently |
| `openwebtext` | Web pages linked from Reddit (GPT-2-style data) |
| `arxiv` | Titles and abstracts of 2M scientific papers |
| `pubmed` | Biomedical research articles with abstracts |
| `squad` | Wikipedia paragraphs with questions and answers about them |
| `trivia-qa` | 138k trivia questions with answers |
| `dolly` | 15k human-written instructions and responses |
| `gsm8k` | Grade-school maths word problems with worked answers |
| `hf:owner/name` | Any Hugging Face dataset; its longest text columns are used |
| **Kaggle** | |
| `kaggle:owner/name` | Any Kaggle dataset: text is read from its .txt/.md/.csv/.tsv/.json/.jsonl/.parquet files. Needs an API key: kaggle.com > Settings > API > Create New Token, saved as `~/.kaggle/kaggle.json` |

Stack Exchange allows about 300 requests a day without a key; the source then waits until the quota resets (the log says when). A free key from https://stackapps.com/apps/oauth/register raises that to 10,000: `--se-key KEY` or `STACKEXCHANGE_KEY=KEY`. For Hugging Face, setting `HF_TOKEN` gives higher rate limits and access to gated datasets.

### How training works

- **Continues by default.** Training keeps improving the model in `checkpoints/wiki/`. Each run adds to it; nothing is lost. Use `--fresh` to start a new model. A continued model keeps its original size and tokenizer, so the size flags (`--size`, `--n-layer`, `--n-embd`, ...) only apply to new models.
- **How long:** `--hours N` trains for N hours, `--steps N` for N optimizer steps (default 3000 when neither is given). The learning rate warms up, then decays over the run.
- **Downloads and training run together** in `multi_train.py`. Each source downloads in its own background thread, and new documents join the training data every 50 steps. With `--mb 0` (the default) downloading continues for the whole run; `--mb N` stops after N MB in total.
- **Sources are mixed in every batch.** A source with 100x more text is sampled only 10x as often, so one big source can't drown out the others. The eval line shows validation loss per source; if one starts rising while training loss falls, the model is memorising that source.
- **Downloads keep pace with training.** Fast sources pause (`paused` in the status) once they're about 100M tokens ahead of what the model has read, or when the text in memory reaches `--ram-gb` (default 3 GB, ~1.5B tokens).
- **Downloads are cached** in `data/sources/`. Each run loads the most recent part of every cache that fits in memory, and each source carries on after the last document it cached. A source that hits an error waits and retries rather than stopping.
- **Q&A format.** Wikipedia articles produce "Q: What is X? / A: X is ..." pairs, and Stack Exchange produces "Q: question / A: top answer" pairs, so the model learns to respond to questions.
- The model is saved every `--eval-every` steps, and Ctrl+C stops training and saves.
- **Model sizes** (`--size`, new models only). Batch sizes are set to fit a 4 GB GPU; speeds are on an RTX 3050 Ti Laptop.

  | `--size` | Shape | Params | Speed |
  |---|---|---|---|
  | `small` | 6 layers, 192-dim, 256 context | ~3M (char) | the original LamLLM |
  | `medium` (default) | 8 layers, 512-dim, 512-token context | ~30M | ~44k tokens/s, ~160M tokens/hour |
  | `large` | 12 layers, 768-dim, 512-token context | ~92M | ~15k tokens/s |

- **Tokenizer.** A new model gets an 8192-token BPE tokenizer trained on the first `--min-start-mb` (2 MB) of downloaded text plus the cache, about 3.7 characters per token. `--tokenizer char` gives the old one-character-per-token tokenizer.
- The GPU trains in bf16 mixed precision.
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
