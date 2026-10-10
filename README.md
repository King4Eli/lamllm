# LamLLM

A small GPT-style language model written from scratch in PyTorch, trained on Wikipedia to answer questions.

## Layout

```
lamllm/model.py          transformer (attention, MLP, blocks, generate)
lamllm/tokenizer.py      character-level tokenizer
lamllm/trainer.py        training loop
websearch/wiki_train.py  download Wikipedia articles and train the model
ask.py                   ask the trained model questions
```

## Usage

```bash
source ~/.venv/bin/activate
pip install -r requirements.txt

python websearch/wiki_train.py --mb 10                                  # download ~10 MB of random articles, then train
python websearch/wiki_train.py --mb 10 --topics "space,biology,history" # fetch articles on these topics first
python ask.py                                                           # interactive Q&A
python ask.py --question "What is photosynthesis?"
```

- Articles come from the official Wikipedia API and are cached in `data/wiki/articles.jsonl`. Re-running reuses the cache and only downloads more if `--mb` is larger. Use `--download-only` to skip training, and `--contact you@example.com` to add contact details to the User-Agent, as Wikimedia asks.
- Training text is the articles plus Q/A pairs built from each article's opening sentences ("Q: What is X?" / "A: X is ..."), so the model learns to answer in that format.
- Each run trains from scratch and saves to `checkpoints/wiki/` every `--eval-every` steps. Ctrl+C stops training and saves.
- Defaults: 6 layers, 192-dim, 256-char context (~2.7M params), 3000 steps, about 75 minutes on CPU. The log prints time remaining. Model size flags: `--n-layer`, `--n-head`, `--n-embd`, `--block-size`.
- `ask.py` prints two answers: the best-matching Wikipedia passage (keyword search, reliable) and LamLLM's own generated answer (a model this small writes plausible-sounding but often wrong text). `--model-only` shows only the model's answer.
- GPU: an NVIDIA GPU (CUDA) or Intel GPU (XPU) is used automatically when PyTorch can see it; the first log line shows `device=`.
