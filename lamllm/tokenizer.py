"""Tokenizers.

CharTokenizer  one token per character. Small vocabulary, but the model has to spend
               its capacity learning to spell.
BPETokenizer   byte-level BPE trained on the downloaded text: common words and word
               pieces become single tokens, so the model sees ~4x more text per context.

Both are saved as tokenizer.json next to the model; load_tokenizer() picks the right one.
"""

import json


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @classmethod
    def ascii(cls):
        """Fixed vocabulary: newline plus printable ASCII. Used when text arrives during training."""
        return cls(["\n"] + [chr(i) for i in range(32, 127)])

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        # Characters never seen in training are dropped.
        return [self.stoi[c] for c in text if c in self.stoi]

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.chars, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))


class BPETokenizer:
    def __init__(self, tok):
        self.tok = tok

    @classmethod
    def train(cls, texts, vocab_size=8192):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
        )
        tok.train_from_iterator(texts, trainer)
        return cls(tok)

    @property
    def vocab_size(self):
        return self.tok.get_vocab_size()

    def encode(self, text):
        return self.tok.encode(text).ids

    def encode_batch(self, texts):
        # encode_batch_fast skips character offsets, which take far more memory than the ids.
        encode = getattr(self.tok, "encode_batch_fast", self.tok.encode_batch)
        return [e.ids for e in encode(texts)]

    def decode(self, ids):
        return self.tok.decode(ids)

    def save(self, path):
        self.tok.save(path)

    @classmethod
    def load(cls, path):
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(path))


def load_tokenizer(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return CharTokenizer(data) if isinstance(data, list) else BPETokenizer.load(path)
