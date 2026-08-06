"""Faithful re-implementation of `tf.keras.preprocessing.text.Tokenizer`.

The original notebook used Keras' tokenizer with `num_words=10000` and
`oov_token='<UNK>'`. This module reproduces its exact behaviour (filters,
lowercasing, frequency-ranked index, OOV handling, post-padding/truncating)
so that the PyTorch re-implementation is comparable to the TensorFlow model
and so that the browser can tokenise text identically at inference time.
"""

from collections import OrderedDict

KERAS_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
DEFAULT_MAXLEN = 50


def text_to_word_sequence(text, filters=KERAS_FILTERS, lower=True, split=" "):
    if lower:
        text = text.lower()
    translate_map = str.maketrans(filters, split * len(filters))
    text = text.translate(translate_map)
    return [w for w in text.split(split) if w]


class Tokenizer:
    def __init__(self, num_words=None, oov_token=None):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_counts = OrderedDict()
        self.word_index = {}
        self.index_word = {}

    def fit_on_texts(self, texts):
        for text in texts:
            for w in text_to_word_sequence(text):
                self.word_counts[w] = self.word_counts.get(w, 0) + 1

        # Keras sorts by descending count, ties broken by insertion order.
        wcounts = sorted(self.word_counts.items(), key=lambda kv: -kv[1])
        sorted_voc = []
        if self.oov_token is not None:
            sorted_voc.append(self.oov_token)
        sorted_voc.extend(w for w, _ in wcounts)

        self.word_index = {w: i + 1 for i, w in enumerate(sorted_voc)}
        self.index_word = {i: w for w, i in self.word_index.items()}
        return self

    def texts_to_sequences(self, texts):
        oov_index = self.word_index.get(self.oov_token) if self.oov_token else None
        num_words = self.num_words
        out = []
        for text in texts:
            seq = []
            for w in text_to_word_sequence(text):
                i = self.word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        if oov_index is not None:
                            seq.append(oov_index)
                    else:
                        seq.append(i)
                elif oov_index is not None:
                    seq.append(oov_index)
            out.append(seq)
        return out

    def vocab(self, limit=None):
        """Index -> word for the indices actually reachable by the model."""
        limit = limit or self.num_words or (len(self.word_index) + 1)
        return [self.index_word.get(i, "") for i in range(limit)]


def pad_sequences(sequences, maxlen=DEFAULT_MAXLEN):
    """padding='post', truncating='post', value=0 — as used in the notebook."""
    out = []
    for s in sequences:
        s = s[:maxlen]
        out.append(s + [0] * (maxlen - len(s)))
    return out
