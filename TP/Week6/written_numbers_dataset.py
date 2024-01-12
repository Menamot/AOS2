from collections.abc import Iterable
from timeit import default_timer as timer
from typing import List

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
import torch
from torch import Tensor, nn
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import build_vocab_from_iterator
digit = [
    "",
    "UN",
    "DEUX",
    "TROIS",
    "QUATRE",
    "CINQ",
    "SIX",
    "SEPT",
    "HUIT",
    "NEUF",
]

MILLIONS = ["", "MILLE", "MILLION", "MILLIARD"]


def millions(n):
    if n <= 3:
        return MILLIONS[n]
    if n == 4:
        return millions(n - 3) + " MILLIARD"
    if n >= 5:
        return millions(n - 3) + " DE MILLIARD"


r10_19 = [
    "DIX",
    "ONZE",
    "DOUZE",
    "TREIZE",
    "QUATORZE",
    "QUINZE",
    "SEIZE",
    "DIX SEPT",
    "DIX HUIT",
    "DIX NEUF",
]

dizaine = [
    "",
    "",
    "VINGT",
    "TRENTE",
    "QUARANTE",
    "CINQUANTE",
    "SOIXANTE",
    "SOIXANTE",
    "QUATRE VINGT",
    "QUATRE VINGT",
]


def leq99(s, n3):
    d, u = [int(digit) for digit in s]

    # MILLE au lieu de UN MILLE
    if u == 1 and d == 0 and (n3 % 3) == 1:
        return [millions(n3)]

    # Le ET pour VINGT *ET* UN...
    if (u == 1) and 2 <= d <= 7:
        et = "ET"
    else:
        et = ""

    # ONZE, DOUZE,...
    if d == 1 or d == 7 or d == 9:
        rest = r10_19
    else:
        rest = digit

    return [dizaine[d], et, rest[u], millions(n3)]


def leq999(s, n3):
    s = s.zfill(3)
    c = int(s[0])

    if s == "000":
        return []

    if c == 0:
        return leq99(s[1:], n3)

    if c == 1:
        if s[:1] == "00":
            return ["CENT"]
        else:
            return ["CENT"] + leq99(s[1:], n3)

    if 2 <= c <= 9:
        if s[1:] == "00":
            return [digit[c], "CENTS", millions(n3)]
        else:
            return [digit[c], "CENT"] + leq99(s[1:], n3)


def integer_to_token_list(n):
    def written_number0(s, n3):
        if len(s) >= 4:
            return written_number0(s[:-3], n3 + 1) + leq999(s[-3:], n3)
        else:
            return leq999(s, n3)

    w = written_number0(str(n), 0)
    return " ".join(w).split()


class NumberDataset(Dataset):
    import numpy as np
    def __init__(
            self,
            seed=None,
            n_numbers=50000,
    ):
        super().__init__()
        self.seed = seed
        self.n_numbers = n_numbers

        self.vocab_src = list("0123456789")
        self.vocab_tgt = [
            "ET",
            "DE",
            "UN",
            "DEUX",
            "TROIS",
            "QUATRE",
            "CINQ",
            "SIX",
            "SEPT",
            "HUIT",
            "NEUF",
            "DIX",
            "ONZE",
            "DOUZE",
            "TREIZE",
            "QUATORZE",
            "QUINZE",
            "SEIZE",
            "VINGT",
            "TRENTE",
            "QUARANTE",
            "CINQUANTE",
            "SOIXANTE",
            "CENT",
            "CENTS",
            "MILLE",
            "MILLION",
            "MILLIARD",
        ]

        rs = np.random.RandomState(self.seed)

        N = 13
        exponents = rs.uniform(low=2, high=N, size=self.n_numbers)
        self.numbers = (10 ** exponents).astype(np.longlong)

    def __len__(self):
        return self.n_numbers

    def __getitem__(self, i):
        assert(i < self.n_numbers)
        number = self.numbers[i]
        return list(str(number)), integer_to_token_list(number)
