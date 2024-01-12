# %% [markdown]
# # The transformer architecture from scratch

# %%
import math
from collections.abc import Iterable
from timeit import default_timer as timer
from typing import List

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import build_vocab_from_iterator

# %% [markdown]
# ## Toy dataset

# %%
def translate_deterministic(input_sequence):
    target_sequence = []
    for i, elt in enumerate(input_sequence):
        try:
            offset = int(elt)
        except ValueError:  # It is a letter
            target_sequence.append(elt)
        else:               # Special token, do the lookup
            if i + offset < 0 or i + offset > len(input_sequence) - 1:
                pass
            else:
                k = min(max(0, i + offset), len(input_sequence) - 1)
                target_sequence.append(input_sequence[k])

    return target_sequence


class GotoDataset(Dataset):
    def __init__(
        self,
        seed=None,
        n_sequences=100,
        min_length=4,
        max_length=20,
        n_letters=3,
        offsets=[4, 5, 6],
    ):
        super().__init__()
        full_vocab = "abcdefghijklmnopqrstuvwxyz"
        full_vocab = list(full_vocab.upper()) + list(full_vocab)
        assert(n_letters <= len(full_vocab))

        self.vocab = np.array(
            [s + str(d) for s in ["+", "-"] for d in offsets] + full_vocab[:n_letters]
        )
        self.n_tokens = len(self.vocab)
        self.min_length = min_length
        self.max_length = max_length
        self.seed = seed
        self.n_sequences = n_sequences

        # Dataset generation
        rs = np.random.RandomState(self.seed)
        seq_lengths = rs.randint(
            self.min_length, self.max_length, size=self.n_sequences
        )
        self.input_sequences = [
            list(self.vocab[rs.randint(self.n_tokens, size=seq_length)])
            for seq_length in seq_lengths
        ]

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, i):
        input_sequence = self.input_sequences[i]
        target_sequence = translate_deterministic(input_sequence)
        return input_sequence, target_sequence

# %% [markdown]
# ## Vocabulary

# %%
dataset = GotoDataset()
special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
vocab = build_vocab_from_iterator([dataset.vocab], specials=special_tokens)
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = vocab.lookup_indices(special_tokens)

# %% [markdown]
# ## Collate function

# %%
def collate_fn(batch: List):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:

        # Numericalize list of tokens using `vocab`.
        #
        # - Don't forget to add beginning of sequence and end of sequence tokens
        #   before numericalizing.
        #
        # - Use `torch.LongTensor` instead of `torch.Tensor` because the next
        #   step is an embedding that needs integers for its lookup table.
        # <answer>
        src_tensor = torch.LongTensor(vocab(["<bos>"] + src_sample + ["<eos>"]))
        tgt_tensor = torch.LongTensor(vocab(["<bos>"] + tgt_sample + ["<eos>"]))
        # </answer>

        # Append numericalized sequence to `src_batch` and `tgt_batch`
        src_batch.append(src_tensor)
        tgt_batch.append(tgt_tensor)

    # Turn `src_batch` and `tgt_batch` that are lists of 1-dimensional
    # tensors of varying sizes into tensors with same size with
    # padding. Use `pad_sequence` with padding value to do so.
    #
    # Important notice: by default resulting tensors are of size
    # `max_seq_length` * `batch_size`; the mini-batch size is on the
    # *second dimension*.
    # <answer>
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    # </answer>

    return src_batch, tgt_batch

# %% [markdown]
# ## Hyperparameters of transformer model

# %%
torch.manual_seed(0)

# Size of source and target vocabulary
VOCAB_SIZE = len(vocab)

# Number of sequences generated for the training set
N_SEQUENCES = 7000

# Number of epochs
NUM_EPOCHS = 20

# Size of embeddings
EMB_SIZE = 64

# Number of heads for the multihead attention
NHEAD = 1

# Size of hidden layer of FFN
FFN_HID_DIM = 128

# Size of mini-batches
BATCH_SIZE = 256

# Number of stacked encoder modules
NUM_ENCODER_LAYERS = 1

# Number of stacked decoder modules
NUM_DECODER_LAYERS = 1

# %% [markdown]
# ## Transformer encoder

# %%
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()

        # Define Tk/2pi for even k between 0 and `emb_size`. Use
        # `torch.arange`.
        # <answer>
        Tk_over_2pi = 10000 ** (torch.arange(0, emb_size, 2) / emb_size)
        # </answer>

        # Define `t = 0, 1,..., maxlen-1`. Use `torch.arange`.
        # <answer>
        t = torch.arange(maxlen)
        # </answer>

        # Outer product between `t` and `1/Tk_over_2pi` to have a
        # matrix of size `maxlen` * `emb_size // 2`. Use
        # `torch.outer`.
        # <answer>
        outer = torch.outer(t, 1 / Tk_over_2pi)
        # </answer>

        pos_embedding = torch.empty((maxlen, emb_size))

        # Fill `pos_embedding` with either sine or cosine of `outer`.
        # <answer>
        pos_embedding[:, 0::2] = torch.sin(outer)
        pos_embedding[:, 1::2] = torch.cos(outer)
        # </answer>

        # Add fake mini-batch dimension to be able to use broadcasting
        # in `forward` method.
        pos_embedding = pos_embedding.unsqueeze(1)

        self.dropout = nn.Dropout(dropout)

        # Save `pos_embedding` when serializing the model even if it is not a
        # set of parameters
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        # `token_embedding` is of size `seq_length` * `batch_size` *
        # `embedding_size`. Use broadcasting to add the positional embedding
        # that is of size `seq_length` * 1 * `embedding_size`.
        # <answer>
        seq_length = token_embedding.size(0)
        positional_encoding = token_embedding + self.pos_embedding[:seq_length, :]
        # </answer>

        return self.dropout(positional_encoding)

# %%
class TransformerEncoder(nn.Module):
    def __init__(
            self,
            p=None,                  # Embedding size of input tokens
            d_ff=None,               # Size of hidden layer in MLP
    ):
        super().__init__()

        # Size of embedding. Here sizes of embedding, keys, queries
        # and values are the same.
        self.p = p
        d_q = d_v = d_k = p

        # Size of hidden layer in MLP
        self.d_ff = d_ff

        # Compute query, key and value from input
        self.enc_Q = nn.Linear(p, d_q)
        self.enc_K = nn.Linear(p, d_k)
        self.enc_V = nn.Linear(p, d_v)

        # Linear transform just before first residual mapping
        self.enc_W0 = nn.Linear(d_v, p)

        # Layer normalization after first residual mapping
        self.enc_ln1 = nn.LayerNorm(p)

        # Position-wise MLP
        self.enc_W1 = nn.Linear(p, d_ff)
        self.enc_W2 = nn.Linear(d_ff, p)

        # Final layer normalization of second residual mapping
        self.enc_ln2 = nn.LayerNorm(p)

    def forward(self, X):
        # Forward propagation in encoder. Input tensor `X` is of size
        # `seq_length` * `batch_size` * `p`.

        # Query, key and value of the encoder. Use `enc_Q`, `enc_K`
        # and `enc_V`.
        # <answer>
        Q = self.enc_Q(X)
        K = self.enc_K(X)
        V = self.enc_V(X)
        # </answer>

        # Score attention from `Q` and `K`. We need to compute `QK^T` but both
        # `Q` and `K` are not just simple matrices but batch of matrices. Both
        # `Q` and `K` are in fact of size `seq_length` * `batch_size` *
        # `emb_size`. Two ways to compute the batched matrix product:
        #
        # - permute dimensions using `torch.permute` so that `batch_size` is the
        #   first dimension and use `torch.bmm` that will perform the batch
        #   matrix product with respect to the first dimension,
        # - use `torch.einsum` to specify the product.
        # S = torch.bmm(Q.permute([1, 0, 2]), K.permute([1, 2, 0])) / math.sqrt(self.p)
        # <answer>
        S = torch.einsum('sbe,Sbe->bsS', Q, K) / math.sqrt(self.p)
        # </answer>

        # Compute attention from `S` and `V`. You can use `F.softmax` with `dim`
        # argument. Since the mini-batch dimension is now the first one for `S`
        # we can use `torch.bmm` with `S` (after softmax). That is not the case
        # for `V` so we need to transpose it first. Don't forget to transpose
        # again after the product to have a matrix `seq_length` * `batch_size` *
        # `emb_size` compatible with `X` for the residual mapping.
        # <answer>
        A = F.softmax(S, dim=-1)
        # T = torch.bmm(A, V.transpose(0, 1)).transpose(0, 1)
        T = torch.einsum("bsS,Sbe->sbe", A, V)
        # </answer>

        # First residual mapping and layer normalization
        # <answer>
        U = self.enc_ln1(self.enc_W0(T) + X)
        # </answer>

        # FFN on each token
        # <answer>
        Z = self.enc_W2(F.relu(self.enc_W1(U)))
        # </answer>

        # Second residual mapping and layer normalization
        # <answer>
        Xp = self.enc_ln2(Z + U)
        # </answer>

        return Xp

# %% [markdown]
# ## Transformer decoder

# %%
class TransformerDecoder(nn.Module):
    def __init__(
            self,
            p=None,                  # Embedding size of input tokens
            d_ff=None,               # Size of hidden layer in MLP
    ):

        super().__init__()

        # Size of embedding. Here, sizes of embedding, keys, queries
        # and values are the same.
        self.p = p
        self.d_q = self.d_v = self.d_k = p

        # Size of hidden layer in MLP
        self.d_ff = d_ff

        # Compute query, key and value from input
        self.dec_Q1 = nn.Linear(p, self.d_q)
        self.dec_K1 = nn.Linear(p, self.d_k)
        self.dec_V1 = nn.Linear(p, self.d_v)

        # Linear transform just before first residual mapping
        self.dec_W0 = nn.Linear(self.d_v, p)

        # Layer normalization after first residual mapping
        self.dec_ln1 = nn.LayerNorm(p)

        # Key-value cross-attention
        self.dec_Q2 = nn.Linear(p, self.d_k)
        self.dec_K2 = nn.Linear(p, self.d_k)
        self.dec_V2 = nn.Linear(p, self.d_v)

        # Linear transform just before first residual mapping
        self.dec_W1 = nn.Linear(self.d_v, p)

        # Layer normalization after second residual mapping
        self.dec_ln2 = nn.LayerNorm(p)

        # Position-wise MLP
        self.dec_W2 = nn.Linear(p, d_ff)
        self.dec_W3 = nn.Linear(d_ff, p)

        # Final layer normalization of second residual mapping
        self.dec_ln3 = nn.LayerNorm(p)

    def forward(self, Xp, Y):
        # Forward propagation in decoder. Input tensor `Xp` is of size
        # `seq_length_src` * `batch_size` * `p` and `Y` is of size
        # `seq_length_tgt` * `batch_size` * `p`.


        # Set number of tokens in target sequence `Y`. Needed to
        # compute the mask.
        m = Y.size(0)

        # Forward propagation of decoder. Use `dec_Q1`, `dec_K` and
        # `dec_V`.
        # <answer>
        Q = self.dec_Q1(Y)
        K = self.dec_K1(Y)
        V = self.dec_V1(Y)
        # </answer>

        # Compute square upper triangular mask matrix of size `m`. You
        # can use `torch.triu` and `torch.full` with `float("-inf")`.
        # <answer>
        M = torch.triu(torch.full((m, m), float("-inf")), diagonal=1)
        # </answer>

        # Score attention from `Q` and `K`. You can use `torch.bmm`
        # and `transpose` but don't forget to add the mask `M`.
        # <answer>
        S = (torch.bmm(Q.permute([1, 0, 2]), K.permute([1, 2, 0])) + M) / math.sqrt(self.p)
        # </answer>

        # Attention
        # <answer>
        A = F.softmax(S, dim=-1)
        T1 = torch.bmm(A, V.transpose(0, 1)).transpose(0, 1)
        # </answer>

        # First residual mapping and layer normalization
        # <answer>
        U1 = self.dec_ln1(self.dec_W0(T1) + Y)
        # </answer>

        # Key-value cross-attention using keys and values from the
        # encoder.
        # <answer>
        Q = self.dec_Q2(U1)
        K = self.dec_K2(Xp)
        V = self.dec_V2(Xp)
        # </answer>

        # Score attention from `Q` and `K`. You can either use
        # `torch.bmm` together with `torch.permute` or `torch.einsum`.
        # S = torch.bmm(Q.permute([1, 0, 2]), K.permute([1, 2, 0])) / math.sqrt(self.p)
        # <answer>
        S = torch.einsum('sbe,Sbe->bsS', Q, K) / math.sqrt(self.p)
        # </answer>

        # Attention
        # <answer>
        A = F.softmax(S, dim=-1)
        T2 = torch.bmm(A, V.transpose(0, 1)).transpose(0, 1)
        # </answer>

        # Second residual mapping and layer normalization
        # <answer>
        U2 = self.dec_ln2(self.dec_W1(T2) + U1)
        # </answer>

        # FFN on each token
        # <answer>
        Z = self.dec_W3(F.relu(self.dec_W2(U2)))
        # </answer>

        # Third residual mapping and layer normalization
        # <answer>
        U3 = self.dec_ln3(Z + U2)
        # </answer>

        return U3


# %% [markdown]
# ## Transformer model

# %%
class Transformer(nn.Module):
    def __init__(self, p=None, d_ff=None, vocab_size=None):
        super().__init__()

        # Declare an embedding, a positional encoder and a transformer
        # encoder.
        self.enc_embedding = nn.Embedding(vocab_size, p)
        self.enc_positional_encoding = PositionalEncoding(p)
        self.encoder = TransformerEncoder(p=p, d_ff=d_ff)

        # Declare an embedding, a positional encoder and a transformer
        # decoder.
        self.dec_embedding = nn.Embedding(vocab_size, p)
        self.dec_positional_encoding = PositionalEncoding(p)
        self.decoder = TransformerDecoder(p=p, d_ff=d_ff)

        self.generator = nn.Linear(p, vocab_size)

    def encode(self, X):
        # Use `self.enc_embedding`, `self.enc_positional_encoding` and
        # `self.encoder` to compute `Xp`
        # <answer>
        X_emb = self.enc_embedding(X)
        X_emb_pos = self.enc_positional_encoding(X_emb)
        Xp = self.encoder(X_emb_pos)
        # </answer>
        return Xp

    def decode(self, Xp, Y):
        # Use `self.dec_embedding`, `self.dec_positional_encoding` and
        # `self.decoder` to compute `outs`
        # <answer>
        Y_emb = self.dec_embedding(Y)
        Y_emb_pos = self.dec_positional_encoding(Y_emb)
        outs = self.decoder(Xp, Y_emb_pos)
        # </answer>
        return outs

    def forward(self, X, Y):
        Xp = self.encode(X)
        outs = self.decode(Xp, Y)
        return self.generator(outs)


def train_epoch(model: nn.Module, dataset: Dataset, optimizer: Optimizer):
    # Training mode
    model.train()

    # Set loss function to use. Don't forget to tell the loss function to
    # ignore entries that are padded.
    # <answer>
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    # </answer>

    # Turn `dataset` into an iterable on mini-batches using `DataLoader`.
    # <answer>
    train_dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )
    # </answer>

    losses = 0
    for X, Y in train_dataloader:
        # Select all but last element in sequences
        # <answer>
        Y_input = Y[:-1, :]
        # </answer>

        # Resetting gradients
        optimizer.zero_grad()

        # Compute output of transformer from `X` and `Y_input`.
        # <answer>
        scores = model(X, Y_input)
        # </answer>

        # Back-propagation through loss function
        # Select all but first element in sequences
        # <answer>
        Y_output = Y[1:, :]
        # </answer>

        # Compute the cross-entropy loss between `scores` and
        # `Y_output`. `scores` is `seq_length` * `batch_size` *
        # `vocab_size` and contains scores and `Y_output` is
        # `seq_length` * `batch_size` and contains integers. Two ways
        # to compute the loss:
        #
        # - reshape both tensors to have `batch_size` * `probs` for `scores` and
        #   `batch_size` for `Y_output`
        # - permute dimensions to have `batch_size` * `vocab_size` *
        #   `seq_length` for `scores` and `batch_size` * `seq_length` for
        #   `Y_output`
        # <answer>
        loss = loss_fn(scores.permute([1, 2, 0]), Y_output.permute([1, 0]))
        # </answer>

        # Gradient descent update
        loss.backward()
        optimizer.step()

        losses += loss.item()

    return losses / len(dataset)


# %% [markdown]
# ## Eval function

# %%
def evaluate(model: nn.Module, val_dataset: Dataset):
    model.eval()

    # Set loss function to use. Don't forget to tell the loss function to
    # ignore entries that are padded.
    # <answer>
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    # </answer>

    # Turn `val_dataset` into an iterable on mini-batches using `DataLoader`.
    # <answer>
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )
    # </answer>

    losses = 0
    for X, Y in val_dataloader:
        # Select all but last element in sequences
        # <answer>
        Y_input = Y[:-1, :]
        # </answer>

        # Compute output of transformer from `X` and `Y_input`.
        # <answer>
        scores = model(X, Y_input)
        # </answer>

        # Select all but first element in sequences
        # <answer>
        Y_output = Y[1:, :]
        # </answer>

        # Compute loss
        # <answer>
        loss = loss_fn(scores.permute([1, 2, 0]), Y_output.permute([1, 0]))
        # </answer>

        losses += loss.item()

    return losses / len(val_dataset)


# %% [markdown]
# ## Learning loop

# %%
transformer = Transformer(
    p=EMB_SIZE,
    d_ff=FFN_HID_DIM,
    vocab_size=VOCAB_SIZE
)

optimizer = Adam(transformer.parameters())

train_set = GotoDataset(n_sequences=N_SEQUENCES)
test_set = GotoDataset(n_sequences=N_SEQUENCES)

for epoch in range(1, NUM_EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(transformer, train_set, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer, test_set)
    print(
        (
            f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
            f"Epoch time = {(end_time - start_time):.3f}s"
        )
    )


# %% [markdown]
# ## Helpers functions

# %%
def greedy_decode(model, src, start_symbol_idx):
    """Autoregressive decoding of `src` starting with `start_symbol_idx`."""

    memory = model.encode(src)
    ys = torch.LongTensor([[start_symbol_idx]])
    maxlen = 100

    for i in range(maxlen):
        m = ys.size(0)
        tgt_mask = torch.triu(torch.full((m, m), float("-inf")), diagonal=1)

        # Decode `ys`. `out` is of size `curr_len` * 1 * `vocab_size`
        out = model.decode(memory, ys)

        # Select encoding of last token
        enc = out[-1, 0, :]

        # Get a set of scores on vocabulary
        dist = model.generator(enc)

        # Get index of maximum
        idx = torch.argmax(dist).item()

        # Add predicted index to `ys`
        ys = torch.cat((ys, torch.LongTensor([[idx]])))

        if idx == EOS_IDX:
            break
    return ys

def translate(model: torch.nn.Module, src_sentence: Iterable):
    """Translate sequence `src_sentence` with `model`."""

    model.eval()

    # Numericalize source
    src_tensor = torch.LongTensor(vocab(["<bos>"] + list(src_sentence) + ["<eos>"]))

    # Fake a minibatch of size one
    src = src_tensor.unsqueeze(1)

    # Translate `src`
    tgt_tokens = greedy_decode(model, src, BOS_IDX)

    tgt_tokens = tgt_tokens.flatten().numpy()
    return " ".join(vocab.lookup_tokens(list(tgt_tokens)[1:-1]))


input, output = dataset[2]

print("Input:", " ".join(input))
print("Output:", " ".join(output))
print("Pred:", translate(transformer, input))
