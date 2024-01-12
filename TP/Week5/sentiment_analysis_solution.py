# %% [markdown]
# # Word embedding and RNN for sentiment analysis
#
# The goal of the following notebook is to predict whether a written
# critic about a movie is positive or negative. For that we will try
# three models. A simple linear model on the word embeddings, a
# recurrent neural network and a CNN.

# %%
from timeit import default_timer as timer
from typing import Iterable, List


import appdirs                  # Used to cache pretrained embeddings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torchtext import datasets
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# %% [markdown]
# ## The IMDB dataset

# %%
torch_cache = appdirs.user_cache_dir("pytorch")
train_iter, test_iter = datasets.IMDB(root=torch_cache, split=("train", "test"))

import random

TRAIN_SET = list(train_iter)
TEST_SET = list(test_iter)
random.shuffle(TRAIN_SET)
random.shuffle(TEST_SET)

# %%
TRAIN_SET[0]

# %% [markdown]
# ## Global variables
#
# First let's define a few variables. `EMBEDDING_DIM` is the dimension
# of the vector space used to embed all the words of the vocabulary.
# `SEQ_LENGTH` is the maximum length of a sequence, `BATCH_SIZE` is
# the size of the batches used in stochastic optimization algorithms
# and `NUM_EPOCHS` the number of times we are going thought the entire
# training set during the training phase.

# %%
# <answer>
EMBEDDING_DIM = 8
SEQ_LENGTH = 64
BATCH_SIZE = 512
NUM_EPOCHS = 10
# </answer>

# %% [markdown]
# We first need a tokenizer that take a text a returns a list of
# tokens. There are many tokenizers available from other libraries.
# Here we use the one that comes with Pytorch.

# %%
tokenizer = get_tokenizer("basic_english")
tokenizer("All your base are belong to us")

# %% [markdown]
# ## Building the vocabulary
#
# Then we need to define the set of words that will be understood by
# the model: this is the vocabulary. We build it from the training
# set.

# %%
def yield_tokens(data_iter: Iterable) -> List[str]:
    for data_sample in data_iter:
        yield tokenizer(data_sample[1])


special_tokens = ["<unk>", "<pad>"]
vocab = build_vocab_from_iterator(
    yield_tokens(TRAIN_SET),
    min_freq=10,
    specials=special_tokens,
    special_first=True)
UNK_IDX, PAD_IDX = vocab.lookup_indices(special_tokens)
VOCAB_SIZE = len(vocab)

vocab['plenty']

# %% [markdown]

# To limit the number of tokens in the vocabulary, we specified
# `min_freq=10`: a token should be seen at least 10 times to be part
# of the vocabulary. Consequently some words in the training set (and
# in the test set) are not present in the vocabulary. We then need to
# set a default index.

# %%
# vocab['pouet']                  # Error
vocab.set_default_index(UNK_IDX)
vocab['pouet']

# %% [markdown]
# # Collate function
#
# The collate function maps raw samples coming from the dataset to
# padded tensors of numericalized tokens ready to be fed to the model.

# %%
def collate_fn(batch: List):
    def collate(text):
        """Turn a text into a tensor of integers."""

        tokens = tokenizer(text)[:SEQ_LENGTH]
        return torch.LongTensor(vocab(tokens))

    src_batch = [collate(text) for _, text in batch]

    # Pad list of tensors using `pad_sequence`
    # <answer>
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    # </answer>

    # Turn 2 (positive review) and 1 (negative review) labels into 1 and 0
    # <answer>
    tgt_batch = torch.Tensor([label - 1 for label, _ in batch])
    # </answer>

    return src_batch, tgt_batch


print(f"Number of training examples: {len(TRAIN_SET)}")
print(f"Number of testing examples: {len(TEST_SET)}")

# %%
collate_fn([
    (1, "i am Groot")
])

# %% [markdown]
# ## Training a linear classifier with an embedding
#
# We first test a simple linear classifier on the word embeddings.


# %%
class EmbeddingNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, seq_length):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # Define an embedding of `vocab_size` words into a vector space
        # of dimension `embedding_dim`.
        # <answer>
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        # </answer>

        # Define a linear layer from dimension `seq_length` *
        # `embedding_dim` to 1.
        # <answer>
        self.l1 = nn.Linear(self.seq_length * self.embedding_dim, 1)
        # </answer>

    def forward(self, x):
        # `x` is of size `seq_length` * `batch_size`

        # Compute the embedding `embedded` of the batch `x`. `embedded` is
        # of size `batch_size` * `seq_length` * `embedding_dim`
        # <answer>
        embedded = self.embedding(x)
        # </answer>

        # Flatten the embedded words and feed it to the linear layer.
        # `flatten` is of size `batch_size` * (`seq_length` * `embedding_dim`)
        # <answer>
        flatten = embedded.view(-1, self.seq_length * self.embedding_dim)
        # </answer>

        # Apply the linear layer and return a squeezed version
        # `l1` is of size `batch_size`
        # <answer>
        return self.l1(flatten).squeeze()
        # </answer>


# %% [markdown]
# We need to implement an accuracy function to be used in the `Trainer`
# class (see below).


# %%
def accuracy(predictions, labels):
    # `predictions` and `labels` are both tensors of same length

    # Implement accuracy
    # <answer>
    return torch.sum((torch.sigmoid(predictions) > 0.5).float() == (labels > .5)).item() / len(
        predictions
    )
    # </answer>


assert accuracy(torch.Tensor([1, -2, 3]), torch.Tensor([1, 0, 1])) == 1
assert accuracy(torch.Tensor([1, -2, -3]), torch.Tensor([1, 0, 1])) == 2 / 3


# %% [markdown]
# Train and test functions

# %%
def train_epoch(model: nn.Module, optimizer: Optimizer):
    model.to(device)

    # Training mode
    model.train()

    loss_fn = nn.BCEWithLogitsLoss()

    train_dataloader = DataLoader(
        TRAIN_SET, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )

    matches = 0
    losses = 0
    for sequences, labels in train_dataloader:
        sequences, labels = sequences.to(device), labels.to(device)

        # Implement a step of the algorithm:
        #
        # - set gradients to zero
        # - forward propagate examples in `batch`
        # - compute `loss` with chosen criterion
        # - back-propagate gradients
        # - gradient step
        # <answer>
        optimizer.zero_grad()
        predictions = model(sequences)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()
        losses += loss.item()
        # </answer>

        acc = accuracy(predictions, labels)

        matches += len(predictions) * acc

    return losses / len(TRAIN_SET), matches / len(TRAIN_SET)

# %%
def evaluate(model: nn.Module):
    model.to(device)
    model.eval()

    loss_fn = nn.BCEWithLogitsLoss()

    val_dataloader = DataLoader(
        TEST_SET, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )

    losses = 0
    matches = 0
    for sequences, labels in val_dataloader:
        sequences, labels = sequences.to(device), labels.to(device)

        predictions = model(sequences)
        loss = loss_fn(predictions, labels)
        acc = accuracy(predictions, labels)
        matches += len(predictions) * acc
        losses += loss.item()

    return losses / len(TEST_SET), matches / len(TEST_SET)


# %%
def train(model, optimizer):
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        train_loss, train_acc = train_epoch(model, optimizer)
        end_time = timer()
        val_loss, val_acc = evaluate(model)
        print(
            f"Epoch: {epoch}, "
            f"Train loss: {train_loss:.3f}, "
            f"Train acc: {train_acc:.3f}, "
            f"Val loss: {val_loss:.3f}, "
            f"Val acc: {val_acc:.3f}, "
            f"Epoch time = {(end_time - start_time):.3f}s"
        )

# %%
def predict_sentiment(model, sentence):
    "Predict sentiment of given sentence according to model"

    tensor, _ = collate_fn([("dummy", sentence)])
    prediction = model(tensor)
    pred = torch.sigmoid(prediction)
    return pred.item()


# %%
embedding_net = EmbeddingNet(VOCAB_SIZE, EMBEDDING_DIM, SEQ_LENGTH)
print(sum(torch.numel(e) for e in embedding_net.parameters()))

device = "cuda:0" if torch.cuda.is_available() else "cpu"

optimizer = Adam(embedding_net.parameters())
train(embedding_net, optimizer)


# # %% [markdown]
# # ## Training a linear classifier with a pretrained embedding
# #
# # Load a GloVe pretrained embedding instead

# Download GloVe word embedding
glove = torchtext.vocab.GloVe(name="6B", dim="100", cache=torch_cache)

# Get token embedding of our `vocab`
vocab_vectors = glove.get_vecs_by_tokens(vocab.get_itos())

# tot_transferred = 0
# for v in vocab_vectors:
#     if not v.equal(torch.zeros(100)):
#         tot_transferred += 1

# tot_transferred, len(vocab)


# %%
class GloVeEmbeddingNet(nn.Module):
    def __init__(self, seq_length, vocab_vectors, freeze=True):
        super().__init__()
        self.seq_length = seq_length

        # Define `embedding_dim` from vocabulary and the pretrained `embedding`.
        # <answer>
        self.embedding_dim = vocab_vectors.size(1)
        self.embedding = nn.Embedding.from_pretrained(vocab_vectors, freeze=freeze)
        # </answer>

        self.l1 = nn.Linear(self.seq_length * self.embedding_dim, 1)

    def forward(self, x):
        # `x` is of size batch_size * seq_length

        # `embedded` is of size batch_size * seq_length * embedding_dim
        embedded = self.embedding(x)

        # `flatten` is of size batch_size * (seq_length * embedding_dim)
        flatten = embedded.view(-1, self.seq_length * self.embedding_dim)

        # L1 is of size batch_size
        return self.l1(flatten).squeeze()


glove_embedding_net1 = GloVeEmbeddingNet(SEQ_LENGTH, vocab_vectors, freeze=True)
print(sum(torch.numel(e) for e in glove_embedding_net1.parameters()))

optimizer = Adam(glove_embedding_net1.parameters())
train(glove_embedding_net1, optimizer)

# %% [markdown]
# ## Use pretrained embedding without fine-tuning

# Define model and freeze the embedding
# <answer>
glove_embedding_net1 = GloVeEmbeddingNet(SEQ_LENGTH, vocab_vectors, freeze=True)
# </answer>


# %% [markdown]
# ## Fine-tuning the pretrained embedding

# %%
# Define model and don't freeze embedding weights
# <answer>
glove_embedding_net2 = GloVeEmbeddingNet(SEQ_LENGTH, vocab_vectors, freeze=False)
# </answer>

# %% [markdown]
# ## Recurrent neural network with frozen pretrained embedding

# %%
class RNN(nn.Module):
    def __init__(self, hidden_size, vocab_vectors, freeze=True):
        super(RNN, self).__init__()

        # Define pretrained embedding
        self.embedding = nn.Embedding.from_pretrained(vocab_vectors, freeze=freeze)

        # Size of input `x_t` from `embedding`
        self.embedding_size = self.embedding.embedding_dim
        self.input_size = self.embedding_size

        # Size of hidden state `h_t`
        self.hidden_size = hidden_size

        # Define a GRU
        # <answer>
        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size)
        # </answer>

        # Linear layer on last hidden state
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, h0=None):
        # `x` is of size `seq_length` * `batch_size` and `h0` is of size 1
        # * `batch_size` * `hidden_size`

        # Define first hidden state in not provided
        if h0 is None:
            # Get batch and define `h0` which is of size 1 *
            # `batch_size` * `hidden_size`
            # <answer>
            batch_size = x.size(1)
            h0 = torch.zeros(self.gru.num_layers, batch_size, self.hidden_size).to(device)
            # </answer>

        # `embedded` is of size `seq_length` * `batch_size` *
        # `embedding_dim`
        embedded = self.embedding(x)

        # Define `output` and `hidden` returned by GRU:
        #
        # - `output` is of size `seq_length` * `batch_size` * `embedding_dim`
        #   and gathers all the hidden states along the sequence.
        # - `hidden` is of size 1 * `batch_size` * `embedding_dim` and is the
        #   last hidden state.
        # <answer>
        output, hidden = self.gru(embedded, h0)
        # </answer>

        # Apply a linear layer on the last hidden state to have a
        # score tensor of size 1 * `batch_size` * 1, and return a
        # tensor of size `batch_size`.
        # <answer>
        return self.linear(hidden).squeeze()
        # </answer>


rnn = RNN(hidden_size=100, vocab_vectors=vocab_vectors)
print(sum(torch.numel(e) for e in rnn.parameters() if e.requires_grad))

optimizer = optim.Adam(filter(lambda p: p.requires_grad, rnn.parameters()), lr=0.001)
train(rnn, optimizer)

# %% [markdown]
# ## CNN based text classification

# %%
class CNN(nn.Module):
    def __init__(self, vocab_vectors, freeze=False):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab_vectors, freeze=freeze)
        self.embedding_dim = self.embedding.embedding_dim

        self.conv_0 = nn.Conv2d(
            in_channels=1, out_channels=100, kernel_size=(3, self.embedding_dim)
        )
        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=100, kernel_size=(4, self.embedding_dim)
        )
        self.conv_2 = nn.Conv2d(
            in_channels=1, out_channels=100, kernel_size=(5, self.embedding_dim)
        )
        self.linear = nn.Linear(3 * 100, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Input `x` is of size `seq_length` * `batch_size`
        embedded = self.embedding(x)

        # The tensor `embedded` is of size `seq_length` * `batch_size` *
        # `embedding_dim` and should be of size `batch_size` *
        # (`n_channels`=1) * `seq_length` * `embedding_dim` for the
        # convolutional layers. You can use `transpose` and `unsqueeze` to make
        # the transformation.
        # <answer>
        embedded = embedded.transpose(0, 1).unsqueeze(1)
        # </answer>

        # Tensor `embedded` is now of size `batch_size` * 1 *
        # `seq_length` * `embedding_dim` before convolution and should
        # be of size `batch_size` * (`out_channels` = 100) *
        # (`seq_length` - `kernel_size[0]` + 1) after convolution and
        # squeezing.
        # Implement the convolution layer
        # <answer>
        conved_0 = self.conv_0(embedded).squeeze(3)
        conved_1 = self.conv_1(embedded).squeeze(3)
        conved_2 = self.conv_2(embedded).squeeze(3)
        # </answer>

        # Non-linearity step, we use ReLU activation
        # <answer>
        conved_0_relu = F.relu(conved_0)
        conved_1_relu = F.relu(conved_1)
        conved_2_relu = F.relu(conved_2)
        # </answer>

        # Max-pooling layer: pooling along whole sequence
        # Implement max pooling
        # <answer>
        seq_len_0 = conved_0_relu.shape[2]
        pooled_0 = F.max_pool1d(conved_0_relu, kernel_size=seq_len_0).squeeze(2)

        seq_len_1 = conved_1_relu.shape[2]
        pooled_1 = F.max_pool1d(conved_1_relu, kernel_size=seq_len_1).squeeze(2)

        seq_len_2 = conved_2_relu.shape[2]
        pooled_2 = F.max_pool1d(conved_2_relu, kernel_size=seq_len_2).squeeze(2)
        # </answer>

        # Dropout on concatenated pooled features
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

        # Linear layer
        return self.linear(cat).squeeze()


# %%
cnn = CNN(vocab_vectors)
optimizer = optim.Adam(cnn.parameters())
train(cnn, optimizer)

# %% [markdown]
# ## Test function
