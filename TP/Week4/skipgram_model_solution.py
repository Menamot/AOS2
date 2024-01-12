# %% [markdown]
# # Skipgram model trained on "20000 lieues sous les mers"
#
# ## Needed libraries

# You will need the following new libraries:
# - `spacy` for tokenizing
# - `gensim` for cosine similarities (use `gensim>=4.0.0`)

# You will also need to download rules for tokenizing a french text.
# ```python
# python -m spacy download fr_core_news_sm
# ```

# %%
import numpy as np
import torch
from torch import nn
import torch.optim as optim

import spacy
from gensim.models.keyedvectors import KeyedVectors

# %%
spacy_fr = spacy.load("fr_core_news_sm")


# %% [markdown]
# ## Tokenizing the corpus

# %%
# Use a french tokenizer to Create a tokenizer for the french language
with open("data/20_000_lieues_sous_les_mers.txt", "r", encoding="utf-8") as f:
    document = spacy_fr.tokenizer(f.read())

# Define a filtered set of tokens by iterating on `document`. Define a
# subset of tokens that are
#
# - alphanumeric
# - in lower case
# <answer>
tokens = [
    tok.text.lower()
    for tok in document if tok.is_alpha or tok.is_digit
]
# </answer>

# Make a list of unique tokens and dictionary that maps tokens to
# their index in that list.
# <answer>
idx2tok = list(set(tokens))
tok2idx = {token: i for i, token in enumerate(idx2tok)}
# </answer>

# %% [markdown]
# ## The continuous bag of words model

# %%
class Skipgram(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # Define an Embedding module (`nn.Embedding`) and a linear
        # transform (`nn.Linear`) without bias.
        # <answer>
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.U_transpose = nn.Linear(self.embedding_size, self.vocab_size, bias=False)
        # </answer>

    def forward(self, center):
        # Implements the forward pass of the skipgram model
        # `center` is of size `batch_size`

        # `e_i` is of size `batch_size` * `embedding_size`
        # <answer>
        e_i = self.embeddings(center)
        # </answer>

        # `UT_e_i` is of size `batch_size` * `vocab_size`
        # <answer>
        UT_e_i = self.U_transpose(e_i)
        # </answer>

        # <answer>
        return UT_e_i
        # </answer>


# Set the size of vocabulary and size of embedding
VOCAB_SIZE = len(idx2tok)
EMBEDDING_SIZE = 32

# Create a Continuous bag of words model
skipgram = Skipgram(VOCAB_SIZE, EMBEDDING_SIZE)

# Send to GPU if any
device = "cuda:0" if torch.cuda.is_available() else "cpu"
skipgram.to(device)

# %% [markdown]
# ## Preparing the data

# %%
# Generate n-grams for a given list of tokens, use yield, use window length of n-grams
def ngrams_iterator(token_list, ngrams):
    """Generates successive N-grams from a list of tokens."""

    for i in range(len(token_list) - ngrams + 1):
        idxs = [tok2idx[tok] for tok in token_list[i:i+ngrams]]

        # Get center element in `idxs`
        center = idxs.pop(ngrams // 2)

        # Yield the index of center word and indexes of context words
        # as a Numpy array (for Pytorch to automatically convert it to
        # a Tensor).
        yield center, np.array(idxs)


# Create center, context data
NGRAMS = 5
ngrams = list(ngrams_iterator(tokens, NGRAMS))

BATCH_SIZE = 512
data = torch.utils.data.DataLoader(ngrams, batch_size=BATCH_SIZE, shuffle=True)

# %% [markdown]
# ## Learn Skipgram model

# %%
# Use the Adam algorithm on the parameters of `skipgram` with a learning
# rate of 0.01
# <answer>
optimizer = optim.Adam(skipgram.parameters(), lr=0.01)
# </answer>

# Use a cross-entropy loss from the `nn` submodule
# <answer>
ce_loss = nn.CrossEntropyLoss()
# </answer>

# %%
EPOCHS = 20
for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    for i, (center, context) in enumerate(data):
        center, context = center.to(device), context.to(device)

        # Reset the gradients of the computational graph
        # <answer>
        skipgram.zero_grad()
        # </answer>

        # Forward pass
        # <answer>
        UT_e_i = skipgram.forward(center)
        # </answer>

        # Define one-hot encoding for tokens in context. `one_hots` has the same
        # size as `UT_e_i` and is zero everywhere except at location
        # corresponding to `context`. You can use `torch.scatter`.
        # <answer>
        one_hots = torch.zeros_like(UT_e_i).scatter(1, context, 1/(NGRAMS-1))
        # </answer>

        # Compute loss between `UT_e_i` and `one_hots`
        # <answer>
        loss = ce_loss(UT_e_i, one_hots)
        # </answer>

        # Backward pass to compute gradients of each parameter
        # <answer>
        loss.backward()
        # </answer>

        # Gradient descent step according to the chosen optimizer
        # <answer>
        optimizer.step()
        # </answer>

        total_loss += loss.data

        if i % 20 == 0:
            loss_avg = float(total_loss / (i + 1))
            print(
                f"Epoch ({epoch}/{EPOCHS}), batch: ({i}/{len(data)}), loss: {loss_avg}"
            )

    # Print average loss after each epoch
    loss_avg = float(total_loss / len(data))
    print("{}/{} loss {:.2f}".format(epoch, EPOCHS, loss_avg))


# %% [markdown]
# ## Prediction functions

# Now that the skipgram model is learned we can give it a word and see what
# context the model predicts.

# %%
def predict_context_words(skipgram, center_word, k=4):
    """Predicts `k` best context words of `center_word` according to model `skipgram`"""

    # Get index of `center_word`
    center_word_idx = tok2idx[center_word]

    # Create a fake minibatch containing just `center_word_idx`. Make sure that
    # `fake_minibatch` is a Long tensor and don't forget to send it to device.
    # <answer>
    fake_minibatch = torch.LongTensor([center_word_idx]).unsqueeze(0).to(device)
    # </answer>

    # Forward propagate through the skipgram model
    # <answer>
    score_context = skipgram(fake_minibatch).squeeze()
    # </answer>

    # Retrieve top k-best indexes using `torch.topk`
    # <answer>
    _, best_idxs = torch.topk(score_context, k=k)
    # </answer>

    # Return actual tokens using `idx2tok`
    # <answer>
    return [idx2tok[idx] for idx in best_idxs]
    # </answer>

# %%
predict_context_words(skipgram, "mille")
predict_context_words(skipgram, "nemo")


# %% [markdown]
# ## Testing the embedding
#
# We use the library `gensim` to easily compute most similar words for
# the embedding we just learned.

# %%
m = KeyedVectors(vector_size=EMBEDDING_SIZE)
m.add_vectors(idx2tok, skipgram.embeddings.weight.detach().cpu().numpy())

# %% [markdown]
# You can now test most similar words for, for example "lieues",
# "mers", "professeur"... You can look at `words_decreasing_freq` to
# test most frequent tokens.

# %%
unique, freq = np.unique(tokens, return_counts=True)
idxs = freq.argsort()[::-1]
words_decreasing_freq = list(zip(unique[idxs], freq[idxs]))

# %%
# <answer>
m.most_similar("lieues")
m.most_similar("professeur")
m.most_similar("mers")
m.most_similar("noire")
m.most_similar("mètres")
m.most_similar("ma")
# </answer>
