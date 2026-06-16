from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
batch_size = 4 # B dimension
block_size = 8 # T dimension
max_iters = 3000
learning_rate = 1e-3
eval_iters = 200
n_embd = 32

torch.manual_seed(1337)

with open(Path.cwd()/'playaround'/'nanogpt'/'input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(list(text)))
vocab_size = len(chars)

stoi = { c:i for i, c in enumerate(chars) }
itos = { i:c for i, c in enumerate(chars) }
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join([ itos[i] for i in l ])

# Data in Torch & Split data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Prepare data in (B, T, C) dimension
def get_batch(split):
    '''
    Return input and target for model.
    input and target is in (B, T) dimension.
    '''
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, size=(batch_size,))
    xb = torch.stack([ data[i:i+block_size] for i in ix ])
    yb = torch.stack([data[i+1:i + block_size + 1] for i in ix])
    return xb, yb

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        '''
        idx, targets: (B,T)
        Return:
        - logits (B * T, C), loss (1, 1) if targets is defined
        - logits (B, T, C), loss None if targets is None
        '''
        # logits score for each token
        logits = self.token_embedding_table(idx) # (B, T, C) Note: C is channel, vocab_size in this case.
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self.forward(idx)
            logits = logits[:, -1, :] # (B, C) on the last time step T
            probs = F.softmax(logits, dim=-1) # softmax on dimension C
            idx_next = torch.multinomial(probs, 1)
            idx = torch.concat((idx, idx_next), dim=-1)
        return idx

m = BigramLanguageModel()
xb, yb = get_batch('train')
init_context = torch.zeros((1, 1), dtype=torch.long) # start w/ single token
print(decode(m.generate(init_context, max_new_tokens=100)[0].tolist()))


optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(decode(m.generate(init_context, max_new_tokens=500)[0].tolist()))