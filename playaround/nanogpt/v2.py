from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
batch_size = 4 # B dimension
block_size = 8 # T dimension
max_iters = 5000
learning_rate = 1e-3
eval_iters = 200
eval_interval = 500
n_embd = 32
head_size = 16

torch.manual_seed(1337)

with open(Path.cwd()/'playaround'/'nanogpt'/'input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(list(text)))
vocab_size = len(chars)
# Tokenization - map character to numeric id
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

@torch.no_grad()
def estimate_loss():
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    return out

class Head(nn.Module):
    """
    Attention head
    """
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size) # (C, head_size)
        self.key = nn.Linear(n_embd, head_size) # (C, head_size)
        self.val = nn.Linear(n_embd, head_size) # (C, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, embd):
        """
        embd: (B, T, C)
        return (B, T, head_size)
        """
        B, T, C = embd.shape

        q = self.query(embd) # (B, T, head_size)
        k = self.key(embd) # (B, T, head_size)
        v = self.val(embd) # (B, T, head_size)

        # Scaled self-attention
        att = q @ k.transpose(1, 2) * head_size**-0.5 # (B, T, head_size) @ (B, head_size, T) → (B, T, T)

        # Mask future token
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=2)

        return att @ v # (B, T, T) @ (B, T, head_size) → (B, T, head_size)


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # (vocab_size, C)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # (T, C)

        self.sa_head = Head()
        self.lm_head = nn.Linear(head_size, vocab_size)

    def forward(self, idx, targets=None):
        '''
        idx, targets: (B,T)
        Return:
        - logits (B * T, C), loss (1, 1) if targets is defined
        - logits (B, T, C), loss None if targets is None
        '''
        B, T = idx.shape # B (batch_size), T (block_size)

        tok_embd = self.token_embedding_table(idx) # (B, T, C)
        pos_embd = self.position_embedding_table(torch.arange(T)) # (T, C)

        x = tok_embd + pos_embd # (B, T, C)

        out = self.sa_head(x) # (B, T, head_size)

        # logits score for each token
        logits = self.lm_head(out) # (B, T, vocab_size)
        if targets is not None:
            B, T, vocab_size = logits.shape
            logits = logits.view(B * T, vocab_size)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self.forward(idx_cond)
            logits = logits[:, -1, :] # (B, C) on the last time step T
            probs = F.softmax(logits, dim=-1) # softmax on dimension C
            idx_next = torch.multinomial(probs, 1)
            idx = torch.concat((idx, idx_next), dim=-1)
        return idx

model = BigramLanguageModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for iter in range(max_iters):
    # Evaluation
    if iter % eval_interval == 0 or iter == max_iters-1:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))