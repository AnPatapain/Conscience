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
    def __init__(self, head_size):
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
        att = q @ k.transpose(1, 2) * k.shape[-1]**-0.5 # (B, T, head_size) @ (B, head_size, T) → (B, T, T)

        # Mask future token
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=2)

        return att @ v # (B, T, T) @ (B, T, head_size) → (B, T, head_size)


class MultiHead(nn.Module):
    """
    Multi-head attention: instead of having one large attention head, we do attention in groups and then concat them.
    (i.e.: each attention-head learns different things, then we merge them.)
    """
    def __init__(self, nums_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(nums_head)])

    def forward(self, embd):
        return torch.concat([h(embd) for h in self.heads], dim=-1) # Merge on dim head_size -> (B, T, nums_head * head_size)

class FeedForward(nn.Module):
    """
    Feed forward network in Transformer model.
    Goal: instead of communicating to other tokens like attention layer, this layer allows each token to "learn" what
    it found from other tokens.
    Idea: give each token a space to "learn".
    """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(), # position-wise (token-level)
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, embd):
        return self.net(embd) # -> (B, T, C)

class Block(nn.Module):
    """
    One decoder block.
    Computation (FFW) followed by communication (ATT)
    """
    def __init__(self, n_embd, nums_head):
        super().__init__()
        self.attention = MultiHead(nums_head, n_embd//4)
        self.ffw = FeedForward(n_embd)

    def forward(self, embd):
        return self.ffw(self.attention(embd))


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # (vocab_size, C)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # (T, C)
        # self.sa_head = MultiHead(4, n_embd//4)
        # self.ffwd = FeedForward(n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, 4),
            Block(n_embd, 4),
            Block(n_embd, 4),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

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

        # Make x go through: communication (self-attention) then computation (feed forward network).
        x = self.blocks(x)

        # logits score for each token
        logits = self.lm_head(x) # (B, T, vocab_size)
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