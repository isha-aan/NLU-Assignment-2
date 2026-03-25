names = [
"Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Reyansh", "Krishna", "Ishaan",
"Shaan", "Kabir", "Rohan", "Aryan", "Yash", "Kunal", "Rahul", "Ananya",
"Saanvi", "Aadhya", "Diya", "Myra", "Ira", "Navya", "Riya", "Sneha",
"Priya", "Pooja", "Kavya", "Neha", "Meera", "Anjali"
]

# Expand to 1000 names by repeating with slight variations
import random

generated_names = []

for _ in range(1000):
    name = random.choice(names)
    
    # slight variation
    if random.random() > 0.5:
        name = name + random.choice(["a", "n", "ya", "it"])
    
    generated_names.append(name)


# save file
with open("TrainingNames.txt", "w") as f:
    for name in generated_names:
        f.write(name + "\n")

print("Dataset created!")

import torch
import torch.nn as nn
import numpy as np

# Load names
with open("TrainingNames.txt", "r") as f:
    names = [line.strip().lower() for line in f]

# Build vocab (characters)
chars = sorted(list(set("".join(names))))
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

vocab_size = len(stoi)

# Convert names to sequences
def encode(name):
    return [stoi[c] for c in name]

encoded_names = [encode(name) for name in names]

print("Data prepared!")

max_len = max(len(name) for name in encoded_names)

def pad(seq):
    return seq + [0]*(max_len - len(seq))

X = [pad(name[:-1]) for name in encoded_names]
Y = [pad(name[1:]) for name in encoded_names]

X = torch.tensor(X)
Y = torch.tensor(Y)

print("Dataset ready!")

class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 32)
        self.rnn = nn.RNN(32, 64, batch_first=True)
        self.fc = nn.Linear(64, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

rnn_model = RNNModel()
print("RNN model ready!")

class BLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 32)
        self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

blstm_model = BLSTMModel()
print("BLSTM ready!")

class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 32)
        self.rnn = nn.RNN(32, 64, batch_first=True)
        self.attn = nn.Linear(64, 1)
        self.fc = nn.Linear(64, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)
        
        weights = torch.softmax(self.attn(out), dim=1)
        context = (weights * out).sum(dim=1)
        
        out = self.fc(context)
        return out

attn_model = AttentionModel()
print("Attention model ready!")

def train(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        optimizer.zero_grad()
        
        out = model(X)
        out = out.view(-1, vocab_size)
        target = Y.view(-1)
        
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

print("Training RNN...")
train(rnn_model)

print("Training BLSTM...")
train(blstm_model)

import random

def generate_name(model, max_len=10):
    model.eval()
    
    # start with random char
    ch = random.choice(list(stoi.keys()))
    result = ch

    for _ in range(max_len):
        x = torch.tensor([[stoi[c] for c in result]])
        
        with torch.no_grad():
            out = model(x)
        
        probs = torch.softmax(out[0, -1], dim=0).numpy()
        next_char = itos[np.random.choice(len(probs), p=probs)]
        
        result += next_char
    
    return result

print("\nGenerated names (RNN):")
for _ in range(10):
    print(generate_name(rnn_model))

print("\nGenerated names (BLSTM):")
for _ in range(10):
    print(generate_name(blstm_model))


generated = [generate_name(rnn_model) for _ in range(100)]

train_set = set(names)

# Novelty
novel = [name for name in generated if name not in train_set]
novelty_rate = len(novel) / len(generated)

# Diversity
diversity = len(set(generated)) / len(generated)

print("\nNovelty Rate:", novelty_rate)
print("Diversity:", diversity)