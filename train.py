import math
import string
import time

import torch.nn as nn

from model import RNN
from utils import DataPrepare

# Hyper Parameters
n_hidden = 128
n_iters = 10000
learning_rate = 0.005
criterion = nn.NLLLoss()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    # why?
    s -= m * 60
    return "%dm %ds" % (m, s)


def train(rnn_model, category_tensor, line_tensor):
    hidden = rnn_model.initHidden()
    rnn_model.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn_model(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    for p in rnn_model.parameters():
        # why this alpha is minus learning rate
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


# Prepare for data
all_letters = string.ascii_letters + " .,;'"
dp = DataPrepare(all_letters)
dp.update_attr("dataset/data/names/*.txt")
n_categories = len(dp.all_categories)
n_letters = len(dp.all_letters)
rnn = RNN(n_letters, n_hidden, n_categories)


# Training process
start = time.time()
print_every = 500
plot_every = 100
current_loss = 0
all_losses = []

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = dp.randomTrainingExample()
    output, loss = train(rnn, category_tensor, line_tensor)
    current_loss += loss

    if iter % print_every == 0:
        guess, guess_i = dp.categoryFromOutput(output)
        correct = "✓" if guess == category else "✗ (%s)" % category
        print(
            "%d %d%% (%s) %.4f %s / %s %s"
            % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct)
        )

    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
