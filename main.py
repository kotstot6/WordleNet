
import argparse
import sys
import csv
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from model import WordleNet

parser = argparse.ArgumentParser(description='WordleNet implementation')

parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--batch_size', type=int, default=512, help='batch size for training')
parser.add_argument('--n_layers', type=int, default=0, help='number of hidden layers')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--wd', type=float, default=0, help='optimizer weight decay')
parser.add_argument('--n_epochs', type=int, default=10000, help='number of training epochs')
parser.add_argument('--epoch_step', type=int, default=1, help='epochs per validation checkpoint')
parser.add_argument('--split_word', type=str, default=None,
                                    help='all words before this word will be in train set. \
                                            all words including + after will be test set')
parser.add_argument('--pretrain', action='store_true', help='pretrain on uniform letter sequences')
parser.add_argument('--save_model', action='store_true', help='save model with best test score')
parser.add_argument('--load_model', type=str, default=None, help='file name of model to be fine-tuned')

parser.set_defaults(pretrain=False, save_model=False)

args = parser.parse_args()

settings = '-'.join(sys.argv[1:]).replace('---', '-').replace('--', '')


print('====================')
print('Settings:', settings)
print('====================')

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open('data/guesses.txt', 'r') as f:
    all_guesses = f.read().split('\n')[:-1]

with open('data/solutions.txt', 'r') as f:
    all_solutions = f.read().split('\n')[:-1]

alphabet = 'abcdefghijklmnopqrstuvwxyz'

if args.pretrain:
    all_guesses = [''.join(np.random.choice(list(alphabet), 5))
                            for i in range(50000)]

    all_solutions = all_guesses

def words2mat(words):
        word_mat = torch.Tensor([[[int(let1 == let2) for let2 in word]
                                                  for let1 in alphabet]
                                                     for word in words])

        return word_mat.reshape(len(words), 26, 5) if len(words) > 1 else word_mat.reshape(26, 5)

def mat2words(word_mat):

    word_indices = torch.argmax(word_mat.reshape(-1,26,5), dim=1)
    return [''.join([alphabet[int(index)] for index in indices]) for indices in word_indices]

guesses_mat = words2mat(all_guesses)


if args.split_word is not None:

    try:
        split_index = int(float(args.split_word) * len(all_solutions))
    except:
        split_index = all_solutions.index(args.split_word)

    train_solutions = all_solutions[:split_index]
    test_solutions = all_solutions[split_index:]
else:
    train_solutions, test_solutions = all_solutions, all_solutions

train_data = [words2mat([sol]) for sol in train_solutions]
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

test_data = [words2mat([sol]) for sol in test_solutions]
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = WordleNet(guesses_mat, args.n_layers, device).to(device)

if args.load_model is not None:
    net.load_state_dict(torch.load('models/' + args.load_model + '.pt'))

optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs*len(train_loader))

def train_loop():

    net.train()

    losses = []

    for sols in train_loader:

        optimizer.zero_grad()

        loss, _, _, _ = net(sols.to(device))
        losses.append(float(loss))

        loss.backward()
        optimizer.step()
        scheduler.step()

    return np.mean(losses)

def eval_loop(data_loader):

    net.eval()

    score_total, solved_total, word_total = 0, 0, 0

    for sols in data_loader:
        with torch.no_grad():
            n_words = sols.shape[0]
            _, score, solved, first_guess = net(sols.to(device))
            score_total += score
            solved_total += solved
            word_total += n_words

    score, solved = float(score_total / word_total), float(solved_total / word_total)
    first_guess = mat2words(first_guess)[0]
    print('Score:', score, '% Solved:', solved, ', First guess:', first_guess)

    return score, solved, first_guess

results = []
best_score = None

for epoch in range(1, args.n_epochs+1):

    loss = train_loop()

    print('Epoch:', epoch, ', Loss:', loss, ', LR:', scheduler.get_last_lr()[0])

    if epoch % args.epoch_step == 0:
        if args.split_word is not None:
            eval_loop(train_loader)
        score, solved, first_guess = eval_loop(test_loader)
        results.append((score, first_guess, epoch))

        if args.save_model and (best_score is None or score < best_score):
            torch.save(net.state_dict(), 'models/' + settings + '.pt')


best_score, best_first_guess, best_epoch = min(results, key=lambda x: x[0])

print('====================')
print('Best score:', best_score)
print('====================')

with open('metrics.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow([settings, best_score, best_first_guess, best_epoch])
