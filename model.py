
import torch
import torch.nn as nn
import torch.nn.functional as F

class WordleNet(nn.Module):

    def __init__(self, guesses_mat, n_layers, device):

        super(WordleNet, self).__init__()

        self.guesses_mat = guesses_mat.reshape(-1, 130).to(device)

        units = [292] + [1000] * n_layers + [130]

        layers = [nn.Linear(units[0], units[1])]

        for i in range(1, len(units) - 1):
            layers += [nn.BatchNorm1d(units[i]), nn.LeakyReLU(0.2), nn.Dropout(0.2), nn.Linear(units[i], units[i+1])]

        self.main = nn.Sequential(*layers)

        self.criterion = CrossEntropy().to(device)

        self.device = device

    def init_hints(self, N):
        return torch.zeros(N, 286).to(self.device)

    def main_loop(self, hints, turn, sols):

        # Get output
        N = sols.shape[0]
        turn_vec = torch.zeros(N, 6).to(self.device)
        turn_vec[:,turn] = 1

        X = torch.cat((hints.reshape(N, -1), turn_vec), dim=1)
        y = self.main(X).reshape(N, 26, 5)

        # Compute loss
        loss = self.criterion(sols, F.log_softmax(y, dim=1))

        return F.softmax(y, dim=1), loss

    def get_nudges(self, word_mat):
        indices = torch.argmax(self.guesses_mat @ word_mat.transpose(0,1), dim=0)
        return (self.guesses_mat[indices,:] - word_mat).detach()

    def get_guesses(self, y_soft):

        y_soft = y_soft.reshape(y_soft.shape[0], -1)
        nudges = self.get_nudges(y_soft)

        return (y_soft + nudges).reshape(-1, 26, 5)

    def update_hints(self, guesses, sols):

        greens = guesses * sols
        sols_letters = sols.sum(dim=2).reshape(-1,26,1).clamp(min=0,max=1)
        yellows = (1 - greens) * (guesses * sols_letters)
        blacks = torch.sum(guesses * (1 - greens) * (1 - yellows), dim=2)

        solved = greens.reshape(greens.shape[0], -1).sum(dim=1) == 5

        return torch.cat((greens.reshape(-1, 130),
                            yellows.reshape(-1, 130),
                            blacks.reshape(-1, 26)) , dim=1), solved

    def forward(self, sols):

        hints = self.init_hints(sols.shape[0])
        total_loss = 0

        scores = 7 * torch.ones(sols.shape[0]).to(self.device)

        for turn in range(6):

            y_soft, loss = self.main_loop(hints, turn, sols)
            total_loss += loss if turn >= 1 else 0
            guesses = self.get_guesses(y_soft)

            if turn == 0:
                first_guess = guesses[0,:,:]

            new_hints, solved = self.update_hints(guesses, sols)
            hints += new_hints
            scores[(scores > turn+1) & solved] = turn+1

        return total_loss, scores.sum(), solved.sum(), first_guess


class CrossEntropy(nn.Module):

    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, sols, y_log_soft):
        return -(sols * y_log_soft).sum(dim=1).mean()
