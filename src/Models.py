import torch
import torch.nn.functional as F

class lstm_model(torch.nn.Module):

    def __init__(self):
        super(lstm_model, self).__init__()
        self.lstm1=torch.nn.LSTM(batch_first=True, input_size=5, hidden_size=1)
        self.out=torch.nn.Linear(1,1)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm1(x)
        x = self.out(x)
        return x.flatten()

