import torch

class Net(torch.nn.Module):
    def __init__(self, ks=256, pool_size=3):
        super(Net, self).__init__()
        self.c = torch.nn.Sequential(
            torch.nn.Conv1d(2, ks, 33, stride=1, padding=16),
            torch.nn.MaxPool1d(pool_size, padding=0),
            torch.nn.Tanh(),
        )
        self.rnns = torch.nn.ModuleList([
            torch.nn.GRU(ks, ks, 1, bidirectional=False),
            torch.nn.GRU(ks, ks, 1, bidirectional=False),
            torch.nn.GRU(ks, ks, 1, bidirectional=False),
            torch.nn.GRU(ks, ks, 1, bidirectional=False),
            torch.nn.GRU(ks, ks, 1, bidirectional=False),
            torch.nn.GRU(ks, ks, 1, bidirectional=False),
        ])
        
        self.out = torch.nn.Linear(ks, 5)
        
    def forward(self, x):
        x = x.permute((0,2,1))
        x = self.c(x)
        x = x.permute((2,0,1))
        for rnn in self.rnns:
            x, _ = rnn(x)
            x = x.flip([0])
        x = x.permute((1,0,2))
        return self.out(x)
