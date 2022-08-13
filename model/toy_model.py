import torch.nn as nn

class ToyModel(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.mlp = nn.Linear(in_features, out_features, bias=True)
                
    def forward(self, hidden_state):
        return self.mlp(hidden_state).squeeze()