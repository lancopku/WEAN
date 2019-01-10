import torch.nn as nn
import torch

class EMA(nn.Module):
     def __init__(self, mu: float) -> None:
         super(EMA, self).__init__()
         self.mu = mu
         self.shadow = {}

     def register(self, name: str, val: torch.Tensor) -> None:
         self.shadow[name] = val.clone()

     def forward(self, name: str, x: torch.Tensor) -> torch.Tensor:
         assert name in self.shadow
         new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
         self.shadow[name] = new_average.clone()
         return new_average