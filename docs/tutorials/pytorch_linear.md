# Linear Regression with `PSDOptimizer`

A tiny PyTorch example showing how to fit a simple linear model using the `PSDOptimizer`.

```python
import torch
from psd_optimizer import PSDOptimizer

x = torch.tensor([[1.0], [2.0]])
y = torch.tensor([[2.0], [4.0]])

model = torch.nn.Linear(1, 1, bias=False)
opt = PSDOptimizer(model.parameters(), lr=0.1)
loss_fn = torch.nn.MSELoss()

for _ in range(50):
    def closure():
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        return loss
    opt.step(closure)

print(model.weight.data)
```

Expected output:

```
tensor([[2.]])
```
