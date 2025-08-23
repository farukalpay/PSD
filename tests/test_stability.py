import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from psd_optimizer import PSDOptimizer  # noqa: E402


def test_gradient_clipping_prevents_explosion() -> None:
    p = torch.nn.Parameter(torch.tensor([10.0]))
    opt = PSDOptimizer([p], lr=1.0, max_grad_norm=1.0)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = (p**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    assert torch.allclose(p.detach(), torch.tensor([9.0]), atol=1e-6)


def test_nan_gradients_handled_gracefully() -> None:
    p = torch.nn.Parameter(torch.tensor([1.0]))
    lr = 1.0
    opt = PSDOptimizer([p], lr=lr)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        p.grad = torch.tensor([float("nan")])
        return torch.tensor(0.0)

    opt.step(closure)
    assert p.item() == pytest.approx(1.0)
    assert opt.param_groups[0]["lr"] < lr


def test_high_condition_number_quadratic() -> None:
    d = 1000
    diag = torch.linspace(1.0, 1e6, d)
    x = torch.randn(d, requires_grad=True)
    opt = PSDOptimizer([x], lr=1e-3, max_grad_norm=10.0)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = 0.5 * (diag * x**2).sum()
        loss.backward()
        return loss

    initial_loss = closure().item()
    for _ in range(5):
        opt.step(closure)
    final_loss = closure().item()
    assert final_loss < initial_loss
