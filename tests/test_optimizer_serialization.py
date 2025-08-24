import json
from pathlib import Path

import torch

from psd.framework_optimizers import PSDTorch


def _compute_state_dict() -> dict:
    model = torch.nn.Linear(1, 1)
    model.weight.data.fill_(1.0)
    model.bias.data.fill_(0.0)
    opt = PSDTorch(model.parameters(), lr=0.1)

    x = torch.tensor([[1.0]])
    y = torch.tensor([[2.0]])
    criterion = torch.nn.MSELoss()

    def closure() -> torch.Tensor:
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        return loss

    opt.step(closure)
    return opt.state_dict()


def test_psd_torch_state_dict_matches_golden() -> None:
    state = _compute_state_dict()
    generated = json.dumps(state, sort_keys=True, indent=2, allow_nan=True)
    golden_path = Path(__file__).parent / "golden" / "psd_torch_state.json"
    expected = golden_path.read_text().strip()
    assert generated == expected
