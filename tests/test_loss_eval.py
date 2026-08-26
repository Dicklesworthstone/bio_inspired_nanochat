import math
import pytest
import torch

from bio_inspired_nanochat.loss_eval import evaluate_bpb


class _MockModelForBPB(torch.nn.Module):
    def __init__(self, device="cpu", per_token_loss=1.0):
        super().__init__()
        self.device = torch.device(device)
        self.per_token_loss = per_token_loss

    def get_device(self):
        return self.device

    def forward(self, x, y, loss_reduction="none"):
        # return per-token loss tensor matching y.shape
        return torch.full(y.shape, self.per_token_loss, dtype=torch.float32, device=self.device)


def test_evaluate_bpb_fast_path():
    device = torch.device("cpu")
    model = _MockModelForBPB(device=device, per_token_loss=math.log(2))  # 1 bit per token in nats

    vocab_size = 10
    # Each token is 1 byte
    token_bytes = torch.ones(vocab_size, dtype=torch.int64)

    # 2 batches of shape (2, 4)
    batches = [
        (torch.zeros(2, 4, dtype=torch.long), torch.ones(2, 4, dtype=torch.long)),
        (torch.zeros(2, 4, dtype=torch.long), torch.ones(2, 4, dtype=torch.long)),
    ]

    bpb = evaluate_bpb(model, batches, steps=2, token_bytes=token_bytes)
    # Total nats = 16 * ln(2), total bytes = 16
    # bpb = 16*ln(2) / (ln(2) * 16) = 1.0
    assert pytest.approx(bpb, 1e-5) == 1.0


def test_evaluate_bpb_with_ignored_targets():
    device = torch.device("cpu")
    model = _MockModelForBPB(device=device, per_token_loss=math.log(2))

    vocab_size = 10
    token_bytes = torch.ones(vocab_size, dtype=torch.int64)

    # First token of each sequence is masked with -1
    y1 = torch.tensor([[ -1, 1, 2, 3], [-1, 1, 2, 3]], dtype=torch.long)
    batches = [
        (torch.zeros_like(y1), y1),
    ]

    bpb = evaluate_bpb(model, batches, steps=1, token_bytes=token_bytes)
    # Valid tokens = 6 (each 1 byte), total nats = 6 * ln(2)
    # bpb = 6*ln(2) / (ln(2) * 6) = 1.0
    assert pytest.approx(bpb, 1e-5) == 1.0


def test_evaluate_bpb_with_special_zero_byte_tokens():
    device = torch.device("cpu")
    model = _MockModelForBPB(device=device, per_token_loss=math.log(2))

    vocab_size = 10
    token_bytes = torch.ones(vocab_size, dtype=torch.int64)
    token_bytes[0] = 0  # <|bos|> has 0 bytes

    # Target contains token 0
    y1 = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    batches = [
        (torch.zeros_like(y1), y1),
    ]

    bpb = evaluate_bpb(model, batches, steps=1, token_bytes=token_bytes)
    # Valid non-zero bytes tokens = 3
    assert pytest.approx(bpb, 1e-5) == 1.0


def test_evaluate_bpb_zero_total_bytes():
    device = torch.device("cpu")
    model = _MockModelForBPB(device=device, per_token_loss=1.0)

    vocab_size = 10
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64)  # all tokens 0 bytes

    y1 = torch.zeros((1, 4), dtype=torch.long)
    batches = [(torch.zeros_like(y1), y1)]

    bpb = evaluate_bpb(model, batches, steps=1, token_bytes=token_bytes)
    assert math.isinf(bpb)
