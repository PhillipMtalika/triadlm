"""Model unit tests (spec §3): causality, loss-shift, ckpt restore, greedy determinism."""
import torch
from triadlm.model import GPT, GPTConfig


def tiny_cfg() -> GPTConfig:
    return GPTConfig(vocab_size=256, block_size=32, n_layer=2, n_head=4,
                     n_embd=64, dropout=0.0, bias=False, tie_weights=True)


def test_causal_mask() -> None:
    """Changing token t+1 must not change logits at position t."""
    torch.manual_seed(0)
    m = GPT(tiny_cfg()).eval()
    x = torch.randint(0, 256, (1, 10))
    x2 = x.clone()
    x2[0, 7] = (int(x2[0, 7]) + 1) % 256
    with torch.no_grad():
        l1, _ = m(x)
        l2, _ = m(x2)
    assert torch.allclose(l1[:, :7, :], l2[:, :7, :], atol=1e-5)


def test_loss_shift_shapes() -> None:
    """targets = idx[:, 1:], inputs = idx[:, :-1]; shapes line up."""
    torch.manual_seed(1)
    m = GPT(tiny_cfg())
    idx = torch.randint(0, 256, (2, 16))
    x, y = idx[:, :-1], idx[:, 1:]
    logits, loss = m(x, y)
    assert logits.shape == (2, 15, 256)
    assert loss is not None and loss.item() > 0


def test_checkpoint_restore(tmp_path) -> None:
    """Save/restore reproduces identical logits on a fixed input."""
    torch.manual_seed(2)
    m = GPT(tiny_cfg()).eval()
    x = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        before, _ = m(x)
    p = str(tmp_path / "m.pt")
    torch.save(m.state_dict(), p)
    m2 = GPT(tiny_cfg()).eval()
    m2.load_state_dict(torch.load(p, map_location="cpu"))
    with torch.no_grad():
        after, _ = m2(x)
    assert torch.equal(before, after)


def test_greedy_deterministic() -> None:
    """temperature=0 generation is deterministic."""
    torch.manual_seed(3)
    m = GPT(tiny_cfg()).eval()
    x = torch.randint(0, 256, (1, 5))
    a = m.generate(x.clone(), max_new_tokens=6, temperature=0.0)
    b = m.generate(x.clone(), max_new_tokens=6, temperature=0.0)
    assert torch.equal(a, b)
    assert a.shape[1] == 11
