"""Tokenizer round-trip tests (spec §2): en + Chichewa + code + punctuation."""
from triadlm.tokenizer import TriadTokenizer

SENTENCES: list[str] = [
    "The capital of Malawi is Lilongwe.",
    "Likulu la Malawi ndi Lilongwe. Moni! Zikomo!",
    "def add(a, b): return a + b  # code snippet",
    "Hello... (well: yes; no?) \"quoted\" — 12 * 8 = 96!",
    "Madzi, chakudya, nyumba: mawu a Chichewa.",
]


def test_round_trip(tmp_path) -> None:
    import glob
    files = sorted(glob.glob("data/raw/*.txt"))
    assert files, "seed corpus missing"
    tok = TriadTokenizer.train(files, vocab_size=512, save_dir=str(tmp_path / "tok"))
    tok2 = TriadTokenizer.load(str(tmp_path / "tok"))
    assert tok2.vocab_size == 512
    assert tok2.special_tokens == {"<bos>": 0, "<eos>": 1, "<pad>": 2, "<unk>": 3}
    for s in SENTENCES:
        assert tok2.decode(tok2.encode(s)) == s, s
    batch = tok2.encode_batch(SENTENCES)
    assert len(batch) == len(SENTENCES)
    for s, ids in zip(SENTENCES, batch):
        assert tok2.decode(ids) == s
