"""Resume-from-checkpoint reproduces the identical loss curve (spec §5, required)."""
import yaml
import triadlm.train as T


def _write_cfg(tmp_path, name, max_steps, out_sub) -> str:
    cfg = {
        "model": {"vocab_size": 512, "block_size": 32, "n_layer": 2,
                  "n_head": 2, "n_embd": 64, "dropout": 0.0,
                  "bias": False, "tie_weights": True},
        "train": {"batch_size": 2, "grad_accum_steps": 1, "max_steps": max_steps,
                  "lr": 5e-4, "min_lr": 5e-5, "warmup_steps": 1,
                  "weight_decay": 0.0, "betas": [0.9, 0.95], "grad_clip": 1.0,
                  "precision": "fp32", "eval_interval": 100, "eval_iters": 2,
                  "checkpoint_interval": 2, "seed": 7},
        "data": {"manifest": str(tmp_path / "m.json"),
                 "shard_dir": str(tmp_path / "shards")},
        "tokenizer": {"vocab_size": 512, "save_dir": str(tmp_path / "tok")},
        "run": {"out_dir": str(tmp_path / out_sub)},
    }
    p = str(tmp_path / name)
    yaml.safe_dump(cfg, open(p, "w"))
    return p


def test_resume_reproduces_loss(tmp_path) -> None:
    import glob
    from triadlm.tokenizer import TriadTokenizer
    from data.preprocess import preprocess
    T.REGISTRY_PATH = str(tmp_path / "registry.csv")
    TriadTokenizer.train(sorted(glob.glob("data/raw/*.txt")), 512,
                         str(tmp_path / "tok"))
    preprocess("data/raw/*.txt", str(tmp_path / "tok"),
               str(tmp_path / "shards"), str(tmp_path / "m.json"),
               train_split=0.9, corpus_name="toy-test")
    full = T.train(_write_cfg(tmp_path, "c4.yaml", 4, "o1"))
    T.train(_write_cfg(tmp_path, "c2.yaml", 2, "o1"))
    resumed = T.train(_write_cfg(tmp_path, "c4b.yaml", 4, "o2"),
                      resume=str(tmp_path / "o1" / "step_2.pt"))
    assert full["loss_history"][2:] == resumed["loss_history"]
    assert resumed["steps"] == 4
