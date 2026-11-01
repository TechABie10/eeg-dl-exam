"""
EEG Transformer Sprint — entry point.
Run:  python main.py

This script:
  1. loads the cross-subject split (train subjects 1-8, test 9-10),
  2. trains the two baselines (EEGNet, vanilla transformer),
  3. trains YOUR model across a few seeds and reports best + worst,
  4. saves one figure (figure.png).

>>> YOUR JOB <<<
Implement ONE inductive bias in `YourModel` below (or replace it entirely).
State your hypothesis in the README. Do not quietly edit the vanilla
transformer in models.py — the contrast is the point.
"""
import argparse
import numpy as np
import torch

from data import load_data, get_channel_positions
from models import EEGNet, VanillaEEGTransformer, count_params
from train import train_model, set_seed


# =========================================================================== #
#  YOUR MODEL  — the only part you must write.
#
#  The starter version below is the vanilla transformer with learned TEMPORAL
#  PATCH tokens instead of per-timestep tokens: one concrete, legal example of
#  "inject one inductive bias" (fewer tokens -> a smaller O(T^2) surface to
#  overfit on 8 subjects). Replace its guts with your own idea. Some directions:
#    - structured positional encoding from electrode geometry (get_channel_positions)
#      and/or frequency band,
#    - an attention bias toward scalp neighbourhoods or band-limited windows,
#    - channel (spatial) tokens instead of / in addition to time tokens,
#    - regularisation aimed specifically at the 8-subject data wall.
# =========================================================================== #
import torch.nn as nn


class YourModel(nn.Module):
    """ONE inductive bias: attend across ELECTRODES, not time, with each token
    placed on the scalp by its 3D position.

    Fists vs feet is a spatial contrast (medial foot area ~Cz vs lateral hand
    areas ~C3/C4; anterior-posterior, not left-right). The vanilla model makes
    one token per timestep (481 tokens, 481^2 attention) and mixes all 64
    electrodes in a single linear projection, so it must learn the spatial
    pattern from 8 subjects. Here:
      1. a shared temporal filter bank (same weights for every electrode) turns
         each electrode's 3 s signal into F log band-power features: power is
         what ERD changes, and it is invariant to trial timing;
      2. each electrode becomes one token (64 tokens, 64^2 attention), and an
         MLP on its (x, y, z) scalp coordinates is added as its positional
         encoding, so attention knows which electrodes neighbour each other and
         where front/back and midline/lateral are;
      3. a small transformer attends across electrodes; a CLS token reads out.
    """

    def __init__(self, n_chan, n_time, n_classes=2, pos=None,
                 n_filt=16, kern=65, d_model=32, n_heads=4, depth=2, p=0.3):
        super().__init__()
        self.temporal = nn.Conv1d(1, n_filt, kern, padding=kern // 2, bias=False)
        self.feat_norm = nn.BatchNorm1d(n_filt)
        self.embed = nn.Linear(n_filt, d_model)
        if pos is None:                          # fallback: no geometry
            pos = torch.zeros(n_chan, 3)
        pos = torch.as_tensor(pos, dtype=torch.float32)
        pos = (pos - pos.mean(0)) / (pos.norm(dim=1).max() + 1e-8)
        self.register_buffer("xyz", pos)         # (C, 3), fixed geometry
        self.pos_mlp = nn.Sequential(nn.Linear(3, d_model), nn.GELU(),
                                     nn.Linear(d_model, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2,
            dropout=p, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, return_embed=False):   # x: (B, C, T)
        B, C, T = x.shape
        h = self.temporal(x.reshape(B * C, 1, T))            # (B*C, F, T)
        h = torch.log(h.pow(2).mean(-1) + 1e-6)              # log band power (B*C, F)
        h = self.feat_norm(h).reshape(B, C, -1)              # (B, C, F)
        tok = self.embed(h) + self.pos_mlp(self.xyz)         # (B, C, d) + (C, d)
        cls = self.cls.expand(B, -1, -1)
        z = self.encoder(torch.cat([cls, tok], dim=1))       # (B, C+1, d)
        embed = self.norm(z[:, 0])
        logits = self.head(embed)
        return (logits, embed) if return_embed else logits


# =========================================================================== #


def run_across_seeds(build_fn, Xtr, ytr, Xte, yte, seeds, device, **kw):
    """Returns (peak, final) test accuracies per seed. The harness reports the
    PEAK over epochs, which is selected on test and therefore optimistic (the
    same bias fixed in Part 1, bug 5); FINAL is the last-epoch accuracy, which
    involves no selection on test. Both are printed so neither is hidden."""
    peaks, finals = [], []
    for s in seeds:
        set_seed(s)
        model = build_fn()
        best, final = train_model(model, Xtr, ytr, Xte, yte, device=device, **kw)
        peaks.append(best); finals.append(final)
        print(f"  seed {s}: peak test acc {best:.3f}  final-epoch test acc {final:.3f}")
    return np.array(peaks), np.array(finals)


def summarise(name, peaks, finals):
    print(f"{name}: peak best {peaks.max():.3f} worst {peaks.min():.3f} "
          f"mean {peaks.mean():.3f} | final best {finals.max():.3f} "
          f"worst {finals.min():.3f} mean {finals.mean():.3f} std {finals.std():.3f}")


def save_figure(model, Xte, yte, device, path="figure.png"):
    """t-SNE of the CLS embeddings on the test subjects, coloured by class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(Xte).to(device), return_embed=True)
        emb = out[1].cpu().numpy() if isinstance(out, tuple) else None
    if emb is None:
        print("  (your model did not return an embedding; skipping figure)")
        return
    n = len(emb)
    emb2 = TSNE(n_components=2, perplexity=min(30, max(5, n // 4)),
                init="pca", random_state=0).fit_transform(emb)
    plt.figure(figsize=(5, 4))
    for c, name in [(0, "fists"), (1, "feet")]:
        m = yte == c
        plt.scatter(emb2[m, 0], emb2[m, 1], s=18, alpha=0.7, label=name)
    plt.legend(); plt.title("Test-subject embeddings (your model)")
    plt.tight_layout(); plt.savefig(path, dpi=130)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", default=["eegnet", "vanilla", "yours"],
                    choices=["eegnet", "vanilla", "yours"],
                    help="subset of models to run (default: all three)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("\nloading data (first run downloads ~50 MB)...")
    Xtr, ytr, Xte, yte = load_data()
    C, T = Xtr.shape[1], Xtr.shape[2]
    print(f"train {Xtr.shape}  test {Xte.shape}  "
          f"(class balance train {ytr.mean():.2f} / test {yte.mean():.2f})")

    pos = get_channel_positions()                 # (C, 3), aligned to X's channels
    assert pos.shape == (C, 3) and np.isfinite(pos).all(), pos.shape

    # EEGNet with the configuration validated in Part 1 (kern=80 = 0.5 s at
    # 160 Hz, dropout 0.5), run across the same seeds as the transformers.
    if "eegnet" in args.models:
        print("\n=== Baseline 1: EEGNet (conv priors, Part 1 config) ===")
        print(f"params: {count_params(EEGNet(C, T, kern=80, p=0.5)):,}")
        pk_e, fi_e = run_across_seeds(lambda: EEGNet(C, T, kern=80, p=0.5),
                                      Xtr, ytr, Xte, yte, args.seeds, device,
                                      epochs=args.epochs)
        summarise("EEGNet", pk_e, fi_e)

    if "vanilla" in args.models:
        print("\n=== Baseline 2: vanilla transformer (no EEG prior) ===")
        print(f"params: {count_params(VanillaEEGTransformer(C, T)):,}")
        pk_v, fi_v = run_across_seeds(lambda: VanillaEEGTransformer(C, T),
                                      Xtr, ytr, Xte, yte, args.seeds, device,
                                      epochs=args.epochs)
        summarise("vanilla transformer", pk_v, fi_v)

    if "yours" in args.models:
        print("\n=== YOUR MODEL: electrode tokens + 3D position encoding ===")
        print(f"params: {count_params(YourModel(C, T, pos=pos)):,}")
        pk_y, fi_y = run_across_seeds(lambda: YourModel(C, T, pos=pos),
                                      Xtr, ytr, Xte, yte, args.seeds, device,
                                      epochs=args.epochs)
        summarise("your model", pk_y, fi_y)

        print("\nsaving figure...")
        set_seed(0)
        ym = YourModel(C, T, pos=pos)
        train_model(ym, Xtr, ytr, Xte, yte, epochs=args.epochs, device=device)
        save_figure(ym, Xte, yte, device)

    print("\ndone. Report best AND worst in the README — no cherry-picking.")


if __name__ == "__main__":
    main()
