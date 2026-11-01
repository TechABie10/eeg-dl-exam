# EEG Transformer Sprint — Starter Kit

You have 90 minutes. This kit removes all the plumbing so you spend the time on
one thing: **giving self-attention a fighting chance on tiny EEG data, and
understanding why your idea does (or doesn't) help.**

## Setup (first ~10 minutes)

```bash
pip install -r requirements.txt
python data.py      # sanity check: downloads ~50 MB, prints array shapes
python main.py      # runs both baselines + the starter YourModel + saves figure.png
```

Everything runs on CPU in well under a minute per model. The first run
downloads the PhysioNet EEGBCI data and caches it.

> **Windows + conda note.** On this machine (conda-forge numpy/scipy plus pip
> torch), MNE's filtering crashed with `OMP: Error #15` (two OpenMP runtimes
> loaded). The numbers in this repo were produced with the standard workaround,
> set only in the shell: `set KMP_DUPLICATE_LIB_OK=TRUE` (cmd) or
> `export KMP_DUPLICATE_LIB_OK=TRUE` (bash). On a clean pip-only environment it
> is not needed.

## The task

**Both fists vs. both feet** (imagined movement), cross-subject: train on
subjects 1–8, test on held-out subjects 9–10.

This is a **bilateral** task. The discriminative signal is largely
anterior–posterior (hand/arm motor area vs. the medial foot representation near
the vertex, ~Cz), **not** the classic left/right lateralised mu/beta ERD. A
spatial prior that only encodes left vs. right will not help you here.

## What's in the box

| File | What it is |
|------|------------|
| `data.py` | Loader: download, 7–30 Hz filter, epoch, cross-subject split, per-channel z-score (train stats only). Also `get_channel_positions()` for 3D electrode coordinates. |
| `models.py` | `EEGNet` (conv-prior baseline) and `VanillaEEGTransformer` (per-timestep tokens, **no** EEG prior — the thing to beat/understand). |
| `train.py` | Shared train/eval loop, seed control, data loaders. |
| `main.py` | Entry point. Runs the baselines, runs **your** model across seeds, reports best + worst, saves `figure.png`. |

## Your job — in `main.py`

Implement **one** inductive bias in `YourModel` (or replace it entirely). The
starter version already shows one legal example: learned **temporal patch**
tokens instead of per-timestep tokens, which shrinks the O(T²) attention
surface. Swap in your own bet.

Do **not** quietly edit the vanilla transformer in `models.py` — the whole point
is the contrast between "attention with no help" and "attention with your one
idea."

## What to hand in

- This repo with your `YourModel` filled in. `python main.py` must reproduce
  your numbers end to end, seed printed.
- A short note (half a page — edit the section below is fine) with:
  1. **Hypothesis** — one line, written *before* you coded it.
  2. **What you changed and why** it targets a real weakness of attention on EEG.
  3. **What happened** — best *and* worst run across seeds, no cherry-picking.
  4. One sentence on what `figure.png` shows.
- A live code walkthrough at the end: share your screen, walk us through the
  file, answer questions.

Accuracy is **not** scored. We're reading your code and your answers for how you
reason about EEG and attention under a clock. A clear negative result beats a
lucky spike you can't explain.

---

## Your write-up (fill this in)

The full write-up (hypothesis, what changed and why, best/worst/mean results
for both runs, the figure, and caveats) is in [`../ANSWERS.md`](../ANSWERS.md),
Part 2. Reproduce with `python main.py` (full 40-epoch protocol), or the
reduced run reported there with
`python main.py --models eegnet yours --epochs 10`.
