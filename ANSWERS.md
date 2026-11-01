# ANSWERS

Repository: https://github.com/TechABie10/eeg-dl-exam. Every number below can be
reproduced from `part1_diagnose/` with
`python train.py --channels <all|frontal|motor> --seed <0-4>` and `python erd.py --band mu`.
Raw results are in `part1_diagnose/results.jsonl`.

---

## Part 1: Diagnose & Trust

### Task 1: Bugs

**Before vs after (held-out test accuracy, `--channels all`):**

| Pipeline | Seeds | Test accuracy per seed | Mean |
| --- | --- | --- | --- |
| Buggy starter (`a840162`) | 0, 1, 2 | 66.7 / 75.0 / 78.6% | **73.4%** |
| All fixes (`902a6fa`) | 0-4 | 52.2 / 46.7 / 71.1 / 47.8 / 53.3% | **54.2%** |

Each test set after the fixes holds 90 trials from 2 unseen subjects. Beating
chance at p<0.05 (exact binomial) needs 54/90 = 60%, which only seed 2 (test
subjects 2 and 5) reaches. The buggy number came from decoding the wrong task,
testing on people the model had trained on, and picking the checkpoint on the
test set.

**The six bugs (one commit each, in this order; accuracy re-run after every fix,
seeds 0-2):**

| # | Commit | File | What was wrong | Effect on the result | Mean after fix |
| --- | --- | --- | --- | --- | --- |
| 1 | `dc60873` | `data.py` | `RUNS = [5, 9, 13]` are *executed* fists/feet runs; imagery is 6, 10, 14. | **Up** (wrong). Real movement gives stronger ERD plus EMG and movement artefacts. | 64.3% (-9.1) |
| 2 | `233ad34` | `data.py` | `TMAX = 5.0` runs past the end of each 125 s run, so MNE silently dropped the last trial of every run (42 of 45 per subject). The last cue is only 4.09-4.59 s before the end; now `TMAX = 4.0`. | **Down** slightly (6.7% of data lost); within noise. | 65.9% (+1.6) |
| 3 | `782ea84` | `train.py` | `train_test_split` shuffled *trials*, so all 10 subjects were in train and test, despite the "subject-independent" comment. Now `GroupShuffleSplit` on subject ids. | **Up** (wrong): the model was tested on people it had seen. Mean held, but spread widened to 54-74% because bug 5 still masked the loss. | 64.1% |
| 4 | `ec9a4af` | `train.py` | Z-score mean/std computed on all trials, test included, before the split. Now training trials only. | **Up**, tiny leak. No measurable change. | 64.1% |
| 5 | `bcc9fff` | `train.py` | Best checkpoint chosen by **test** accuracy; validation printed and ignored. Now chosen on `va_acc`. | **Up** (wrong). Max of 30 noisy test scores. On the fixed code, test-picking gives 62.4% vs 54.2% honest (5 seeds). | 56.7% on seeds 0-2 (-7.4); 54.2% on seeds 0-4 |
| 6 | `902a6fa` | `erd.py` | ERD `BASELINE = (0.0, 0.5)` is *after* the cue and contains the cue-evoked response. Now (-1.5, -0.5) s. | No accuracy change. It pushed true rest to about -10% and shrank the real ERD. | (plot only) |

How I knew each one was wrong:
1. The files downloaded were `R05/R09/R13`, and Part 2's loader uses 6/10/14 for the same task.
2. MNE's drop log showed 3 `BAD boundary`/`TOO_SHORT` drops per subject, at the end of each run.
3. I rebuilt the split: every subject had 5-13 test trials.
4. From the code order: normalisation ran before the split.
5. Seed 0's log: the validation peak at epoch 14 had 61.9% test, but 66.7% was reported.
6. The buggy plot had a +0.25 s spike inside the baseline window and pre-cue rest at about -10%.

### Task 2: ERD (`erd_mu.png`)

After the fixes, pre-cue rest sits at 0%, and hands imagery produces sustained
mu ERD from about 0.5 s: roughly -15 to -30% at C3 and -10 to -25% at C4. That
matches the bilateral hand areas on the lateral convexity under C3/C4. Feet
imagery gives weaker ERD at C3/C4 (about -5 to -15%, returning towards 0% after
2 s). At Cz, both classes desynchronise by a similar -10 to -20%, so the
expected feet-specific ERD at the vertex is not clear. Cz is weak because the
foot area lies on the medial wall deep in the interhemispheric fissure. Its
sources are farther from the scalp and tangentially oriented, and the left and
right foot areas face each other, so their fields partly cancel at Cz. (The
brief positive spike at about +0.25 s at every site is the phase-locked
response to the visual cue, not imagery.)

### Task 3: What is the model using?

`frontal` = Fp1, Fpz, Fp2, AF7, AF3, AFz, AF4, AF8. I also added a `motor` control
group (FC3-FC4, C3-C4, CP3-CP4). Same five subject splits:

| Channels | Seeds 0-4 | Mean |
| --- | --- | --- |
| all (64) | 52.2 / 46.7 / 71.1 / 47.8 / 53.3% | 54.2% |
| motor (15) | 53.3 / 53.3 / 62.2 / 65.6 / 58.9% | 58.7% |
| frontal (8) | 52.2 / 58.9 / 68.9 / 52.2 / 62.2% | 58.9% |

Eight frontal electrodes with no motor cortex under them decode fists vs feet
as well as the motor strip and better than all channels, so the model can reach
its accuracy without any motor-imagery signal. The likely non-brain carrier is
vertical electro-oculogram (EOG): the fists target is at the top of the screen
and the feet target at the bottom, so the eyes move up or down. Cue-locked
forehead averages (0.1-15 Hz) confirm it. From about 0.2 s (saccade latency),
Fp1/Fpz/Fp2 swing about +30 uV more positive for fists than feet, in 9 of 10
subjects: the positive cornea moves towards the forehead when looking up. The
same-sign shift also reaches Cz (+20.8 uV, 10/10 subjects), so even the motor
group is partly contaminated. Frontalis EMG from looking up is a secondary
suspect. (Gaps under about 5 points are within noise at 90 test trials.)

### Task 4: Short answers

1. **45 trials at 58%:** that is 26/45 correct. An exact one-sided binomial
   test against 50% gives p = 0.19 (95% Wilson CI 43-71%), so it is **not**
   above chance; at p<0.05 it would need 29/45 (64%). Better still, run a
   label-permutation test (shuffle labels, retrain, repeat) to get the null
   distribution for this exact pipeline.
2. **EEGNet's depthwise layer** `Conv2d(F1, F1*D, (n_ch, 1), groups=F1)`: for
   each of the F1 temporal (frequency-band) filters, it learns D weightings
   across all electrodes, i.e. frequency-specific spatial filters, like CSP.
   Each one combines the scalp channels into a virtual source (for example,
   mu power over C3/C4 vs Cz). Their weights read as scalp topographies, and
   `groups=F1` keeps each band's spatial filters separate.
3. **Removing the eye confound:** present the cue at the centre of the screen
   (or as an auditory cue) with a fixation cross, so gaze direction no longer
   differs between fists and feet. In the analysis instead, remove ocular
   activity (ICA, or regression on EOG/Fp channels) and reject saccade trials
   before training.

---

## Part 2: Build & Defend

*(to be filled in)*

**Hypothesis:**

**What I changed and why:**

**Results (best / worst / mean across seeds):**

**Figure:**
