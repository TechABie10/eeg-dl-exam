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

**Why each fix moved the result, with confidence grades.** Grades are my
confidence in the explanation, not a clinical evidence standard:
**A** = well-established mechanism that fits the size of the change;
**B** = plausible mechanism, not directly tested here;
**C** = the change is within noise, so no mechanism is claimed.

| # | Most likely cause of the change | Grade |
| --- | --- | --- |
| 1 | Executed movement drives sensorimotor cortex harder than imagery (motor output plus sensory feedback from the moving limbs gives larger, more consistent mu/beta ERD), and adds non-brain class signal: forearm/neck/jaw EMG at 20-40 Hz inside the 1-40 Hz band, and electrode-shift artefacts. | A |
| 2 | 7% more data and a slightly different 90-trial test set; one standard error is about 5 points. | C |
| 3 | People differ in skull thickness, cortical folding (which sets the scalp topography), individual mu peak frequency, cap placement and imagery strategy, so accuracy depends on who is held out. The mean held only because bug 5 still masked the loss. | B |
| 4 | Channel statistics pooled over 6-8 people barely change when two are left out. | C |
| 5 | Statistical: the best of 30 noisy test scores is a lucky draw; selecting on validation removes it. | A |
| 6 | The 0-0.5 s window contained the visual evoked response to the cue (P1/N1, about 100-200 ms, volume-conducted to central sites) and probably the saccade to the target, inflating the reference power. | A |

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

Confidence grades: hand-area ERD at C3/C4 matching the somatotopy, **A**;
the foot-area depth and orientation explaining the weak Cz difference, **B**
(the anatomy is established, but its effect was not measured here); the
+0.25 s spike being a cue-evoked response, **A**.

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

Confidence grades: vertical EOG as the main non-brain carrier, **A** (the
polarity, latency, amplitude and frontal maximum all match the cornea-retina
dipole); frontalis EMG contributing, **B** (not tested); eye potential
contaminating the motor group via Cz, **B** (the shift was measured, but its
contribution to the motor group's accuracy was not).

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

### Limitations

The pipeline is now free of the data leaks I found. The *experiment* is not:

1. **Eye-movement confound (largest).** Frontal electrodes decode as well as
   the motor strip, and the eye potential reaches Cz (Task 3). So 54.2% is an
   upper bound on motor-imagery decoding, not a clean measure of it. Next
   check: remove ocular activity (ICA, or regression on Fp channels) and
   re-run the motor group.
2. **Few, overlapping test sets.** Each estimate rests on 90 trials from 2
   people, and the five seeds reuse subjects (3 and 9 each appear twice), so
   they are not independent. Leave-two-subjects-out cross-validation would
   test every subject exactly once.
3. **Noisy validation.** Two validation subjects make checkpoint selection
   fragile (seed 1 chose epoch 1).
4. **Chance test assumes independent trials.** Trials within one person are
   correlated, so the binomial p-values are somewhat optimistic. A
   label-permutation test would give the real null distribution.

---

## Part 2: Build & Defend

**Hypothesis** (written and committed before any model code):
Fists vs feet is a *spatial* contrast (medial foot area near Cz vs the lateral
hand areas near C3/C4, which is anterior-posterior and medial-lateral, not
left-right). So attention over **electrode tokens**, each carrying learned
band-power features plus an encoding of its 3D scalp position, will beat
per-timestep attention. It removes the 481x481 temporal attention surface the
vanilla model overfits on 8 subjects. I expect it to beat the vanilla
transformer and to cut seed variance more than it raises peak accuracy.

**What I changed and why:**
`YourModel` (in `part2_transformer/main.py`, commit `baff821`) swaps the axis
attention runs over: **electrodes instead of timesteps**.

- *Weakness targeted.* The vanilla transformer makes one token per timestep
  (481 + CLS = 482 tokens) and squeezes all 64 electrodes into each token
  through one linear layer. Attention therefore spends its capacity relating
  time points: a 482x482 map per head, 232k entries, x4 heads x3 layers. Yet
  the class information here is a *spatial* pattern of band power (medial Cz
  vs lateral C3/C4, plus the frontal eye signal from Part 1). The vanilla
  model has to discover that pattern through a single projection, from 8
  subjects, with no notion of which electrodes are neighbours.
- *The bias.* (1) A temporal filter bank shared by every electrode
  (16 x 65-tap filters, about 0.4 s) turns each electrode's 3 s signal into
  16 log band-power features. Power is what ERD changes, and it does not
  depend on exact timing, which varies across people. (2) Each electrode
  becomes one token, and its positional encoding is an MLP of its real
  (x, y, z) scalp coordinates (`get_channel_positions()`), so attention can
  learn "front vs back" and "midline vs lateral" as geometry rather than as
  64 arbitrary IDs. (3) Two transformer layers attend across the 64
  electrodes; a CLS token reads out.
- *What it costs and buys.* Sequence length 65 instead of 482, so the
  attention map is 65^2 = 4,225 entries per head, 55x smaller. Parameters:
  20,050 vs 135,746 (EEGNet: 2,738). The compute moves from attention into the
  filter bank (64 x 16 x 481 x 65, about 32M multiply-adds per trial), which
  is why it trains no faster on CPU.
- *What I kept honest.* `models.py`, `train.py` and `data.py` are unchanged.
  EEGNet runs with the Part 1 configuration (kern=80, dropout 0.5) across the
  same seeds. Every model prints both the harness's peak test accuracy (picked
  on test, optimistic: the Part 1 bug 5 bias) and the final-epoch accuracy
  (no selection). Final-epoch is the number to trust.

**Results (best / worst / mean across seeds): PARTIAL RUN.**
The full `python main.py` run (40 epochs, seeds 0-2) needed about an hour on
this laptop CPU, far more than the starter's estimate. I stopped it at the
deadline, 52 minutes in, during the vanilla transformer's third seed.
`YourModel` was never reached. Everything that finished is below, unedited;
the console log is in `part2_transformer/run_log.txt`.

| Model | Params | Seeds finished | Peak test acc (best / worst / mean) | Final-epoch test acc (best / worst / mean) |
| --- | --- | --- | --- | --- |
| EEGNet (Part 1 config) | 2,738 | 3 of 3 | 62.2 / 58.9 / 61.1% | 61.1 / 54.4 / 58.1% |
| Vanilla transformer | 135,746 | 2 of 3 | 60.0 / 58.9 / 59.4% | 51.1 / 50.0 / 50.6% |
| YourModel (electrode tokens) | 20,050 | 0 of 3 | not reached | not reached |

What the finished runs do show:
- **The vanilla transformer ends at chance.** Its final-epoch accuracy is
  50.0% and 51.1%, while its peak was 59-60% at some epoch. That 9-point
  peak-to-final gap means the peak was selection luck on the test set, not a
  learned, transferable representation. That is exactly the weakness the
  hypothesis targets: 136k parameters and a 482x482 attention map, trained on
  360 trials from 8 people.
- **EEGNet keeps most of its peak.** Final 58.1% vs peak 61.1%, a 3-point gap.
  Its conv priors (temporal filters, then depthwise spatial filters) give a
  representation that partly transfers to unseen subjects 9 and 10, though no
  single seed's final accuracy clears the 60% (54/90) chance threshold.
- **The hypothesis is untested.** Whether electrode tokens with 3D position
  encoding close the gap between the vanilla model (about 50%) and EEGNet
  (about 58%) is the open question. The next step is simply to finish the
  run: `python main.py` reproduces everything above and then trains
  `YourModel`.

**Figure:**
Not produced. `main.py` draws the t-SNE of `YourModel`'s test embeddings only
after all models finish, which this run never reached. The `figure.png` left
over in the folder came from a 2-epoch smoke test, so it is not submitted.

**How to read these numbers (caveats):**
- **Final-epoch accuracy is the trusted number.** The starter's `train_model`
  returns the *peak* test accuracy over 40 epochs, which is model selection on
  the test set (the bias fixed in Part 1, bug 5). I did not edit the harness,
  but every model also reports its final-epoch accuracy, which involves no
  selection. The peak is shown only as an optimistic upper bound.
- **Seeds vary initialisation only.** The split is fixed (train subjects 1-8,
  test 9-10), so best/worst across seeds measures training noise, not how
  much the result depends on which people are tested. That dependence was
  large in Part 1 (47-71% across subject pairs).
- **Small test set.** 90 trials from 2 people: one standard error is about
  5 points, and p<0.05 vs chance needs 54/90 (60%). Gaps of a few points
  between models are not evidence either way.
- **The eye confound may persist.** Part 2's 7-30 Hz filter removes most of the
  slow vertical-EOG shift found in Part 1, but saccades also produce sharp,
  broadband spike potentials. I have not measured how much eye signal remains,
  so any model's accuracy is an upper bound on motor-imagery decoding.
- **Environment.** Runs used `KMP_DUPLICATE_LIB_OK=TRUE` to work around an
  OpenMP clash on this Windows/conda setup (see the Part 2 README). It does
  not change the maths, but it is recorded for reproducibility.

**Notes on the attention design:**
- *Scaling by sqrt(d_k).* For random q and k with unit-variance entries, q.k
  has variance d_k, so raw scores grow with head width. Large scores push the
  softmax towards one-hot, where its gradient is almost zero, so training
  stalls and attention collapses onto a single token. Dividing by sqrt(d_k)
  keeps the score variance near 1 for any head size.
- *Sequence length and cost.* Vanilla: 482 tokens, 482^2 = 232,324 entries per
  head per layer, O(T^2 d), about 15M multiply-adds per layer for QK^T alone.
  Electrode tokens: 65 tokens, 4,225 entries, O(C^2 d), independent of trial
  length.
- *With more subjects.* The geometry prior matters most when data is scarce.
  With around 80 subjects the vanilla model could learn spatial structure
  itself, so the gap should shrink. Electrode tokens with log-power features
  would stay cheaper and more interpretable, and cross-subject variability
  (Part 1) does not disappear with more data.
- *Where the model should attend.* CLS-to-electrode attention weights on
  C3/C4 vs Cz/FCz would be neurophysiologically sensible. Weight on
  Fp1/Fpz/Fp2 would mean the model is reusing the vertical eye-movement signal
  found in Part 1, which Part 2's 7-30 Hz filter reduces but does not remove.
  These weights have not been measured yet.
