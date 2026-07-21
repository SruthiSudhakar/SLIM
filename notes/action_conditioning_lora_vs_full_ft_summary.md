# Ctrl-World action-conditioning experiments — summary of changes & findings

Branch: `action-conditioning-improvements`. Goal: strengthen how the SVD-based Ctrl-World world
model conditions on robot action trajectories (inspired by a HunyuanVideo-1.5 DiT paper).

**TL;DR: A/B/C are all marginal-to-null. The per-frame-MLP baseline already conditions about as
well as any addition on this data — the task is not action-conditioning-bottlenecked. All changes
are flag-gated and OFF by default, so nothing in the default model changed. Investigation CLOSED.**

---

## Changes implemented (all flag-gated, default OFF)

| change | what | flag (env) | new params |
|---|---|---|---|
| **A** — temporal action encoder | 4-layer transformer over the per-frame action tokens (trajectory-aware); residual + zero-init → no-op at start | `USE_TEMPORAL_ACTION_ENCODER=1` | ~35 M |
| **B** — action modulation | project action tokens (zero-init) and add onto the timestep embedding `emb` (a global bias into every ResNet block) | `USE_ACTION_MODULATION=1` | ~1.3 M |
| **C** — full temporal action context | monkeypatch the temporal cross-attn to attend over ALL F action tokens instead of stock `[:,0]` (frame-0 only) | `USE_TEMPORAL_ACTION_COND=1` | 0 (reuses pretrained attn2) |

Note: C adds no weights → NOT checkpoint-detectable → its flag must be set at **both train and eval**.

## Methodology / infra fixes (kept regardless of A/B/C)
- **Seeded generation** in rollout eval (`--gen_seed`) — the diffusion sampling was unseeded, giving ±0.6 dB run-to-run noise that flipped results. Now baseline vs variant use identical noise (paired).
- **Step-limit bug** in `train_wm.py` — epoch formula multiplied by `total_batch_size` with no hard stop → ran ~8× past `MAX_TRAIN_STEPS`. Fixed formula + hard `break`.
- **PSNR/MSE drift metrics** written to `rollout_metrics.json`; **grad-norm & LR logging** to wandb; **eval auto-detect** of architecture from checkpoint keys; `meta.json` records `use_temporal_action_cond`.
- **Param-group LR** knob (`ACTION_ENCODER_LR`) — see finding below (boosting it HURTS).

---

## Findings (all seeded, paired by episode)

### 1. Open-drawer (494 train / 20 val — small, overfitting regime), Change A, base LR
| step | baseline PSNR | A PSNR | Δ(A−base) | p | MSE |
|---|---|---|---|---|---|
| 1500 | 18.21 | 17.80 | −0.41 | 0.25 | +4.8% |
| 3500 | 18.71 | 18.65 | −0.05 | 0.81 | +1.9% |
| 4000 | 18.40 | **19.11** | **+0.71** | **0.019** | **−18%** |

A wins at 4000 (16/20 episodes) — but **because the baseline OVERFITS** (its val drops 3500→4000: 18.71→18.40) while A keeps improving. So A = regularizer, not a ceiling gain.

### 2. LR sweep (open-drawer): boosting `ACTION_ENCODER_LR` significantly HURTS
Monotonic: higher LR → lower val PSNR. All boosted LRs significantly worse than base 1e-5.
| ACTION_ENCODER_LR vs base(1e-5) @1000 | Δ | p |
|---|---|---|
| 1e-4 | −0.44 | 0.15 |
| 3e-4 | −0.82 | 0.001 |
| 1e-3 | −1.59 | <0.001 |
Reason: the new module injects into a WARM UNet; a big LR dumps signal faster than the UNet can adapt → destabilizes. **Use base LR 1e-5.** (The deleted `big_changeA_lr1e4` big-dataset run confirmed this: worse than baseline at 10k, gr00t −1.16 dB p=0.0004.)

### 3. Big dataset (atomic_all + gr00t_rollouts, ~1M windows — no overfitting), base LR, n=36

Mean PSNR ↑ / MSE ↓ (`*` = recovered/unseeded baseline@5k):

**atomic_all**
| step | baseline | A | B | A+C |
|---|---|---|---|---|
| 5000 | 17.95/1622* | 17.85/1647 | 17.77/1715 | 17.64/1720 |
| 10000 | 18.12/1564 | 18.38/1447 | 18.28/1595 | 18.34/1467 |
| 25000 | 18.77/1403 | 18.54/1465 | 18.94/1277 | **19.00/1367** |

**gr00t**
| step | baseline | A | B | A+C |
|---|---|---|---|---|
| 5000 | 18.98/1131* | 18.79/1405 | 18.49/1319 | 18.44/1413 |
| 10000 | 19.50/1160 | **20.03/935** | 19.45/1097 | 19.36/1035 |
| 25000 | **20.27/863** | 19.89/956 | 20.06/912 | 20.00/991 |

Paired Δ PSNR vs baseline (p in parens):
| | A | B | A+C |
|---|---|---|---|
| 10k atomic | +0.26 (0.07) | +0.16 (0.32) | +0.22 (0.06) |
| 10k gr00t | **+0.53 (0.07)** | −0.05 (0.83) | −0.14 (0.67) |
| 25k atomic | −0.23 (0.18) | +0.17 (0.32) | +0.23 (0.15) |
| 25k gr00t | −0.38 (**0.02**) | −0.20 (0.17) | −0.27 (0.29) |

Reading:
- **A = faster convergence, lower ceiling.** Ahead at 10k (gr00t +0.53), but plateaus and **baseline overtakes by 25k** (gr00t −0.38, p=0.02 — the only p<0.05 cell, and it's baseline winning).
- **B = neutral early, faint late lean** (best atomic@25k, MSE −9% but PSNR p=0.32). Nothing significant.
- **A+C = null; C did not help A** — 10k/gr00t A was +0.53 but A+C is −0.14 (C erased A's best result). Best A+C cell +0.23 (p=0.15). On atomic A+C keeps climbing (17.64→18.34→19.00, top at 25k) but a wash on gr00t.

---

## Conclusion
- **No config beats baseline significantly** on the big dataset; the only significant big-data cell is **baseline beating A** at 25k/gr00t. At 25k the whole family clusters within ~0.4 dB.
- **A** has a real but narrow niche: **low-data / short-step-budget** (open-drawer +0.71 dB p=0.019), acting as a regularizer against overfitting — not a general improvement.
- **B and C add essentially nothing**; C slightly hurts A.
- **The task is not action-conditioning-bottlenecked** — the baseline per-frame MLP is sufficient. Bottleneck is likely elsewhere (data / base model / training length). Effort pivoted to a few-shot / LoRA new-concept direction.
- All A/B/C flags default OFF; the shipped default model is unchanged.
