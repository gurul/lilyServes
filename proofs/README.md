# Lean proofs of the fusion invariants

This directory is a Lean 4 (v4.34.1, core only, no Mathlib) model of
`lily/detection/fusion.py`, `heuristics.max_level`, and the displayed-level
update in `lily/app.py`. It proves the safety invariants that the fusion
docstring states, with no `sorry` and no new axioms. A synthetic voice never
lowers transcript risk and never raises it above 1. The EWMA stays in [0, 1]
for every call history. An LLM "scam" verdict always gives `hard_alert` and
High, and its 0.60 floor survives a later "safe" or "uncertain" verdict. A
lone payment marker with a safe verdict and EWMA below 0.60 is not a hard
alert. A corroborated marker is always a hard alert. The level shown to the
user never goes down during a call. Python test vectors from
`tests/test_detection_fusion_rules.py` are replayed as `#guard` checks, so the
model is tied to the code and not only to the docstring.

## Check

```sh
cd proofs && ~/.elan/bin/lake build
```

The build fails if a proof or a `#guard` fails. `LilyProofs/Axioms.lean`
prints the axioms of each theorem during the build. The expected set is a
subset of `propext`, `Quot.sound`, `Classical.choice`; `sorryAx` must not appear.

## Number representation

All arithmetic is exact. There is no rounding in the model.

| Quantity | Lean representation | Why it is exact |
|---|---|---|
| `r`, `d`, `VERDICT_RISK` | `Nat` on scale `S = 10000` | `score` is an `int`, so `score/10` is `score * 1000`; 0.9, 0.4, 0 are exact |
| `r' = min(1, r(1 + 0.6 d))` | `Nat` on scale `U = 10 S^2` | `r(10S + 6d) / (10 S^2)` has no remainder |
| EWMA, `phi = 1/3` | `Ewma = (num, exp)`, value `num / (U * 3^exp)` | one step maps `(n, k)` to `(r' 3^k + 2n, k+1)` |
| `THETA_ALERT`, `THETA_WARN` | `0.60 U`, `0.35 U` | test `e >= theta` is `theta * 3^exp <= num` |

## Correspondence

| Lean (`LilyProofs/Model.lean`) | Python |
|---|---|
| `S`, `U`, `thetaAlert`, `thetaWarn`, `gainTenths` | `fusion.py`: `THETA_ALERT`, `THETA_WARN`, `DEEPFAKE_GAIN` |
| `verdictRisk`, `Verdict` | `fusion.py`: `_VERDICT_RISK` |
| `paymentSignals` | `fusion.py`: `PAYMENT_STAGE_SIGNALS` |
| `Heur` | `heuristics.py`: `HeuristicResult` (score, signals) |
| `baseRisk`, `verdictFloor`, `amplify` | `fusion.py`: `RiskFusion.on_sentence`, the three `risk = ...` lines |
| `Ewma.step` | `fusion.py`: `RiskFusion.on_sentence`, `ewma = PHI * risk + (1 - PHI) * ewma` |
| `Ewma.floorAlert` | `fusion.py`: `RiskFusion.on_llm`, `ewma = max(ewma, THETA_ALERT)` |
| `Fusion` | `fusion.py`: `RiskFusion` fields |
| `onSentence`, `onLLM`, `onDeepfake` | `fusion.py`: `RiskFusion.on_sentence`, `.on_llm`, `.on_deepfake` |
| `corroborated`, `lone`, `cleared` | `fusion.py`: `_corroborated`, `_lone_marker`, `_marker_cleared` |
| `hardAlert`, `level`, `stage`, `state` | `fusion.py`: `RiskFusion.state`, `RiskFusion.stage` |
| `maxLevel`, `maxLevelN` | `heuristics.py`: `max_level` |
| `displayStep`, `displaySeq` | `app.py`: `_on_final_line` and `_refine_scam`, `session.scam_level = max_level(...)` |

## Theorems (`LilyProofs/Theorems.lean`)

| Theorem | Statement |
|---|---|
| `deepfake_never_lowers` | `r <= S -> d <= S -> r * 10S <= amplify r d /\ amplify r d <= U` |
| `ewma_bounded` | `r' <= U -> e <= 1 -> step e r' <= 1` |
| `ewma_bounded_run` | for every event list, `run {} es` has `ewma <= 1` and `deepfake <= 1` |
| `scam_verdict_hard` | for every state, `on_llm(scam)` gives `hard = true` and level High |
| `scam_floor_holds` | after `on_llm(scam)`, `ewma >= THETA_ALERT` |
| `scam_floor_survives_reverdict` | `on_llm(scam)` then `on_llm(v)` for any `v` is still hard |
| `lone_marker_not_hard` | marker, one signal, verdict safe, `ewma < 0.60` -> `hard = false` |
| `lone_marker_medium_needs_llm` | same, before any LLM answer -> Medium and `needs_llm` |
| `maxLevel_ge_left/right`, `maxLevelN_ge_mem` | `max_level` is at least each argument |
| `display_monotone` | the displayed level sequence is non-decreasing for any input sequence |
| `display_high_sticky` | once High is shown, every later update shows High |
| `corroborated_marker_hard` | marker and `|signals| >= 2` -> hard and High, for any verdict and EWMA |

## Modelling gaps

- **Floats compared with exact values.** Python uses IEEE-754 doubles. The
  model uses exact rationals. They can disagree exactly at a threshold. A
  fuzz of 200,000 random call histories found one: deepfake 0.6, then
  sentences with scores 1, 9, 7 and no signals. The exact EWMA is
  0.136, 0.424, 0.6, so the model gives `hard_alert = True` and High. Python
  computes `0.5999999999999999` and returns `hard_alert = False` and Medium.
  `Vectors.lean` (`floatGap`) records this trace. `fusion.py` now compares
  against each threshold with a 1e-9 slack (`_reaches`), and
  `tests/test_detection_fusion_rules.py::test_threshold_reached_despite_float_rounding`
  replays the trace. The remaining gap is on the safe side: Python can alert
  when the exact EWMA is up to 1e-9 below a threshold. The bound
  `0 <= ewma <= 1` held in every fuzz run.
- **Deepfake score quantization.** The model takes `d` in basis points.
  Python takes any float in [0, 1]. The theorems hold for every `d` on that
  grid.
- **Heuristic scoring is not modelled.** `score_text` (regex weights,
  combination rules, `heur.level`) is an input. The model takes `score : Nat`
  and the signal list as given. `display_monotone` covers `heur.level` because
  it holds for any input level sequence.
- **Unknown verdict strings.** Python ignores an `on_llm` verdict outside
  scam/uncertain/safe. The Lean `Verdict` type has only the three values.
- **`state().score` rounding.** Python reports `round(ewma, 4)`. The model
  exposes the exact EWMA; no decision reads the rounded value.
- **Concurrency.** `_refine_scam` runs as an async task. The model treats
  each update as atomic, which matches the single-threaded event loop.
- **Monotonic display per session only.** `display_monotone` is about one
  `session.scam_level` sequence. The end-of-call summary
  (`max_level(session.scam_level, summary risk_level)`) is also a `max_level`
  and so also does not go down, but it is not a separate theorem.
