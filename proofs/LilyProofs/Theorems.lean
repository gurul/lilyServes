import LilyProofs.Model

/-! # Safety invariants of the fusion model -/

namespace Lily

/-! ## 1. A synthetic voice never lowers risk, and never pushes it above 1 -/

/-- `r <= r' <= 1` for `r, d` in `[0, 1]`. `r` (scale `S`) is lifted to scale
`U` by multiplying with `10 * S`. -/
theorem deepfake_never_lowers (r d : Nat) (hr : r ≤ S) (_hd : d ≤ S) :
    r * (10 * S) ≤ amplify r d ∧ amplify r d ≤ U := by
  unfold amplify
  refine ⟨?_, Nat.min_le_left _ _⟩
  apply Nat.le_min.mpr
  constructor
  · have : r * (10 * S) ≤ S * (10 * S) := Nat.mul_le_mul_right _ hr
    simp only [U]; simp only [S] at this ⊢; omega
  · exact Nat.mul_le_mul_left _ (Nat.le_add_right _ _)

/-- The per-sentence risk fed to the EWMA is `<= 1` whatever the score. -/
theorem sentence_risk_le_one (r d : Nat) : amplify r d ≤ U := Nat.min_le_left _ _

/-! ## 2. The EWMA stays in `[0, 1]` -/

/-- One EWMA step is a convex combination: `r' <= 1` and `e <= 1` give `e' <= 1`. -/
theorem ewma_bounded (e : Ewma) (r' : Nat) (hr : r' ≤ U) (he : e.le1) :
    (e.step r').le1 := by
  unfold Ewma.le1 Ewma.step at *
  have h1 : r' * 3 ^ e.exp ≤ U * 3 ^ e.exp := Nat.mul_le_mul_right _ hr
  simp only [Nat.pow_succ]
  rw [← Nat.mul_assoc]
  omega

/-- The `on_llm("scam")` floor keeps `e <= 1` (it raises to 0.6 at most). -/
theorem ewma_floor_bounded (e : Ewma) (he : e.le1) : e.floorAlert.le1 := by
  unfold Ewma.le1 Ewma.floorAlert at *
  have : thetaAlert * 3 ^ e.exp ≤ U * 3 ^ e.exp :=
    Nat.mul_le_mul_right _ (by decide)
  simp only
  omega

/-- The fusion invariant: EWMA in `[0, 1]` and deepfake score in `[0, 1]`. -/
def Inv (f : Fusion) : Prop := f.ewma.le1 ∧ f.deepfake ≤ S

theorem inv_init : Inv ({} : Fusion) := by
  refine ⟨?_, ?_⟩ <;> simp [Ewma.le1]

theorem inv_onSentence (f : Fusion) (h : Heur) (hf : Inv f) : Inv (onSentence f h) := by
  obtain ⟨he, hd⟩ := hf
  refine ⟨?_, hd⟩
  unfold onSentence
  simp only
  split
  · simp [Ewma.le1, sentence_risk_le_one]
  · exact ewma_bounded _ _ (sentence_risk_le_one _ _) he

theorem inv_onLLM (f : Fusion) (v : Verdict) (hf : Inv f) : Inv (onLLM f v) := by
  obtain ⟨he, hd⟩ := hf
  cases v with
  | scam => exact ⟨ewma_floor_bounded _ he, hd⟩
  | uncertain => exact ⟨he, hd⟩
  | safe => exact ⟨he, hd⟩

theorem inv_onDeepfake (f : Fusion) (d : Option Nat) (hf : Inv f) :
    Inv (onDeepfake f d) := by
  cases d with
  | none => exact hf
  | some x => exact ⟨hf.1, Nat.min_le_left _ _⟩

/-- For every call history, `0 <= ewma <= 1` (`0 <=` is free in `Nat`). -/
theorem ewma_bounded_run (es : List Event) : Inv (run {} es) := by
  suffices ∀ f, Inv f → Inv (run f es) from this _ inv_init
  induction es with
  | nil => intro f hf; exact hf
  | cons e es ih =>
    intro f hf
    apply ih
    cases e with
    | sentence h => exact inv_onSentence f h hf
    | llm v => exact inv_onLLM f v hf
    | deepfake d => exact inv_onDeepfake f d hf

/-! ## 3. An LLM "scam" verdict is always a hard alert -/

theorem scam_verdict_hard (f : Fusion) :
    (state (onLLM f .scam)).hard = true ∧ (state (onLLM f .scam)).level = .high := by
  simp [state, onLLM, level, hardAlert]

/-- The floor itself: after `on_llm("scam")`, `ewma >= THETA_ALERT`. -/
theorem scam_floor_holds (f : Fusion) : (onLLM f .scam).ewma.ge thetaAlert = true := by
  simp only [onLLM, ↓reduceIte, Ewma.floorAlert, Ewma.ge, decide_eq_true_eq]
  exact Nat.le_max_right _ _

/-- Why the floor matters: a later "uncertain"/"safe" verdict (with no new
sentence) still leaves a hard alert, because the EWMA keeps the floor. -/
theorem scam_floor_survives_reverdict (f : Fusion) (v : Verdict) :
    (state (onLLM (onLLM f .scam) v)).hard = true := by
  have h := scam_floor_holds f
  cases v <;> simp_all [state, hardAlert, onLLM, Ewma.floorAlert]

/-! ## 4. A lone payment marker is not a hard alert (the false-positive fix) -/

theorem lone_marker_not_hard (f : Fusion)
    (hm : f.markerSeen = true) (hs : f.signals.length = 1)
    (hv : f.verdict = .safe) (he : f.ewma.ge thetaAlert = false) :
    (state f).hard = false := by
  simp [state, hardAlert, corroborated, hm, hs, hv, he]

/-- Same situation without an LLM answer yet: Medium, and an LLM check is requested. -/
theorem lone_marker_medium_needs_llm (f : Fusion)
    (hm : f.markerSeen = true) (hs : f.signals.length = 1)
    (hv : f.verdict = .safe) (he : f.ewma.ge thetaAlert = false)
    (ha : f.llmAfterMarker = false) :
    (state f).level = .medium ∧ (state f).needsLlm = true := by
  simp [state, level, hardAlert, lone, cleared, corroborated, hm, hs, hv, he, ha]

/-! ## 5. The displayed level never goes down -/

theorem maxLevel_ge_left (a b : Level) : a ≤ maxLevel a b := by
  show a.rank ≤ (maxLevel a b).rank
  unfold maxLevel; split <;> omega

theorem maxLevel_ge_right (a b : Level) : b ≤ maxLevel a b := by
  show b.rank ≤ (maxLevel a b).rank
  unfold maxLevel; split <;> omega

theorem foldl_maxLevel_ge (acc : Level) (ls : List Level) :
    acc ≤ ls.foldl maxLevel acc := by
  induction ls generalizing acc with
  | nil => exact Nat.le_refl _
  | cons x xs ih =>
    exact Nat.le_trans (maxLevel_ge_left acc x) (ih (maxLevel acc x))

theorem foldl_maxLevel_ge_mem (ls : List Level) (x acc : Level) (hx : x ∈ ls) :
    x ≤ ls.foldl maxLevel acc := by
  induction ls generalizing acc with
  | nil => cases hx
  | cons y ys ih =>
    cases hx with
    | head => exact Nat.le_trans (maxLevel_ge_right acc x) (foldl_maxLevel_ge _ ys)
    | tail _ h' => exact ih (maxLevel acc y) h'

/-- `max_level(*levels)` is at least every argument. -/
theorem maxLevelN_ge_mem (ls : List Level) (x : Level) (hx : x ∈ ls) :
    x ≤ maxLevelN ls := foldl_maxLevel_ge_mem ls x .low hx

/-- One display update never lowers the shown level and covers every fresh level. -/
theorem displayStep_ge (cur : Level) (fresh : List Level) :
    cur ≤ displayStep cur fresh ∧ ∀ x ∈ fresh, x ≤ displayStep cur fresh :=
  ⟨maxLevelN_ge_mem _ _ (List.mem_cons_self ..),
   fun _ hx => maxLevelN_ge_mem _ _ (List.mem_cons_of_mem _ hx)⟩

/-- For any sequence of fused levels, the displayed sequence is non-decreasing. -/
theorem display_monotone (cur : Level) (us : List (List Level)) :
    Sorted (displaySeq cur us) := by
  induction us generalizing cur with
  | nil => trivial
  | cons u us ih =>
    cases us with
    | nil => exact ⟨(displayStep_ge cur u).1, trivial⟩
    | cons u' us' => exact ⟨(displayStep_ge cur u).1, ih _⟩

/-- Once High is shown, it stays High. -/
theorem display_high_sticky (fresh : List Level) : displayStep .high fresh = .high := by
  have h := (displayStep_ge .high fresh).1
  revert h
  generalize displayStep .high fresh = l
  cases l <;> decide

/-! ## 6. A corroborated payment marker is a hard alert -/

theorem corroborated_marker_hard (f : Fusion)
    (hm : f.markerSeen = true) (hs : 2 ≤ f.signals.length) :
    (state f).hard = true ∧ (state f).level = .high := by
  simp [state, level, hardAlert, corroborated, hm, hs]

end Lily
