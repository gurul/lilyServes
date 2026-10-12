import LilyProofs.Theorems

/-! Axiom audit: each theorem must depend on at most
`propext`, `Quot.sound`, `Classical.choice`, and never `sorryAx`. -/

open Lily in
section
#print axioms deepfake_never_lowers
#print axioms ewma_bounded
#print axioms ewma_floor_bounded
#print axioms ewma_bounded_run
#print axioms scam_verdict_hard
#print axioms scam_floor_holds
#print axioms scam_floor_survives_reverdict
#print axioms lone_marker_not_hard
#print axioms lone_marker_medium_needs_llm
#print axioms maxLevel_ge_left
#print axioms maxLevel_ge_right
#print axioms maxLevelN_ge_mem
#print axioms display_monotone
#print axioms display_high_sticky
#print axioms corroborated_marker_hard
end
