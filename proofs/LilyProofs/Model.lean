/-!
# Exact model of `lily/detection/fusion.py` and `heuristics.max_level`

Number representation (all arithmetic is exact, no rounding anywhere):

* Unit-interval inputs `r` (transcript risk) and `d` (deepfake score) are
  naturals on the scale `S = 10000` (basis points). Python's
  `r = score / 10` with an integer `score` is exact on this scale
  (`score * 1000`), and so are `VERDICT_RISK` (0.9, 0.4, 0) and the
  thresholds 0.35 and 0.60.
* The amplified risk `r' = min(1, r * (1 + 0.6 d))` is a natural on the
  finer scale `U = 10 * S * S`: `r/S * (10 S + 6 d)/(10 S) = r (10 S + 6 d) / U`.
  This is exact for every `r, d` on the `S` grid.
* EWMA with `phi = 1/3` is not closed on any fixed grid, so the EWMA is a
  pair `(num, exp)` meaning the rational `num / (U * 3^exp)`. One update
  `phi r' + (1 - phi) e` maps `(n, k)` to `(r' * 3^k + 2 n, k + 1)`, which is
  exact. A threshold test `e >= theta` (theta on scale `U`) is
  `n >= theta * 3^k`.
-/

namespace Lily

/-- Basis-point scale for unit-interval inputs. -/
def S : Nat := 10000
/-- Scale of the amplified risk `r'` and of the EWMA numerator at `exp = 0`. -/
def U : Nat := 10 * S * S

/-- `THETA_ALERT = 0.60` on scale `U`. -/
def thetaAlert : Nat := 6 * U / 10
/-- `THETA_WARN = 0.35` on scale `U`. -/
def thetaWarn : Nat := 35 * U / 100

/-- `fusion.py: DEEPFAKE_GAIN = 0.6`, as the numerator over 10. -/
def gainTenths : Nat := 6

inductive Verdict | scam | uncertain | safe
  deriving DecidableEq, Repr

/-- `fusion.py: _VERDICT_RISK`, on scale `S`. -/
def verdictRisk : Verdict → Nat
  | .scam => 9000
  | .uncertain => 4000
  | .safe => 0

/-- `fusion.py: PAYMENT_STAGE_SIGNALS`. -/
def paymentSignals : List String :=
  ["gift cards", "wire transfer", "crypto payment", "verification code",
   "remote access", "secrecy", "courier/cash pickup", "payment app to self",
   "protect-your-money"]

/-- `heuristics.HeuristicResult` restricted to what fusion reads. -/
structure Heur where
  score : Nat
  signals : List String
  deriving Repr

/-- `r = min(1, max(0, score / 10))` on scale `S` (exact for integer scores). -/
def baseRisk (score : Nat) : Nat := min S (score * (S / 10))

/-- `r = max(r, VERDICT_RISK[v])`, scale `S`. -/
def verdictFloor (r : Nat) (v : Verdict) : Nat := max r (verdictRisk v)

/-- `r' = min(1, r * (1 + DEEPFAKE_GAIN * d))`; input scale `S`, output scale `U`. -/
def amplify (r d : Nat) : Nat := min U (r * (10 * S + gainTenths * d))

/-- An exact EWMA value `num / (U * 3^exp)`. -/
structure Ewma where
  num : Nat
  exp : Nat
  deriving Repr, DecidableEq

/-- `e >= theta` for `theta` on scale `U`. -/
def Ewma.ge (e : Ewma) (theta : Nat) : Bool := decide (theta * 3 ^ e.exp ≤ e.num)

/-- `e <= 1`. -/
def Ewma.le1 (e : Ewma) : Prop := e.num ≤ U * 3 ^ e.exp

/-- `e_t = phi r' + (1 - phi) e_{t-1}` with `phi = 1/3`, exactly. -/
def Ewma.step (e : Ewma) (r' : Nat) : Ewma :=
  { num := r' * 3 ^ e.exp + 2 * e.num, exp := e.exp + 1 }

/-- `ewma = max(ewma, THETA_ALERT)`. -/
def Ewma.floorAlert (e : Ewma) : Ewma :=
  { e with num := max e.num (thetaAlert * 3 ^ e.exp) }

/-- `fusion.py: RiskFusion` fields. `signals` is the set `_signals_seen`,
kept duplicate-free by `addSignal`. -/
structure Fusion where
  sentences : Nat := 0
  ewma : Ewma := ⟨0, 0⟩
  verdict : Verdict := .safe
  deepfake : Nat := 0
  markerSeen : Bool := false
  llmAfterMarker : Bool := false
  signals : List String := []
  deriving Repr

def addSignal (l : List String) (s : String) : List String :=
  if s ∈ l then l else l ++ [s]

/-- `RiskFusion.on_sentence`. -/
def onSentence (f : Fusion) (h : Heur) : Fusion :=
  let sentences := f.sentences + 1
  let signals := h.signals.foldl addSignal f.signals
  let hit := h.signals.any (fun s => s ∈ paymentSignals)
  let (marker, after) :=
    if !f.markerSeen && hit then (true, false) else (f.markerSeen, f.llmAfterMarker)
  let r := verdictFloor (baseRisk h.score) f.verdict
  let r' := amplify r f.deepfake
  let ewma := if sentences = 1 then ⟨r', 0⟩ else f.ewma.step r'
  { f with sentences, signals, markerSeen := marker, llmAfterMarker := after, ewma }

/-- `RiskFusion.on_llm` (unknown verdict strings are a no-op in Python and are
not representable here). -/
def onLLM (f : Fusion) (v : Verdict) : Fusion :=
  let f1 := { f with verdict := v,
                     llmAfterMarker := if f.markerSeen then true else f.llmAfterMarker }
  if v = .scam then { f1 with ewma := f1.ewma.floorAlert } else f1

/-- `RiskFusion.on_deepfake`; the score is on scale `S`, clamped to `[0, S]`. -/
def onDeepfake (f : Fusion) (d : Option Nat) : Fusion :=
  match d with
  | some x => { f with deepfake := min S x }
  | none => f

inductive Level | low | medium | high
  deriving DecidableEq, Repr

inductive Stage | contact | engagement | payment
  deriving DecidableEq, Repr

structure FusionState where
  level : Level
  ewma : Ewma
  stage : Stage
  hard : Bool
  verdict : Verdict
  needsLlm : Bool
  deriving Repr

def corroborated (f : Fusion) : Bool := f.markerSeen && decide (2 ≤ f.signals.length)
def lone (f : Fusion) : Bool := f.markerSeen && !corroborated f
def cleared (f : Fusion) : Bool :=
  lone f && f.llmAfterMarker && decide (f.verdict = .safe)

def hardAlert (f : Fusion) : Bool :=
  corroborated f || f.ewma.ge thetaAlert || decide (f.verdict = .scam)

def level (f : Fusion) : Level :=
  if hardAlert f then .high
  else if f.ewma.ge thetaWarn
        || (decide (f.verdict = .uncertain) && !f.signals.isEmpty)
        || (lone f && !cleared f) then .medium
  else .low

def stage (f : Fusion) : Stage :=
  if f.markerSeen && !cleared f then .payment
  else if !f.signals.isEmpty || decide (3 < f.sentences) then .engagement
  else .contact

/-- `RiskFusion.state()`: a pure function of the fields. -/
def state (f : Fusion) : FusionState :=
  { level := level f, ewma := f.ewma, stage := stage f, hard := hardAlert f,
    verdict := f.verdict, needsLlm := lone f && !f.llmAfterMarker }

/-- Inputs the call loop can feed into one `RiskFusion`. -/
inductive Event
  | sentence (h : Heur)
  | llm (v : Verdict)
  | deepfake (d : Option Nat)

def Fusion.apply (f : Fusion) : Event → Fusion
  | .sentence h => onSentence f h
  | .llm v => onLLM f v
  | .deepfake d => onDeepfake f d

def run (f : Fusion) (es : List Event) : Fusion := es.foldl Fusion.apply f

/-! ## Display level (`heuristics.max_level`, `app._on_final_line`, `app._refine_scam`) -/

def Level.rank : Level → Nat
  | .low => 0 | .medium => 1 | .high => 2

instance : LE Level := ⟨fun a b => a.rank ≤ b.rank⟩
instance (a b : Level) : Decidable (a ≤ b) := inferInstanceAs (Decidable (a.rank ≤ b.rank))

/-- Two-argument `max_level`: the more severe level. -/
def maxLevel (a b : Level) : Level := if a.rank ≤ b.rank then b else a

/-- Variadic `max_level(*levels)`: Python starts at `best = 0` (Low). -/
def maxLevelN (ls : List Level) : Level := ls.foldl maxLevel .low

/-- One display update. `_on_final_line` passes
`[scam_level, heur.level, state.level]`; `_refine_scam` passes
`[scam_level, state.level]`. Both have the shape `scam_level :: fresh`. -/
def displayStep (cur : Level) (fresh : List Level) : Level := maxLevelN (cur :: fresh)

/-- The displayed sequence: the initial `Low`, then one entry per update. -/
def displaySeq (cur : Level) : List (List Level) → List Level
  | [] => [cur]
  | u :: us => cur :: displaySeq (displayStep cur u) us

/-- Non-decreasing in `≤`. -/
def Sorted : List Level → Prop
  | [] => True
  | [_] => True
  | a :: b :: t => a ≤ b ∧ Sorted (b :: t)

end Lily
