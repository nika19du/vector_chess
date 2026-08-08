# VectorChess Interactive Application

This document is the architecture design for Milestone 5 (Interactive UI) and the
real-time-facing half of Milestone 4b (Audio Layer 2). It is a design document, not an
implementation plan and not a mathematics document — see `docs/mathematics.md` for the
math and `docs/audio.md` for the sonification design this document extends. No
mathematics changes here: every visual and sonic element below is sourced from an
already-implemented, already-tested object in `analysis/`, `chess_engine/models.py`, or
the sonification rules already ratified in `docs/audio.md`.

**Status: architecture frozen (v3) — ready for implementation.** See "Final Self-Review"
at the end of this document.

**Revision history.**
- **v1** — initial design across ten parts (philosophy, layout, board, visualization,
  animation, music, interaction, visual identity, technology, roadmap).
- **v2** — incorporated a senior-architect review of v1. Added the software-architecture
  layer that was entirely missing (SessionState, position caching, cross-move identity,
  layer registration, thread-safe audio hand-off), fixed several internal
  inconsistencies, added performance budgets and accessibility. See "Review Disposition
  — v1 → v2" below.
- **v3 (this revision)** — a final architecture-hardening pass, conducted specifically
  before any implementation starts. Closes the one BLOCKER and five HIGH findings from
  the v2 review, and a further eight MEDIUM findings that improve long-term
  maintainability without adding unnecessary mechanism. See "Hardening Pass Disposition
  — v2 → v3" below. Nothing in v3 touches the mathematical pipeline, `docs/mathematics.md`,
  `docs/audio.md`, the application philosophy, or the visual identity — all remain
  exactly as v2 (and, for the first two, as they always were) left them.

## Where this starts from

Today the only interface is `console_app/main.py`, a blocking REPL: type a move, type a
command (`plot`, `gradient_plot`, `morse_smale_plot`, `audio`, …), get one static
matplotlib window or one `.wav` file. Nothing is live, nothing animates, nothing plays
back. This document designs the application that replaces that workflow with a real-time
desktop instrument, without touching the math that produces the data.

Grounding used throughout:
- **Stack today**: `chess==1.11.2`, `matplotlib==3.11.1`, `numpy==2.5.1`, `scipy==1.18.0`,
  `pytest==9.1.1`. No GUI toolkit, no audio-playback library — audio is numpy → stdlib
  `wave` → `.wav` file, never played back live.
- **`analysis/`** builds every field on demand (bicubic `RectBivariateSpline`, refit from
  scratch every call — no caching) and hands back plain `@dataclass` objects from
  `chess_engine/models.py`. **`visualization/`** is one matplotlib module per concept,
  every one ending in a blocking `plt.show()`.
- **`docs/audio.md`** is a complete, already-approved sonification design. Part 7 below
  extends it into a live/continuous context; it does not re-decide anything it already
  settled, and this document is deliberately conservative about that boundary throughout
  (see both Review Disposition sections' rejected/deferred items).
- **`docs/mathematics.md`** Sections 9–11 define Critical Points, Ridge/Valley, and the
  Morse-Smale Complex precisely, including their termination/rejection vocabulary — this
  document's correspondence design (Part 4.5) is built to respect that vocabulary
  exactly, never to extend or reinterpret it.
- **`ui_demo.html` / `ui-imgs/prototip0.png`** is the existing hand-built mockup (dark navy
  `#0f172a`, panel `#1e293b`, border `#334155`, White `#38bdf8`, Black `#f43f5e`, tension
  gold `#eab308`, dashed control-triangle, SVG coordinate grid, a balance bar). Unchanged
  as the seed aesthetic across every revision.
- **`python-chess`** already models a branching game tree (`chess.pgn.GameNode`,
  `.variations`, `.add_variation()`), which turns out to already solve most of the
  "timeline vs. branching" problem raised in the v1→v2 review — see Part 4.4.

---

## Review Disposition — v1 → v2

| # | Review finding | Verdict | Where addressed |
|---|---|---|---|
| 1 | No single state-ownership architecture | **Accepted** | Part 4 — SessionState |
| 2 | No cross-panel data-ownership rules | **Accepted** | Part 4 — Data Ownership |
| 3 | Critical points / chains / cells have no identity across moves; interpolation and Compare-mode diffing both assume a correspondence that doesn't exist | **Accepted** | Part 4 — Stable Identity |
| 4 | Cache keyed by ply index is wrong once undo/branching exists | **Accepted** | Part 4 — Position Cache |
| 5 | Timeline model assumes a single line; chess analysis is a tree | **Accepted** | Part 4 — Game Tree; Part 8 |
| 6 | Exporting by capturing the live real-time loop risks A/V desync | **Accepted** | Part 8, Part 10 — Offline Export |
| 7 | No thread-safety design between the UI thread and the real-time audio callback | **Accepted** | Part 4 — Audio Thread Boundary |
| 8 | Layer stack is a hardcoded 7-tuple, not extensible | **Accepted** | Part 5 — Layer Registry; Part 12 |
| 9 | No performance/overdraw budget for simultaneous layers | **Accepted** | Part 11 |
| 10 | No keyboard shortcuts specified | **Accepted** | Part 8 |
| 11 | No canvas zoom/pan | **Accepted** | Part 8 |
| 12 | Dominance encoded by hue alone; no accessibility path | **Accepted** | Part 9 — Accessibility |
| 13 | Top-bar Freeze button contradicts Part 7's two Freeze modes | **Accepted** | Part 2, Part 8 |
| 14 | Milestone 4b hard-blocks all of Milestone 5, but only live playback needs it | **Accepted** | Part 13 |
| 15 | Video/MIDI export bundled into the core UI milestone inflates its scope | **Accepted** | Part 13 |
| 16 | "Solo" vocabulary claims a 1:1 layer↔voice mapping that doesn't exist (7 layers, 5 voices) | **Rejected as stated** | See below |
| 17 | Source Potential and Ridge/Valley both claim stereo pan, colliding | **Deferred, not fixed here** | See below |

**Rejected (#16): forcing a 1:1 mapping between visual layers and audio voices.**
Restructuring the audio engine to have exactly one voice per visual layer would mean
either fragmenting `docs/audio.md`'s already-justified voice roles to hit a count, or
forcing future visual layers to be added only in lockstep with new audio voices — both
violate this document's instruction to reuse the existing philosophy rather than
redesign it. The fix that survives is documentation accuracy (layer-solo and voice-solo
are independently scoped controls that share an interaction convention, not a
structural mapping), not a structural change. **v3 note:** the new Voice Registry (Part
5) is deliberately designed not to reopen this — it changes how voices are discovered,
not how many exist or what they correspond to.

**Deferred (#17): the Source Potential / Ridge-Valley stereo-pan collision.**
This is a decision about what a specific mathematical object should sound like, which is
`docs/audio.md`'s domain, not this document's. `docs/audio.md` §5 explicitly gates every
Layer 2 mapping (including both of the colliding ones) behind its own milestone and its
own listener-legibility validation. It is recorded here so it isn't lost, flagged as an
open input to whoever scopes the Audio Layer 2 milestone in detail. **v3 note:** still
not resolved here, unchanged from v2.

---

## Hardening Pass Disposition — v2 → v3

This is the final architecture-hardening pass, conducted before any implementation
starts. Every BLOCKER and HIGH finding from the v2 review is accepted. MEDIUM findings
are accepted only where they improve long-term maintainability without adding
unnecessary mechanism; two are deferred as out of this document's scope. LOW findings
are deferred as not load-bearing for a first implementation.

| # | Finding | Severity | Verdict | Where addressed |
|---|---|---|---|---|
| B1 | `SessionState` update-propagation mechanism unspecified | BLOCKER | **Accepted** | Part 4.1 |
| H1 | Position Cache has no concurrency/async model; a cache miss risks stalling the frame budget | HIGH | **Accepted** | Part 4.3 |
| H2 | Export driver's read of a live-mutating `SessionState`/game tree is unspecified | HIGH | **Accepted** | Part 4.7 |
| H3 | No stated audio latency budget, despite Part 1 promising audio-visual synchrony | HIGH | **Accepted** | Part 4.6 |
| H4 | Audio voices remain a hardcoded set; no registry symmetric to the Layer Registry | HIGH | **Accepted** | Part 5 |
| H5 | No concurrency/race-testing strategy for the lock-free snapshot mechanism | HIGH | **Accepted** | Part 14 |
| M1 | Correspondence matching recomputed every animation frame instead of once per position pair | MEDIUM | **Accepted** | Part 4.5 |
| M2 | Cache sizing (200 entries) lacks a stated worst-case justification | MEDIUM | **Accepted** | Part 4.3 |
| M3 | Qt event loop vs. VisPy/OpenGL render loop integration boundary unstated | MEDIUM | **Deferred** | Part 10 |
| M4 | Audio engine's handling of rapid successive snapshot updates (zipper noise) unaddressed | MEDIUM | **Deferred** | — |
| M5 | Live vs. exported audio identity guarantee ambiguous | MEDIUM | **Accepted** | Part 7 |
| M6 | `LayerDefinition.animator` implied a closed set of exactly two shapes | MEDIUM | **Accepted** | Part 5 |
| M7 | GPU-side buffer caching not addressed alongside the CPU-side Position Cache | MEDIUM | **Accepted** | Part 11 |
| M8 | No cross-phase end-to-end integration test named | MEDIUM | **Accepted** | Part 14 |
| M9 | Performance test only gated once, at phase 5a | MEDIUM | **Accepted** | Part 11, Part 14 |
| M10 | Hardcoded "200×200" resolution assumption scattered through the document | MEDIUM | **Accepted** | Part 4.3, Part 11 |
| L1 | `canvas_id` not formally defined as a first-class identifier | LOW | **Deferred** | — |
| L2 | Audio→UI position hand-off wording under-specified | LOW | **Deferred** | — |
| L3 | Unconditional-redraw rationale undocumented | LOW | **Deferred** | — |

**Deferred (M3): Qt event loop vs. VisPy/OpenGL render loop integration.** A well-known,
low-risk implementation pattern (a `QTimer`-driven `paintGL` cycle). Specifying a
particular mechanism here would add implementation-level detail this document doesn't
need in order to be buildable, and risks going stale if PySide6/VisPy's own recommended
integration approach shifts before 5a starts. Left to 5a's implementation.

**Deferred (M4): audio glide-state handling for rapid snapshot updates.** This is a
DSP/sonification behavior decision — how the synthesizer responds to a changing
parameter — not a software-architecture concern. It belongs to the audio engine's
detailed design once Audio Layer 2 is scoped, under the same boundary that already
produced v2's deferral of finding #17 above: this document owns *mechanism* (how
parameters reach the audio thread safely), not *behavior* (what the synthesizer does
with them).

**Deferred (L1–L3): all three LOW findings.** None are load-bearing for a first
implementation. `canvas_id` can be formalized when Compare mode (5h) is actually built;
the audio→UI position hand-off is already functionally covered by Part 4.6's design,
only its wording was imprecise; the unconditional-redraw rationale is a documentation
nicety, not a behavior gap (Part 6 already implies it). Deferring these keeps this
revision focused on changes with a real maintainability payoff, per this pass's own
instruction not to add complexity without one.

---

## Part 1 — Application Philosophy

*(Unchanged since v1.)*

VectorChess is not a chess GUI with a visualization bolted on. It is a scientific/musical
instrument whose input happens to be a chess position. The distinction matters for every
decision below: **the mathematical canvas is the largest panel and the primary focus of
attention; the board is the input device.**

**What the user should feel.** Contemplative curiosity, not competitive tension. Ableton
Live and TouchDesigner do not tell you if you're winning — they let you watch a system
respond to your input. `docs/audio.md` already commits to this register ("observing a
system than being told a story"); the visual and interaction design must match it. There
is no engine evaluation bar, no "+2.3 = you're winning" framing. The one always-visible
readout is the same `balance` scalar `docs/audio.md` already treats as the primary,
continuously-audible signal — reframed as *dominance*, not *advantage*.

**What the user should learn.** That positional concepts they may already have an
intuition for — space, tension, an open file, a weak square, a fortress in the endgame —
are not metaphors. They are measurable geometry. Hearing a corridor "flicker" apart in
sound the moment a blocking pawn is captured, at the same instant the ridge visibly
fragments on screen, is the specific experience the whole architecture is built to
produce.

**Why this differs from ChessBase or Lichess.** Those tools answer "what move is best."
VectorChess never computes or displays a best move — it has no engine. It answers a
different question: *what shape does this position have, and how does that shape sound?*

---

## Part 2 — Main Window

*(Unchanged since v2.)* Four fixed panels plus a thin top bar and a bottom transport
strip. Ratios (1600×1000 reference): top bar 40px, board panel 22% width, canvas panel
56% width, inspector 22% width, bottom strip 180px.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ VectorChess  [Play][Analyze][Compare]  [Load PGN] [Freeze Vis ▢][Freeze Audio ▢] [Record][Export] │
├───────────────────┬──────────────────────────────────────────────────┬───────────────────┤
│                    │  zoom 140%                          [Layers ▾]  │                   │
│    CHESS BOARD     │              MATHEMATICAL CANVAS                 │     INSPECTOR      │
│      (input)       │           (the protagonist — biggest panel)      │                   │
│                    │                                                  │  Move 14 Nxd5      │
│  8 r n b q k b n r │     [ GL canvas, composited registered layers,   │  Balance:  +2.3    │
│  7 p p p p . p p p │       animating continuously, pan/zoom enabled ] │  Mobility: 31 / 24  │
│  6 . . . . . . . . │                                                  │  Crit.pts: 3 (2 ok) │
│  5 . . . p . . . . │                                                  │  Ridges:   2 (1 ok) │
│  4 . . . P . . . . │                                                  │  Cells:    6 (5 ok) │
│  3 . . . . . . . . │                                                  │                   │
│  2 P P P . P P P P │                                                  │  ▸ selection:      │
│  1 R N B Q K B N R │                                                  │    saddle          │
│    a b c d e f g h │                                                  │    x=4.1  y=3.8    │
│                    │  Layers: [AI][Surf][Grad][Equip][CP][RV][MS]    │    D = -0.014       │
├───────────────────┴──────────────────────────────────────────────────┴───────────────────┤
│ ⏮ ⏪ ▶ ⏩ ⏭   |======●===┬══════════════════════════════|  14 / 38 ply   ↳ branch (2)     │
│ ♪  Harmony [M][S]  Melody [M][S]  Accent [M][S]  Drone [M][S]  Space [M][S]      🔊 ▮▮▮▯▯   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

Panel roles: board = input, canvas = protagonist, inspector = read-only mirror, bottom
strip = transport + mixer + branch navigation. The `┬` / `↳ branch (2)` markers indicate
a variation point in the game tree (Part 4.4). The mixer row is now driven by the Voice
Registry (Part 5) rather than five hardcoded names, though the default five voices'
labels are unchanged.

---

## Part 3 — Interactive Chess Board

*(Unchanged since v2.)* Built on `python-chess` and
`chess_engine.analyzer.analyze_position`. Drag/click-to-move, hover highlight,
legal-move highlight, jump to any move, and tree-aware undo/redo: undo moves the
"current node" pointer to its parent; playing a move that doesn't match an existing
child calls `add_variation()`, creating a sibling branch rather than deleting the old
line (Part 4.4). No new chess logic — `chess.pgn.GameNode`, `.variations`, and
`.add_variation()` already exist in the pinned `python-chess` dependency; Milestone 5
was always going to use `chess.pgn` for "Load PGN," and this reuses the same object for
live play instead of a second, lossy linear structure.

---

## Part 4 — Software Architecture

Presentation-layer architecture only. No `chess_engine`, `analysis`, or `audio` module
is modified by anything in this part, in v2 or v3.

### 4.1 SessionState — single source of truth

**Problem it fixes.** Cross-panel behavior (hover updates the Inspector, selecting a
critical point updates the Inspector, layer-solo and voice-solo share a vocabulary) needs
one place that owns it, or panels end up wired directly to each other.

**Design.** One object, `SessionState`, owned by the application root (not by any panel),
is the only thing that holds cross-panel state. Panels subscribe to the slices they care
about and call `SessionState` methods to request changes; **no panel ever reads or
mutates another panel's internal widget state directly.**

| SessionState field | Meaning | Written by |
|---|---|---|
| `game_tree` | The branching move tree (Part 4.4) wrapping `chess_engine.board.ChessGame` | Board panel (moves), Timeline (branch switch), PGN loader |
| `current_node` | Pointer to the tree node defining "the current position" | Board, Timeline |
| `selection` | Currently selected/hovered square, critical point, ridge/valley chain, or MS cell | Board (hover), Canvas (click) |
| `layer_state` | Per-registered-layer visibility / opacity / solo, keyed by `layer_id` (Part 5) | Canvas's layer strip |
| `mixer_state` | Per-registered-voice mute / solo / volume, keyed by `voice_id` (Part 5) | Bottom mixer strip |
| `freeze_visualization`, `freeze_audio` | The two independent freeze flags | Top bar |
| `transport_state` | Playing / Paused / Scrubbing, plus scrub position | Timeline |
| `camera` (per canvas instance) | Pan offset + zoom scale | Canvas |
| `compare_state` | Off, or the two node references being compared | Top bar mode switch |

`layer_state` and `mixer_state` are keyed by plain string IDs rather than by references
to `LayerDefinition`/`VoiceDefinition` objects — `SessionState` never imports the Layer
or Voice Registry (Part 5); anything that needs both a state value and its registry
definition (the layer strip, the mixer strip, the renderer) cross-references the two
separately. This keeps the dependency one-directional (Registry code may read
`SessionState`; `SessionState` never depends on Registry code) and avoids a circular
import between the two.

**Update propagation.** `SessionState` exposes one typed change-notification signal per
logical slice, matching the table above row for row: a `game_tree`/`current_node`
signal, a `selection` signal, a `layer_state` signal, a `mixer_state` signal, a
`freeze_*` signal, a `transport_state` signal, a `camera` signal per canvas instance, and
a `compare_state` signal — implemented as Qt signals, since PySide6 is already the
chosen shell (Part 10). This is deliberately neither one signal per individual field
(which would fragment into dozens of near-duplicate connections as the state grows) nor
one monolithic "something changed" signal (which would force every panel to re-render on
every unrelated change, defeating the point of scoping ownership per row in the first
place). A panel connects only to the slices Part 4.2's ownership table says it reads. No
panel polls `SessionState`, and no panel inspects another panel's internals to detect a
change — this is the one, uniform mechanism every "X updates when Y happens" statement
elsewhere in this document relies on.

**Why needed.** Without an explicit propagation mechanism, "panels subscribe to the
slices they care about" is a description of an outcome, not a design — different phases
(5b's board, 5e's timeline, 5g's inspector) would each be free to invent their own
wiring, reproducing exactly the ad hoc coupling this document exists to prevent. This was
the single foundational gap remaining after v2 fixed *what* is shared and *who owns it*
without fixing *how changes propagate*.

**Integration with existing code.** `SessionState.game_tree` wraps
`chess_engine.board.ChessGame` — it does not replace it. `ChessGame` continues to own
chess rules and legality exactly as today; `SessionState` only adds the presentation
concerns (selection, layers, audio, camera, freeze) around it.

**API changes.** None. `SessionState` is a new, additive module in the UI layer.

**Effect on future milestones.** 5a (rendering substrate) now has an explicit first
deliverable — build the `SessionState` skeleton with its per-slice signals — before any
panel is wired up, since every later phase (5b–5j) depends on both existing and depends
on connecting to the *same* mechanism rather than each inventing its own.

### 4.2 Data ownership

Restates the table above from the "who may write / who may read" angle, and names the
one category not owned by `SessionState` at all — the per-position math results:

| Data | Owner | Written by | Read by |
|---|---|---|---|
| Chess rules state | `chess_engine.board.ChessGame`, inside `SessionState.game_tree` | Board (via `SessionState.make_move`) | All panels |
| Per-position math fields (Surface, Gradient, Critical Points, Ridge/Valley, Morse-Smale) | Position Cache (Part 4.3) | Populated asynchronously on first visit to a position | Canvas, Inspector |
| Cross-move correspondence between two cached positions | Correspondence cache (Part 4.5), stored alongside the Position Cache | Computed on first request, memoized | Canvas's animation driver, Compare mode's diffing |
| Selection / hover | `SessionState.selection` | Board, Canvas | Inspector, Canvas |
| Layer / voice visibility, solo, mute | `SessionState.layer_state`, `mixer_state` | Canvas's layer strip, mixer strip | Canvas, Layer Registry, Voice Registry, audio engine |
| Transport / freeze | `SessionState.transport_state`, `freeze_*` | Timeline, top bar | Canvas (animation driver), audio engine |
| Camera | `SessionState.camera[canvas_id]` | Canvas's own zoom/pan handling | Canvas only |

Inspector never computes anything; it only renders values already produced by
`analysis/` and mirrored into `SessionState`/the Position Cache — an explicit,
checkable rule, not just an intent.

### 4.3 Position Cache — keyed by FEN, not ply index

**Problem it fixes.** Ply index is not a stable key: undo followed by a different move
produces a different position at the same ply index, so a ply-indexed cache would
silently serve stale data for the new line.

**Design.** The cache key is `chess.Board.board_fen()` — the piece-placement field of
FEN, already a method `python-chess` provides today. This is deliberately narrower than
full FEN: `AttackInfluenceField` and everything built on it depend only on piece
placement, not on side-to-move, castling rights, or move counters, so keying on
`board_fen()` alone is both correct and gives a small free benefit — two different move
orders that transpose to the same piece placement share one cache entry. The cached
value bundles everything `analysis/` can produce for that position (`MoveAnalysis`,
`AttackInfluenceSurface`, `GradientField`, `CriticalPointQualityAssessment` list,
`RidgeValleyQualityAssessment` list, `MorseSmaleComplex` + cell quality).

**Concurrency and asynchronous population.** A cache miss is not computed synchronously
on the UI thread — it is dispatched to a background worker (a process pool, since the
math pipeline's `scipy`/`numpy` work is CPU-bound and a process pool avoids GIL
contention with the UI thread). Each cache entry carries an explicit state — `missing`,
`computing`, or `ready` — rather than being treated as simply present-or-absent. The
transition to `ready` is published through the same signal mechanism as 4.1 (a
`position_ready(fen)` notification), so the canvas re-renders the newly-available layer
on its next frame rather than polling cache state. The renderer (Part 5) draws whatever
registered layers are already `ready` immediately, and shows a lightweight, stated
placeholder for any layer still `computing`, rather than blocking the frame. A small
prefetch policy computes the one or two plies adjacent to the current scrub position
ahead of time, so ordinary sequential scrubbing rarely observes the placeholder at all.
Because the cache is now written by a worker and read by the UI thread, entries are
inserted via a single atomic swap — the same snapshot discipline as 4.6, applied here to
cache entries rather than audio parameters — so a UI-thread read can never observe a
partially-populated entry.

**Cache sizing.** A bounded LRU, default 200 entries. This comfortably covers one full
game (typically well under 100 plies) plus a second full game in Compare mode, with
headroom for switching between two or three loaded games in one session. A user who
exceeds this degrades gracefully to more frequent recomputation, not incorrect data — an
explicit, accepted trade-off. Each entry is a few hundred KB to low single-digit MB
(several dense float grids), so 200 entries is a bounded, modest memory footprint even
in the worst case this document anticipates.

**Resolution is read, not assumed.** Every reference elsewhere in this document to a
specific grid resolution refers to `analysis/attack_influence_surface.py`'s current
`DEFAULT_RESOLUTION`, not a fact this document should assume permanent. The cache, the
decimation logic, and GPU buffer sizing (Part 11) read the actual resolution from the
cached `AttackInfluenceSurface`'s own array shape at runtime. If `DEFAULT_RESOLUTION`
ever changes, nothing in the presentation layer needs to change with it.

**Why needed.** Correctness (no stale data across branches), a resilient answer to
"what happens when the math pipeline's own constants evolve," and — for the
concurrency design specifically — protecting the Part 11 frame budget from the entirely
normal act of scrubbing into a never-visited position.

**Integration with existing code.** A thin new module wrapping unmodified `analysis/*`
builder calls; `board_fen()` is an existing `python-chess` method, not a new API.

**API changes.** None.

**Effect on future milestones.** Directly enables Part 4.4 (a tree of positions caches
correctly regardless of how a node was reached); 5a's scope includes building this
cache module, its background worker, and its signal integration alongside
`SessionState`.

### 4.4 Timeline vs. branching game tree

*(Unchanged since v2.)* `SessionState.game_tree` is a `chess.pgn.GameNode` tree — the
same structure `python-chess` already uses for PGN parsing, reused for live play. Undo
moves `current_node` to its parent. Playing a move that doesn't match an existing child
calls `add_variation()`, creating a sibling rather than deleting the previous line. The
Timeline panel (Part 8) renders the path from the tree root to `current_node` as its
main scrubber, with a branch indicator at any node with more than one child, letting the
user switch the active child without losing the other branch.

**Why needed.** Normal chess-analysis behavior (try a move, take it back, try another)
that a linear model cannot represent without silently discarding work.

**Integration with existing code.** No new chess logic — `chess.pgn.GameNode`,
`.variations`, and `.add_variation()` already exist in the pinned `python-chess`
dependency.

**API changes.** None.

**Effect on future milestones.** 5e ("Timeline & scrubbing") includes branch-point
rendering and switching, not just a linear slider.

### 4.5 Stable identity across moves — the correspondence problem

**Problem it fixes.** Animating between two positions and diffing two positions
(Compare mode) both require matching "this saddle at move 14" to "the corresponding
saddle at move 15." `CriticalPointCandidate`, `RidgeValleyChain`, and `MorseSmaleCell`
carry no persistent ID across positions — each position's math is recomputed
independently.

**Design.** A new, additive UI-layer module — **not** a change to `chess_engine.models`
or any `analysis/*` builder — computes a correspondence between two already-cached
positions' results, purely as a rendering/comparison aid:
- **Critical points**: matched by nearest spatial distance, constrained to the same
  classification, with a maximum-distance threshold beyond which a point counts as
  *appeared* or *disappeared* rather than *moved*. The threshold is a rendering
  constant, not a mathematical one.
- **Ridge/Valley chains**: matched primarily by their anchoring critical point's own
  match, falling back to endpoint/centroid proximity when the anchor doesn't carry over.
- **Morse-Smale cells**: matched by their bounding critical points' matches.

This mapping is never stored back into the math layer's data and produces no new
mathematical claim — a "match" is a rendering/UI judgment about *which drawn object to
morph into which*, not a statement about chess structure.

**Memoization.** A move-to-move transition (Part 6) replays the same matched pairing
across roughly thirty frames at 60fps over its ~500ms duration. The correspondence
between two adjacent positions is computed once, when the pair is first needed, and
cached alongside the Position Cache entries for that pair (Part 4.2's table; evicted
together with them, under the same LRU policy as Part 4.3) rather than recomputed every
frame.

**Why needed.** Without it, smooth interpolation and Compare-mode diffing are both
unimplementable — they need a concept the math layer intentionally doesn't provide.
Without memoization, the same well-defined computation runs redundantly dozens of times
per transition for no benefit.

**Integration with existing code.** A new module (conceptually `interactive_viz/
correspondence`), consuming the outputs of `analysis/critical_points.py`,
`analysis/ridge_valley.py`, and `analysis/morse_smale.py` unchanged, adding no fields to
any existing dataclass.

**API changes.** None.

**Effect on future milestones.** 5d (move-to-move animation) depends on this module
existing first, including its memoization. It also resolves Compare mode's diffing
(5h): a "diff" is the same matching with unmatched objects reported as
appeared/vanished.

### 4.6 Thread-safe UI ↔ audio communication

**Problem it fixes.** `sounddevice`'s real-time callback thread must never block. If it
read `SessionState` directly, a lock or a torn read under UI/GPU load could produce
audible glitches.

**Design.** A snapshot/publish pattern: whenever an audio-relevant `SessionState` field
changes, the UI thread publishes a new **immutable** snapshot; the audio callback thread
only ever reads the latest published snapshot via a single atomic pointer swap — no
locks, no partially updated state, ever, on the audio thread's hot path. The reverse
direction (the audio engine reporting actual playback position back to the UI, to drive
the Timeline's moving playhead) uses the same pattern in reverse: a lock-free position
counter written by the audio thread, copied into `SessionState.transport_state` once per
UI-thread frame tick, and published through 4.1's normal `transport_state` signal from
there — the audio thread never writes to `SessionState` itself.

**Latency budget.** End-to-end output latency of 20ms or less, which sets the
`sounddevice` buffer size. The audio callback checks for a newer published snapshot at
least once per buffer interval, so audio can never lag the visually-scrubbed position by
more than one buffer's worth of time. This gives Part 1's promise of felt audio-visual
synchrony, and 5f's own "stated latency bound" acceptance test, an actual number to be
measured against.

**Why needed.** A naive shared-mutable-state design here produces audible dropouts under
normal use, which would be a regression from the MVP's already-correct, glitch-free
(because offline) audio. The latency number is needed because Part 1 makes felt
synchrony a philosophy-level promise, not just a nice-to-have — and because a bound that
isn't stated can't be tested.

**Integration with existing code.** Wraps `audio/synthesis.py`'s existing pure
sample-generation functions with a new real-time scheduling layer; `audio/synthesis.py`
itself is unmodified.

**API changes.** None.

**Effect on future milestones.** 5f (live audio) is scoped to build this snapshot
mechanism, at the stated latency budget, as its first task, before wiring the mixer UI
to it. Verifying the mechanism is correct under concurrent load — not just that it
exists — is a dedicated testing concern; see Part 14.

### 4.7 Export isolation

**Problem it fixes.** The offline export driver (Part 10) reads a selected line's
positions to render frames and audio in lockstep. Left unspecified: whether export runs
synchronously (blocking the whole application for a multi-minute render) or in the
background — and if in the background, whether it reads `SessionState.game_tree` live
while the user is free to keep playing, which could mutate the very tree structure
export is iterating.

**Design.** Export runs in the background, not blocking the UI. When export is
requested, the export driver takes an immutable snapshot of the exact node range being
exported and its cached position/correspondence data — the same snapshot/publish
pattern 4.6 established for audio parameters, applied here to a slice of the game tree —
before starting. The live session is then free to keep mutating `game_tree` (new moves,
new branches) with no effect on the in-progress export, and the export driver never
touches `SessionState` again after that initial snapshot.

**Why needed.** Without this, a user playing a move during a multi-minute export could
corrupt the structure the export driver is iterating, or crash it outright — the same
class of hazard 4.6 solves for audio, not yet applied to the second background thread
this document introduces.

**Integration with existing code.** Reuses the snapshot pattern already specified in
4.6; no new mechanism, only a second application of it. Does not change Part 10's
offline rendering/DSP functions.

**API changes.** None.

**Effect on future milestones.** 5j (Export) is scoped to include building this snapshot
step as its first task, mirroring how 5f is scoped around 4.6's snapshot.

---

## Part 5 — Layer Registry, Voice Registry & Mathematical Visualization

**Problem it fixes.** A hardcoded layer stack requires touching the layer strip UI, the
cache, and the animation/audio-association logic every time a future mathematical
object (Temporal Derivative, Force Network — both named in CLAUDE.md's roadmap) needs a
visualization.

**Design — Layer Registry.** A `LayerDefinition` is a registered unit with:
- `id`, `display_name` — unchanged visual identity per layer (Part 9's colors, unchanged)
- `data_source(position_cache_entry) -> LayerFrame` — a thin adapter over one existing
  `analysis/*` builder's output
- `renderer(LayerFrame)` — draws it (Part 11's rendering strategy applies uniformly)
- `animator(LayerFrame, LayerFrame) -> interpolated frame` (optional) — reuses Part 4.5's
  correspondence module for discrete-object layers (critical points, chains, cells), or a
  plain grid lerp for continuous-field layers (surface, gradient)
- `audio_voice` (optional, explicitly *not* required to be one-to-one — see Review
  Disposition #16) — which registered voice, if any, this layer is associated with

A `LayerRegistry` holds an ordered list of these. The layer strip (Part 2), the layer
solo/visibility logic, and the animation driver (Part 6) all iterate the registry
generically; none of them hardcode a list of names. The six math-derived layers (Attack
Influence, Surface, Gradient, Equipotential, Critical Points, Ridge/Valley, Morse-Smale)
become six `LayerDefinition` registrations in the same dependency order established from
v1 — board squares/pieces remain outside the registry, since they aren't derived from
`analysis/` output.

**Animator is an open strategy, not a closed pair.** Grid lerp and Part 4.5's
point-correspondence matching are the two built-in implementations needed by the six
layers that exist today, not an exhaustive list. `animator` is an interface any future
layer can implement its own strategy against — a hypothetical future graph-shaped object
(e.g., CLAUDE.md's own "Force Network" idea) would supply its own edge-matching
strategy rather than being forced into "grid" or "point set." No such object exists yet
and none is designed here; this is a documentation clarification, not new mechanism.

**Design — Voice Registry (mirrors the Layer Registry).** A `VoiceDefinition` has an
`id`, `display_name`, mute/solo state, and a contribution to the 4.6 snapshot the audio
engine reads. The `VoiceRegistry` holds an ordered list of these; the mixer strip (Part
2, Part 8) and the audio engine both iterate it generically. `docs/audio.md`'s five
current voices (Harmony, Melody, Accent, Drone, Space) become five initial
registrations. This registry decides only *how voices are discovered and iterated* — it
does not decide *which voices exist or what they mean musically*, which remains
`docs/audio.md`'s domain, exactly as established by the rejection of finding #16 above.

**Why needed.** Makes adding a future mathematical object's visualization additive (one
new `LayerDefinition`) instead of invasive. On the audio side, `docs/audio.md` §5 names
a future Generative Music Engine stage with more voices (plausibly one per Morse-Smale
cell, per Part 7/12's framing of cells as musical sections); without a registry, that
stage would require hand-touching the mixer UI and the audio engine's voice-iteration
logic directly — the same invasive pattern the Layer Registry avoids on the visual side.

**Integration with existing code.** Each `LayerDefinition.data_source` calls an existing
`analysis/*` builder function unchanged. Each registered voice's synthesis still calls
the unmodified `audio/mapping.py`/`audio/synthesis.py` functions `docs/audio.md`
already defines; the registry only changes how the mixer and engine enumerate voices,
not what any voice does.

**API changes.** None.

**Effect on future milestones.** 5c becomes "build the Layer Registry and Voice
Registry and register the six existing field layers and five existing voices through
them" — each registration independently testable. 5f's mixer UI is built against the
Voice Registry from the start, so a future voice addition (still gated behind its own
Audio Layer/Generative Engine milestone) is additive.

**Rendering pipeline.** Each field is computed once per position (via the Part 4.3
cache, asynchronously) and rendered every frame from the cached arrays; recomputation
only happens when `current_node` changes, never on hover, pan/zoom, or idle animation.

---

## Part 6 — Animation

*(Unchanged since v2.)* Per-object animation, driven by Part 4.5's correspondence
module for discrete-object layers:

| Object | Animation | Driven by |
|---|---|---|
| Surface | Linear interpolation of the cached `z` grid between two positions, ~500ms eased | Plain grid lerp (both grids share domain/resolution exactly) |
| Gradient | Tracer particles advected along the (static, between moves) vector field, looping | Static per-position field; particle motion, not data interpolation |
| Critical Points | Position/confidence interpolate between matched pairs; unmatched points fade in/out | Part 4.5 correspondence (memoized) |
| Ridge / Valley | Chain points interpolate along matched chains; a chain whose match breaks animates a visible split at the tracer's own recorded termination point | Part 4.5 correspondence (memoized) |
| Morse-Smale cells | Fill/boundary cross-fade on birth/death of matched cells | Part 4.5 correspondence (memoized) |
| Camera | Smooth pan/zoom on timeline jump, ~300ms ease-in-out; user-driven zoom/pan (Part 8) is instantaneous, not eased | `SessionState.camera` |

What stays static: board squares/pieces, the coordinate grid, all UI chrome.

---

## Part 7 — Music

Mappings unchanged since v1 (an extension of `docs/audio.md`, not a redesign of it).

- **Delivery** names Part 4.6's snapshot mechanism, at its 20ms latency budget, as the
  hand-off between the UI thread's `SessionState` and the real-time `sounddevice`
  callback — the synthesis pipeline itself (`audio/synthesis.py`, `audio/mapping.py`)
  is unchanged. The mixer is built against the Voice Registry (Part 5) rather than a
  fixed voice list.
- **Live vs. exported audio identity.** They are guaranteed identical whenever the user
  is not actively scrubbing faster than the audio engine's glide can track — both paths
  call the same `audio/synthesis.py`/`audio/mapping.py` functions against the same
  cached data, and the offline driver (Part 10) exists specifically to reproduce that
  output deterministically. The one allowed divergence is during fast, manual live
  scrubbing, where the real-time engine reacts to actual scrub speed and an export of
  the same range (walked at a fixed frame rate, not the scrub gesture's speed)
  necessarily does not — an inherent difference between "watching a performance" and
  "rendering a fixed sequence," not a bug.
- **Source Potential (pan) vs. Ridge/Valley (pan) collision** is *not* resolved in this
  document (Review Disposition #17) — a Layer 2 mapping decision belonging to
  `docs/audio.md`'s own governance, flagged here for whoever scopes that milestone.

All other content (base mappings, calm/tactical/endgame descriptions, style constraints)
is unchanged since v1 — see `docs/audio.md` directly.

---

## Part 8 — User Interaction

*(Unchanged since v2.)* Builds on the interaction list below:

- **Freeze — two explicit, independently-exposed controls:** **Freeze Visualization**
  stops idle ambient animation and holds the canvas at an exact still frame; **Freeze
  Audio** sustains the current position's harmonic bed instead of gliding to the next.
  Both are `SessionState` flags with their own top-bar toggle. Neither freezes chess
  play. Re-enabling Freeze Audio after a scrub resumes the glide from the frozen state
  to the current one, using the same portamento mechanism as a normal transition.
- **Scrub through time** — continuous, driving Part 6's lerp machinery at the mouse's
  fractional position, scoped to the mainline path from tree root to `current_node`;
  scrubbing across a branch point follows whichever child is currently active.
- **Switch branches** — clicking the `↳ branch (n)` indicator at a variation point
  switches which child is active without discarding the other.
- **Solo one layer / one voice** — independently-scoped controls sharing an interaction
  convention ("solo hides/mutes everything else in this domain"), not a claim that a
  layer and a voice are the same thing (Review Disposition #16).
- **Compare two positions / split screen** — each panel gets its own `SessionState.
  camera` entry so zoom/pan is independent per panel; audio follows whichever panel is
  currently focused — audio has one owner at a time, never both panels simultaneously.
- **Zoom / Pan** — mouse wheel zooms around the cursor; a modifier-drag pans. Stored
  per-canvas in `SessionState.camera`. Remains active under Freeze Visualization —
  freezing stops data animation, not user camera control.
- **Keyboard shortcuts:** Space = play/pause; Left/Right = step move; Up/Down = switch
  active branch at the nearest variation point; number keys = toggle the corresponding
  registered layer (bound by registry position); `F` / `Shift+F` = Freeze Visualization
  / Freeze Audio; `+`/`-`/scroll = zoom; `0` = reset camera. Every shortcut dispatches to
  an existing `SessionState` method.
- **Export screenshot** — a one-frame capture of the current canvas state.
- **Record / export video, export MIDI, export WAV** — implementation defined in Part
  10 (offline deterministic export) and isolated from live session mutation by Part 4.7.

---

## Part 9 — Visual Identity & Accessibility

*(Unchanged since v2.)* Chrome, dominance colors, scientific field colors, board colors,
typography, icons, grid, glow, line widths, and animation speed are all unchanged since
v1 — this document does not redesign the visual language.

**Accessibility (additive, not a palette change):**
- An optional toggleable overlay (off by default) adds a small "W"/"B" glyph or a
  texture difference to dominance-colored regions, extending the balance meter's
  existing text-label discipline to the canvas.
- An optional, user-selectable alternate hue pair (e.g., blue/orange) swaps only the two
  dominance hues, leaving every other rule unchanged — a user preference layered on top
  of the existing identity, not a default change.
- Full keyboard navigability (Part 8) covers the motor-accessibility path.

---

## Part 10 — Technology

| Concern | Choice |
|---|---|
| Application shell | PySide6 (Qt for Python) |
| Real-time canvas rendering | VisPy, embedded in a `QOpenGLWidget`; fall back to raw ModernGL for the surface/gradient hot path only if profiling shows the Part 11 budget can't be met. **Phase 5a implementation note:** neither VisPy (latest 0.16.2, wheels through Python 3.12) nor ModernGL (latest 5.12.0, wheels through Python 3.13) ship a wheel for this project's pinned Python 3.14.5; PySide6 (`abi3`) and PyOpenGL (`py3-none-any`) both do. Phase 5a uses **PySide6 + PyOpenGL directly** (a hand-written core-profile shader/VBO in `desktop_app/gl_canvas.py`) instead of VisPy. This does not change SessionState, the Position Cache, the Layer Registry, or the rendering pipeline design, all of which are renderer-agnostic — only the specific binding library used to issue GL calls. VisPy remains a candidate to revisit once its wheels support the project's Python version. |
| Static/offline visualization | `visualization/*.py` (matplotlib) unchanged — the console app's and regression tests' renderer, not replaced |
| Audio synthesis | `audio/synthesis.py` (numpy) unchanged |
| Real-time audio output | `sounddevice`, tuned to Part 4.6's 20ms latency budget |
| MIDI export | `mido` |
| Video export | FFmpeg, via `imageio-ffmpeg` or subprocess |
| Testing | `pytest`, plus `pytest-qt` for the GUI event loop and the concurrency stress harness (Part 14) |
| Explicitly rejected | FluidSynth/soundfont playback (wrong tool for a pure-oscillator palette, per `docs/audio.md` §2); Dear PyGui/Tkinter as the primary shell |

**Export.** The offline deterministic export driver (Part 10 origin, unchanged since
v2) steps the selected line at a fixed frame rate, rendering each frame off-screen and
generating the corresponding audio samples in lockstep from the same cached data and the
same `audio/synthesis.py`/`audio/mapping.py` functions — no wall-clock or real-time audio
device involved, so video and audio are sample-accurate by construction. It now
explicitly depends on Part 4.7's snapshot mechanism to avoid reading a live-mutating
`SessionState`; no change to the rendering/DSP functions themselves.

**Qt/VisPy render-loop integration** (the specific mechanism connecting Qt's event loop
to VisPy/OpenGL's render loop, e.g. a `QTimer`-driven `paintGL` cycle) is deliberately
left to 5a's implementation rather than fixed here — a well-known, low-risk pattern
whose specific form doesn't need to be frozen at the architecture level (Hardening Pass
Disposition, M3).

---

## Part 11 — Performance Budget & Rendering Strategy

- **Target**: 60fps sustained at the reference canvas size, with **all six registered
  field layers visible simultaneously** — the worst case, not the common case.
- **Grid layers** keep full fidelity in the Position Cache (needed for Inspector
  precision and the offline export driver's full-quality frames), but the live
  interactive mesh may subsample to a coarser, stated grid for real-time display — an
  explicit, tunable rendering constant, not a silent assumption. Both the cached and the
  decimated resolutions are read from the cached `AttackInfluenceSurface`'s own array
  shape at runtime, never hardcoded (Part 4.3).
- **Alpha-blended layers**: gradient tracer particles are capped at a stated maximum
  count. Morse-Smale cell fills are cheap by a fortunate property of the math itself —
  accepted cells partition the surface and don't overlap by construction, so their fills
  never stack.
- **GPU-side buffer caching.** Uploaded vertex/texture buffers per layer per position are
  cached on the GPU side the same way their source data is cached on the CPU side (Part
  4.3) — uploaded once when a position is first rendered, reused across frames,
  invalidated only when `current_node` changes. Without this, CPU-side caching wouldn't
  fully reach the rendering path, since re-uploading unchanged data to the GPU every
  frame has its own real cost.
- **Audio latency** is tracked as its own budget (20ms, Part 4.6), on its own thread, not
  merged into the frame budget above.
- **Objective GL fallback trigger**: if VisPy cannot sustain the stated 60fps/six-layer/
  reference-size budget in profiling, that is the specific, measured condition under
  which the surface/gradient hot path drops to raw ModernGL.
- **Offline export** is explicitly exempt from this real-time budget — it can render at
  full fidelity and take longer than one frame-interval per frame, since it is not
  interactive.
- **Standing regression, not a one-time gate.** This budget is re-verified at the end of
  every phase from 5a onward (Part 13), not only once at 5a — see Part 14.

---

## Part 12 — Extensibility

- **A future mathematical object** (e.g., Temporal Derivative `dF/dt`) requires: one new
  `analysis/` module (unaffected by this document), one new `LayerDefinition` registered
  (Part 5), and — because Part 4.5's correspondence module and Part 5's `animator`
  interface are both written against *shapes* of data, not specific field types — a
  grid-shaped new object gets interpolation for free, and a discrete-object-shaped one
  only needs to state its own classification/anchor rule.
- **A future audio voice** (e.g., a per-cell voice for the Generative Music Engine, per
  `docs/audio.md` §5) is equally additive: register it in the Voice Registry (Part 5)
  the same way a future visual layer is registered in the Layer Registry — neither
  requires touching the mixer UI or the layer strip directly.
- **N-way Compare** (beyond two panels) is not built now, but nothing in Part 4's design
  assumes exactly two: `SessionState.camera` and `compare_state` are already scoped
  per-canvas-instance rather than hardcoded to a pair.
- **A future Generative Music Engine or Live Performance Mode** (per `docs/audio.md` §5)
  plugs into the same Part 4.6 thread-safety boundary and the same Voice Registry
  already defined here — no new integration surface needed when that milestone arrives.
- **A general plugin system for adding entirely new panels** is explicitly *not*
  designed here — nothing in the current roadmap justifies that generality, and building
  it speculatively would be the kind of complexity CLAUDE.md's own philosophy warns
  against. The Layer Registry and Voice Registry solve the extensibility problems that
  are actually anticipated; a panel-level plugin system is not one of them.

---

## Part 13 — Implementation Roadmap

`docs/roadmap.md` places Milestone 4b (Audio Layer 2) before Milestone 5 (Interactive
UI). Only the phase that plays audio live needs 4b to exist first; everything else
proceeds independently.

| Phase | Scope | Depends on | Independent test |
|---|---|---|---|
| **5a** — Substrate | PySide6 shell; `SessionState` skeleton with its per-slice update signals (4.1); Position Cache with async background population, entry-state model, and prefetch (4.3); VisPy canvas rendering one static field, no animation | — | Rendered output matches `visualization/attack_influence_plot.py` for a fixed position; sustains Part 11's frame budget; a simulated cache miss during scrubbing never blocks a frame |
| **5b** — Interactive board | Drag/drop, legal-move highlight, tree-aware undo/redo (4.4), PGN load with variations | 5a | A scripted move/PGN sequence (including a variation) produces identical `MoveAnalysis` per node to the console app |
| **5c** — Layer Registry & Voice Registry | Both registries (Part 5) stood up; all six field layers and five audio voices registered and togglable, layers still (non-animated); accessibility overlay toggle | 5a | Registering a seventh dummy layer, or a sixth dummy voice, requires no change to either UI strip |
| **5d** — Correspondence & animation | Part 4.5's matching module, memoized; move-to-move animation (Part 6) | 5c | Interpolation invariants hold at t=0/t=1; a synthetic two-position pair with a known appearing/disappearing critical point matches correctly; the match is computed once per pair, not per frame |
| **5e** — Timeline, scrubbing, branches | Continuous scrub; branch-point indicator and switching (4.4) | 5b, 5d | Scrubbing to an exact ply reproduces the same static state as jumping directly; switching branches preserves the non-active line |
| **5f** — Live audio | Depends on **Milestone 4b**; Part 4.6's snapshot mechanism at the 20ms latency budget; mixer UI built against the Voice Registry | 5e, **4b** | Measured playback position tracks the visual scrub position within the 20ms budget; the concurrency stress test (Part 14) passes; no audible glitches under a stated GPU-load stress test |
| **5g** — Inspector | Live numeric readouts; selection linking (4.2) | 5c | Every displayed number traces to a specific field in `chess_engine/models.py`, none computed ad hoc in the UI |
| **5h** — Freeze / solo / compare | Both freeze modes; layer/voice solo; split-screen compare with independent camera | 5d, 5f | Freeze provably stops only rendering/audio, never game state; Compare diff matches a manual diff via the 4.5 correspondence module |
| **5i** — Keyboard, zoom/pan | Full shortcut set; pan/zoom | 5c | Every mouse interaction has a working keyboard equivalent |
| **5j** — Export | Offline deterministic export driver with Part 4.7's snapshot isolation; video, MIDI, WAV range export | 5d, 5g | Exported audio matches an offline WAV render sample-for-sample on repeated runs; a move played mid-export does not affect the in-progress export's output |

Generative Music Engine and Live Performance Mode (`docs/audio.md` §5) remain gated
behind their own future math milestones, unchanged by this document.

---

## Part 14 — Testing Strategy

Consolidates cross-cutting testing concerns the per-phase table above doesn't fully
capture on its own.

- **Component-level tests**, mirroring the project's existing model/builder/
  visualization/tests discipline (CLAUDE.md), applied to the new presentation-layer
  components: `SessionState` (state-transition tests — e.g., undo from a node with
  existing children creates a sibling, never deletes), Position Cache (FEN-keying
  correctness, transposition sharing, LRU eviction, entry-state transitions), the
  Correspondence module (synthetic before/after critical-point sets matched correctly,
  including the appear/disappear distance threshold, and memoization actually avoids
  recomputation), Layer Registry and Voice Registry (registering an additional entry
  requires no change elsewhere — already 5c's acceptance test).
- **Concurrency stress test.** A dedicated test target, separate from ordinary unit
  tests, applied to every place this document introduces a lock-free snapshot: the
  audio parameter hand-off (4.6), Position Cache population (4.3), and export isolation
  (4.7). Multiple threads mutate `SessionState` continuously while a tight-loop reader
  (standing in for the real audio callback, or the export driver's initial snapshot)
  asserts every observed snapshot is internally consistent — never a torn or
  partially-updated read. This is the one class of defect in this document that would
  otherwise manifest as an intermittent, hard-to-reproduce production bug rather than a
  clean test failure.
- **Cross-phase end-to-end test.** One scripted scenario exercising SessionState,
  Position Cache, Correspondence, both registries, and Export together — load a PGN
  containing a variation, scrub through it with several layers toggled, freeze audio,
  switch branches, export a range — since integration boundaries between
  independently-tested phases are where bugs concentrate.
- **Standing performance regression.** Part 11's frame-budget measurement is re-run at
  the end of every phase from 5a onward, not only once at 5a, so a regression
  introduced by, say, 5d's correspondence computation or 5j's export driver is caught by
  the same test that validated 5a rather than going unmeasured.

---

## Final Self-Review

**Is the architecture internally consistent?** Yes. The same snapshot/isolation
pattern, established once in Part 4.6 for audio parameters, is deliberately reused
rather than reinvented in two more places (Position Cache population, Part 4.3; export
isolation, Part 4.7) — one mechanism, three applications, rather than three different
ones. The Layer Registry and Voice Registry (Part 5) are structurally identical for the
same reason. `SessionState`'s update-propagation signals (Part 4.1) are the single
mechanism every cross-panel behavior in Parts 5–8 relies on, with no panel using a
different one. No circular dependency was found or introduced: Registry code may read
`SessionState`, `SessionState` never depends on Registry code (layer/voice state is
keyed by plain string IDs, Part 4.1); the Correspondence module depends only on
`analysis/*` outputs, never the reverse.

**Are there any remaining BLOCKER issues?** None identified. The one BLOCKER from the
v2 review (SessionState's update-propagation mechanism) is resolved in Part 4.1, and
that resolution is load-bearing throughout the rest of the document rather than
isolated to one section.

**Are there any remaining HIGH issues?** None identified. All five HIGH findings from
the v2 review are resolved: Position Cache concurrency (Part 4.3), export/session
isolation (Part 4.7), the audio latency budget (Part 4.6), the Voice Registry (Part 5),
and the concurrency stress-testing strategy (Part 14).

**Is the architecture ready for implementation?** Yes. The presentation layer
(SessionState, Position Cache, Correspondence, Layer/Voice Registry) is now fully
specified: what each component owns, how it's populated, how it's kept safe under
concurrency, how it's tested, and how a future mathematical object or audio voice
extends it without invasive change. Two MEDIUM findings (Qt/VisPy render-loop
integration, audio glide-state handling) and three LOW findings remain deliberately
deferred — none are load-bearing for a first implementation, and each has a stated
reason it belongs to a later, more specific design pass rather than this one. The
mathematical pipeline, `docs/mathematics.md`, `docs/audio.md`'s sonification content,
the application philosophy, and the visual identity are untouched throughout.

**Verdict: Architecture frozen — ready for implementation.**
