# Measured Frequency Ranges — Phase A Diagnosis
Total moves measured: 46 across 3 games.
## Melody fundamental (destination-square pitch)
- **White fundamental**: min=366.7 Hz, median=586.7 Hz, max=880.0 Hz, n=24
- **Black fundamental**: min=440.0 Hz, median=880.0 Hz, max=880.0 Hz, n=22

Both colors draw from the *same* pitch formula (`_pitch_for_square`) — the fundamental range is not color-dependent; any White/Black fundamental range difference above is purely which squares each side happened to move to in these particular games, not a systematic mapping.

## Harmonic partials actually sounded (additive_tone output)
- **White partials (all, richness=1 -> == fundamental)**: min=366.7 Hz, median=586.7 Hz, max=880.0 Hz, n=24
- **Black partials (all, richness=3 -> fundamental+2nd+3rd)**: min=440.0 Hz, median=1485.0 Hz, max=2640.0 Hz, n=66
- Black partials > 8000 Hz: 0/66 (0.0%). White partials > 8000 Hz: 0/24.
- Black partials > 10000 Hz: 0/66 (0.0%). White partials > 10000 Hz: 0/24.

The ear's peak sensitivity to perceived harshness/piercingness sits roughly in the 2-5 kHz range (equal-loudness contours), with 8-10kHz+ content read as 'hiss'/'shrill' rather than tonal pitch. Black's 3rd partial reaching into or past this band on any move whose destination square sits in the upper half of the board's pitch range is the direct, measured mechanism for the shrillness complaint -- not a bug, but a direct consequence of summing un-capped upper harmonics on top of an already-wide (220-3520 Hz) fundamental range.

## Harmony voice
- **Harmony frequency (all moves)**: min=311.1 Hz, median=933.4 Hz, max=1244.5 Hz, n=46
- Harmony interval ratio range: min=1.0000, max=1.4142 (production bounds: 1.0 unison .. 1.4142 tritone, 1.25 check floor). Ratio values observed are continuous/unquantized, confirming harmony is never snapped to a discrete musical interval in production.

## Renderer-level compounding factors (cited, not re-measured)
- `HARMONY_VOICE_WEIGHT = 0.6` (audio/renderer.py) -- harmony is quieter than melody by design, so the shrillness is dominated by the melody voice's own upper partials, not the harmony voice.
- `SUSTAINED_PEAK_HEADROOM = 0.85` (audio/renderer.py) -- the melody+harmony bed is normalized to a fixed peak regardless of loudness, so a Black move's dense upper-partial energy is *not* attenuated relative to a White move's; normalization does not selectively soften Black's extra harmonics.

## Per-game breakdown

### scholars_mate (7 plies)
| ply | move | color | dest | pitch_hz | partials_hz | harmony_hz | ratio | check | capture | label |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | e2e4 | white | e4 | 586.7 | 587 | 829.7 | 1.414 | False | False | - |
| 2 | e7e5 | black | e5 | 733.3 | 733, 1467, 2200 | 733.3 | 1.000 | False | False | tense |
| 3 | d1h5 | white | h5 | 880.0 | 880 | 1244.5 | 1.414 | False | False | tense |
| 4 | b8c6 | black | c6 | 733.3 | 733, 1467, 2200 | 1037.1 | 1.414 | False | False | active |
| 5 | f1c4 | white | c4 | 495.0 | 495 | 700.0 | 1.414 | False | False | active |
| 6 | g8f6 | black | f6 | 880.0 | 880, 1760, 2640 | 1244.5 | 1.414 | False | False | active |
| 7 | h5f7 | white | f7 | 880.0 | 880 | 1244.5 | 1.414 | True | True | tense |

### fastest_stalemate (19 plies)
| ply | move | color | dest | pitch_hz | partials_hz | harmony_hz | ratio | check | capture | label |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | e2e3 | white | e3 | 495.0 | 495 | 700.0 | 1.414 | False | False | - |
| 2 | a7a5 | black | a5 | 495.0 | 495, 990, 1485 | 700.0 | 1.414 | False | False | active |
| 3 | d1h5 | white | h5 | 880.0 | 880 | 1244.5 | 1.414 | False | False | chaotic |
| 4 | a8a6 | black | a6 | 586.7 | 587, 1173, 1760 | 829.7 | 1.414 | False | False | active |
| 5 | h5a5 | white | a5 | 495.0 | 495 | 700.0 | 1.414 | False | True | active |
| 6 | h7h5 | black | h5 | 880.0 | 880, 1760, 2640 | 1244.5 | 1.414 | False | False | active |
| 7 | a5c7 | white | c7 | 880.0 | 880 | 1244.5 | 1.414 | False | True | tense |
| 8 | a6h6 | black | h6 | 880.0 | 880, 1760, 2640 | 1244.5 | 1.414 | False | False | tense |
| 9 | h2h4 | white | h4 | 825.0 | 825 | 1166.7 | 1.414 | False | False | active |
| 10 | f7f6 | black | f6 | 880.0 | 880, 1760, 2640 | 1244.5 | 1.414 | False | False | active |
| 11 | c7d7 | white | d7 | 880.0 | 880 | 1244.5 | 1.414 | True | True | active |
| 12 | e8f7 | black | f7 | 880.0 | 880, 1760, 2640 | 1244.5 | 1.414 | False | False | active |
| 13 | d7b7 | white | b7 | 825.0 | 825 | 1166.7 | 1.414 | False | True | chaotic |
| 14 | d8d3 | black | d3 | 440.0 | 440, 880, 1320 | 311.1 | 1.414 | False | False | chaotic |
| 15 | b7b8 | white | b8 | 880.0 | 880 | 622.3 | 1.414 | False | True | tense |
| 16 | d3h7 | black | h7 | 880.0 | 880, 1760, 2640 | 1244.5 | 1.414 | False | False | chaotic |
| 17 | b8c8 | white | c8 | 880.0 | 880 | 1244.5 | 1.414 | False | True | chaotic |
| 18 | f7g6 | black | g6 | 880.0 | 880, 1760, 2640 | 1244.5 | 1.414 | False | False | tense |
| 19 | c8e6 | white | e6 | 880.0 | 880 | 1244.5 | 1.414 | False | False | active |

### sicilian_najdorf (20 plies)
| ply | move | color | dest | pitch_hz | partials_hz | harmony_hz | ratio | check | capture | label |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | e2e4 | white | e4 | 586.7 | 587 | 829.7 | 1.414 | False | False | - |
| 2 | c7c5 | black | c5 | 586.7 | 587, 1173, 1760 | 829.7 | 1.414 | False | False | active |
| 3 | g1f3 | white | f3 | 550.0 | 550 | 777.8 | 1.414 | False | False | active |
| 4 | d7d6 | black | d6 | 825.0 | 825, 1650, 2475 | 825.0 | 1.000 | False | False | tense |
| 5 | d2d4 | white | d4 | 550.0 | 550 | 777.8 | 1.414 | False | False | active |
| 6 | c5d4 | black | d4 | 550.0 | 550, 1100, 1650 | 777.8 | 1.414 | False | True | calm |
| 7 | f3d4 | white | d4 | 550.0 | 550 | 777.8 | 1.414 | False | True | tense |
| 8 | g8f6 | black | f6 | 880.0 | 880, 1760, 2640 | 1244.5 | 1.414 | False | False | tense |
| 9 | b1c3 | white | c3 | 412.5 | 412 | 583.4 | 1.414 | False | False | active |
| 10 | a7a6 | black | a6 | 586.7 | 587, 1173, 1760 | 829.7 | 1.414 | False | False | calm |
| 11 | f1e2 | white | e2 | 412.5 | 412 | 583.4 | 1.414 | False | False | calm |
| 12 | e7e5 | black | e5 | 733.3 | 733, 1467, 2200 | 1037.1 | 1.414 | False | False | active |
| 13 | d4b3 | white | b3 | 366.7 | 367 | 518.5 | 1.414 | False | False | active |
| 14 | f8e7 | black | e7 | 880.0 | 880, 1760, 2640 | 1244.5 | 1.414 | False | False | calm |
| 15 | e1g1 | white | g1 | 412.5 | 412 | 583.4 | 1.414 | False | False | active |
| 16 | e8g8 | black | g8 | 880.0 | 880, 1760, 2640 | 1244.5 | 1.414 | False | False | active |
| 17 | c1e3 | white | e3 | 495.0 | 495 | 700.0 | 1.414 | False | False | active |
| 18 | c8e6 | black | e6 | 880.0 | 880, 1760, 2640 | 1244.5 | 1.414 | False | False | active |
| 19 | c3d5 | white | d5 | 660.0 | 660 | 933.4 | 1.414 | False | False | tense |
| 20 | e6d5 | black | d5 | 660.0 | 660, 1320, 1980 | 933.4 | 1.414 | False | True | tense |
