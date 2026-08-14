# Measured Frequency Ranges — Phase A Diagnosis
Total moves measured: 46 across 3 games.
## Melody fundamental (destination-square pitch)
- **White fundamental**: min=412.5 Hz, median=804.5 Hz, max=2200.0 Hz, n=24
- **Black fundamental**: min=531.4 Hz, median=1443.8 Hz, max=3300.0 Hz, n=22

Both colors draw from the *same* pitch formula (`_pitch_for_square`) — the fundamental range is not color-dependent; any White/Black fundamental range difference above is purely which squares each side happened to move to in these particular games, not a systematic mapping.

## Harmonic partials actually sounded (additive_tone output)
- **White partials (all, richness=1 -> == fundamental)**: min=412.5 Hz, median=804.5 Hz, max=2200.0 Hz, n=24
- **Black partials (all, richness=3 -> fundamental+2nd+3rd)**: min=531.4 Hz, median=2590.9 Hz, max=9900.0 Hz, n=66
- Black partials > 8000 Hz: 1/66 (1.5%). White partials > 8000 Hz: 0/24.
- Black partials > 10000 Hz: 0/66 (0.0%). White partials > 10000 Hz: 0/24.

The ear's peak sensitivity to perceived harshness/piercingness sits roughly in the 2-5 kHz range (equal-loudness contours), with 8-10kHz+ content read as 'hiss'/'shrill' rather than tonal pitch. Black's 3rd partial reaching into or past this band on any move whose destination square sits in the upper half of the board's pitch range is the direct, measured mechanism for the shrillness complaint -- not a bug, but a direct consequence of summing un-capped upper harmonics on top of an already-wide (220-3520 Hz) fundamental range.

## Harmony voice
- **Harmony frequency (all moves)**: min=375.7 Hz, median=1400.1 Hz, max=4666.9 Hz, n=46
- Harmony interval ratio range: min=1.0000, max=1.4142 (production bounds: 1.0 unison .. 1.4142 tritone, 1.25 check floor). Ratio values observed are continuous/unquantized, confirming harmony is never snapped to a discrete musical interval in production.

## Renderer-level compounding factors (cited, not re-measured)
- `HARMONY_VOICE_WEIGHT = 0.6` (audio/renderer.py) -- harmony is quieter than melody by design, so the shrillness is dominated by the melody voice's own upper partials, not the harmony voice.
- `SUSTAINED_PEAK_HEADROOM = 0.85` (audio/renderer.py) -- the melody+harmony bed is normalized to a fixed peak regardless of loudness, so a Black move's dense upper-partial energy is *not* attenuated relative to a White move's; normalization does not selectively soften Black's extra harmonics.

## Per-game breakdown

### scholars_mate (7 plies)
| ply | move | color | dest | pitch_hz | partials_hz | harmony_hz | ratio | check | capture | label |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | e2e4 | white | e4 | 804.5 | 805 | 1137.8 | 1.414 | False | False | - |
| 2 | e7e5 | black | e5 | 1082.8 | 1083, 2166, 3249 | 1082.8 | 1.000 | False | False | tense |
| 3 | d1h5 | white | h5 | 1443.8 | 1444 | 2041.8 | 1.414 | False | False | tense |
| 4 | b8c6 | black | c6 | 1214.5 | 1214, 2429, 3643 | 1717.6 | 1.414 | False | False | active |
| 5 | f1c4 | white | c4 | 670.5 | 670 | 948.2 | 1.414 | False | False | active |
| 6 | g8f6 | black | f6 | 1619.3 | 1619, 3239, 4858 | 2290.1 | 1.414 | False | False | active |
| 7 | h5f7 | white | f7 | 2179.5 | 2179 | 3082.2 | 1.414 | True | True | tense |

### fastest_stalemate (19 plies)
| ply | move | color | dest | pitch_hz | partials_hz | harmony_hz | ratio | check | capture | label |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | e2e3 | white | e3 | 597.8 | 598 | 845.4 | 1.414 | False | False | - |
| 2 | a7a5 | black | a5 | 721.9 | 722, 1444, 2166 | 1020.9 | 1.414 | False | False | active |
| 3 | d1h5 | white | h5 | 1443.8 | 1444 | 2041.8 | 1.414 | False | False | chaotic |
| 4 | a8a6 | black | a6 | 971.6 | 972, 1943, 2915 | 1374.0 | 1.414 | False | False | active |
| 5 | h5a5 | white | a5 | 721.9 | 722 | 1020.9 | 1.414 | False | True | active |
| 6 | h7h5 | black | h5 | 1443.8 | 1444, 2888, 4331 | 2041.8 | 1.414 | False | False | active |
| 7 | a5c7 | white | c7 | 1634.6 | 1635 | 2311.7 | 1.414 | False | True | tense |
| 8 | a6h6 | black | h6 | 1943.2 | 1943, 3886, 5830 | 2748.1 | 1.414 | False | False | tense |
| 9 | h2h4 | white | h4 | 1072.7 | 1073 | 1517.1 | 1.414 | False | False | active |
| 10 | f7f6 | black | f6 | 1619.3 | 1619, 3239, 4858 | 2290.1 | 1.414 | False | False | active |
| 11 | c7d7 | white | d7 | 1743.6 | 1744 | 2465.8 | 1.414 | True | True | active |
| 12 | e8f7 | black | f7 | 2179.5 | 2179, 4359, 6538 | 3082.2 | 1.414 | False | False | active |
| 13 | d7b7 | white | b7 | 1471.1 | 1471 | 2080.5 | 1.414 | False | True | chaotic |
| 14 | d8d3 | black | d3 | 531.4 | 531, 1063, 1594 | 375.7 | 1.414 | False | False | chaotic |
| 15 | b7b8 | white | b8 | 1980.0 | 1980 | 1400.1 | 1.414 | False | True | tense |
| 16 | d3h7 | black | h7 | 2615.3 | 2615, 5231, 7846 | 3698.7 | 1.414 | False | False | chaotic |
| 17 | b8c8 | white | c8 | 2200.0 | 2200 | 3111.3 | 1.414 | False | True | chaotic |
| 18 | f7g6 | black | g6 | 1821.7 | 1822, 3643, 5465 | 2576.3 | 1.414 | False | False | tense |
| 19 | c8e6 | white | e6 | 1457.4 | 1457 | 2061.1 | 1.414 | False | False | active |

### sicilian_najdorf (20 plies)
| ply | move | color | dest | pitch_hz | partials_hz | harmony_hz | ratio | check | capture | label |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | e2e4 | white | e4 | 804.5 | 805 | 1137.8 | 1.414 | False | False | - |
| 2 | c7c5 | black | c5 | 902.4 | 902, 1805, 2707 | 1276.1 | 1.414 | False | False | active |
| 3 | g1f3 | white | f3 | 664.2 | 664 | 939.3 | 1.414 | False | False | active |
| 4 | d7d6 | black | d6 | 1295.5 | 1295, 2591, 3886 | 1349.1 | 1.041 | False | False | tense |
| 5 | d2d4 | white | d4 | 715.2 | 715 | 1011.4 | 1.414 | False | False | active |
| 6 | c5d4 | black | d4 | 715.2 | 715, 1430, 2145 | 1011.4 | 1.414 | False | True | calm |
| 7 | f3d4 | white | d4 | 715.2 | 715 | 1011.4 | 1.414 | False | True | tense |
| 8 | g8f6 | black | f6 | 1619.3 | 1619, 3239, 4858 | 2290.1 | 1.414 | False | False | tense |
| 9 | b1c3 | white | c3 | 498.1 | 498 | 704.5 | 1.414 | False | False | active |
| 10 | a7a6 | black | a6 | 971.6 | 972, 1943, 2915 | 1374.0 | 1.414 | False | False | calm |
| 11 | f1e2 | white | e2 | 444.1 | 444 | 628.1 | 1.414 | False | False | calm |
| 12 | e7e5 | black | e5 | 1082.8 | 1083, 2166, 3249 | 1531.4 | 1.414 | False | False | active |
| 13 | d4b3 | white | b3 | 448.3 | 448 | 634.0 | 1.414 | False | False | active |
| 14 | f8e7 | black | e7 | 1961.5 | 1962, 3923, 5885 | 2774.0 | 1.414 | False | False | calm |
| 15 | e1g1 | white | g1 | 412.5 | 412 | 583.4 | 1.414 | False | False | active |
| 16 | e8g8 | black | g8 | 3300.0 | 3300, 6600, 9900 | 4666.9 | 1.414 | False | False | active |
| 17 | c1e3 | white | e3 | 597.8 | 598 | 845.4 | 1.414 | False | False | active |
| 18 | c8e6 | black | e6 | 1457.4 | 1457, 2915, 4372 | 2061.1 | 1.414 | False | False | active |
| 19 | c3d5 | white | d5 | 962.5 | 963 | 1361.2 | 1.414 | False | False | tense |
| 20 | e6d5 | black | d5 | 962.5 | 963, 1925, 2888 | 1361.2 | 1.414 | False | True | tense |
