# B2 Organic Synthesis -- Spectral Measurements

## Spectral centroid (Hz), White vs Black, same pipeline

- **White centroid**: min=956.3, median=1690.8, max=1690.8, n=4
- **Black centroid**: min=924.1, median=924.1, max=1109.5, n=3

Black/White median centroid ratio: 0.547 (B1's Phase A measurement of the old additive-synthesis pipeline was 3.2x higher for Black; here Black is *lower*, by design -- see audio/organic_synthesis.py's BLACK_MODAL_PARAMS docstring for why).

## Energy fraction above 4000 Hz

- **White > 4000Hz**: min=0.0, median=0.0, max=0.0, n=4
- **Black > 4000Hz**: min=0.0, median=0.0, max=0.0, n=3

## Energy fraction above 8000 Hz

- **White > 8000Hz**: min=0.0, median=0.0, max=0.0, n=4
- **Black > 8000Hz**: min=0.0, median=0.0, max=0.0, n=3

## Peak / RMS (isolated Melody voice, pre-normalization)

- **White peak**: min=0.4, median=0.5, max=0.5, n=4
- **Black peak**: min=0.3, median=0.3, max=0.4, n=3
- **White RMS**: min=0.1, median=0.1, max=0.1, n=4
- **Black RMS**: min=0.1, median=0.1, max=0.1, n=3

## Per-move detail

| move | color | fundamental_hz | resonant_peaks_hz | centroid_hz | >4kHz frac | >8kHz frac | peak | rms |
|---|---|---|---|---|---|---|---|---|
| e2e4 (white) | white | 586.7 | 587, 1173, 1760, 2347 | 1129.6 | 0.0000 | 0.0000 | 0.353 | 0.059 |
| e7e5 (black) | black | 733.3 | 367, 733, 1467, 2200 | 924.1 | 0.0000 | 0.0000 | 0.381 | 0.073 |
| d1h5 (white) | white | 880.0 | 880, 1760, 2640, 3520 | 1690.8 | 0.0000 | 0.0000 | 0.531 | 0.088 |
| b8c6 (black) | black | 733.3 | 367, 733, 1467, 2200 | 924.1 | 0.0000 | 0.0000 | 0.279 | 0.054 |
| f1c4 (white) | white | 495.0 | 495, 990, 1485, 1980 | 956.3 | 0.0000 | 0.0000 | 0.389 | 0.065 |
| g8f6 (black) | black | 880.0 | 440, 880, 1760, 2640 | 1109.5 | 0.0000 | 0.0000 | 0.278 | 0.054 |
| h5f7 (white) | white | 880.0 | 880, 1760, 2640, 3520 | 1690.8 | 0.0000 | 0.0000 | 0.531 | 0.088 |

## Performance preview (B3 feasibility, not implemented)

- Whole-note offline render, White (4 modes): 6.350 ms per 1.6s note (3.969 ms per second of audio, no per-sample recursion).
- Whole-note offline render, Black (5 modes): 7.901 ms per 1.6s note (4.938 ms per second of audio, no per-sample recursion).
- Simulated single-voice (5 modes) block render at blocksize=256: 49.8 us compute for 5.80 ms of audio -> 0.858% of the block period (~117x real-time headroom for one voice; three simultaneous voices -- Melody+Harmony+Accent -- would still use only ~2.58% of the block period).
- State per active voice for the B3 port: 3 float64 arrays of length <=5 (frequencies, weights, dampings) plus a scalar onset_time -- no delay-line buffer, no per-sample recursion (contrast Karplus-Strong's O(sample_rate/frequency) delay line per note, which does not vectorize across a block). See the B2 plan's ModalVoiceState design.

## Harmony / Accent (single-move samples, for reference)

- Harmony: centroid=1055.4 Hz, peak=0.245, rms=0.038
- Accent: centroid=2512.0 Hz, peak=0.688, rms=0.101
