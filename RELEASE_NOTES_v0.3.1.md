# Arabic Doc AI v0.3.1 — Decision Engine 2.0

## Why this release exists

The v0.3.0 benchmark exposed a ranking failure: baseline-text similarity and line-count similarity rewarded candidates for resembling the original OCR output. That can penalize a transformation that corrects an OCR mistake. On the supplied test image, perspective correction achieved lower reference CER/WER than the original, even though the runtime heuristic selected the original.

## Changes

- Removed baseline text agreement and line-count similarity from the heuristic score. They remain diagnostics only.
- Rebalanced the transparent runtime heuristic across confidence, text integrity, line quality, and geometry coverage/overlap.
- Reduced the per-operation penalty; transformations still need to clear the configured improvement margin.
- Added strict CER/WER and conservative Arabic-normalized CER/WER.
- Normalized metrics fold Arabic/Persian digit forms, tatweel, diacritics, Unicode presentation forms, and common alef variants. They deliberately preserve distinctions such as ى vs ي.
- Added benchmark-only oracle ranking by CER/WER to `compare_preprocessing.py`. Oracle metrics are never fed into runtime selection.
- Added explicit report language warning that an inaccurate ground-truth transcription can mis-rank candidates.
- Version metadata updated to 0.3.1.

## Important limitations

- Runtime selection is still a heuristic. It is not a calibrated probability and is not guaranteed to choose the lowest-CER candidate.
- A manually transcribed reference must be checked independently against the source. The previously used identity-card transcription was drafted from a low-resolution image and should be treated as provisional until verified.
- One image is not enough to train a reliable candidate-selection model. Collect diverse documents and split by document/template before training.
