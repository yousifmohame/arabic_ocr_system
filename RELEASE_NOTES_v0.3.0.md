# Arabic Doc AI v0.3.0

## Intelligent Preprocessing

- Original image is always OCR-processed first and retained as the fallback.
- Added bounded image-quality analysis: brightness, contrast, percentile range, Laplacian sharpness, broad illumination variation, edge density, Hough skew estimate and document-ROI metrics.
- Added adaptive candidate planner for perspective correction, deskew, local enhancement and upscaling.
- Candidates are processed sequentially using the already-loaded OCR engine; no parallel model copies are created.
- Added transparent heuristic candidate scoring with a conservative improvement margin and a bias toward the source image on near-ties.
- Added optional per-page ground-truth CER/WER reporting. Ground truth never affects automatic candidate selection.
- Added a v0.3 benchmark script that reports candidate metrics, elapsed time and peak process-tree RSS.
- Added tests for quality analysis, skew detection, ROI masking, candidate selection, pipeline decisions and coordinate preservation.

## Important limitations

- The heuristic candidate score is not calibrated OCR accuracy. Validate the selected result against manually transcribed data.
- Image preprocessing is not guaranteed to improve every document.
- This release has passed unit tests in the build environment, but the real PaddleOCR model inference and RAM peak must still be checked on the user's Mac.
- Model weights are intentionally not included in this source archive. Existing local `models/` weights must be preserved.
