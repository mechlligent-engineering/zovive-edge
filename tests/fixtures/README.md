# Test fixtures

`frames/` and `clips/` are for real sample images/video used by
integration tests once you have footage from the actual CP Plus camera
(day + IR, with and without animals). Empty for now — `tests/helpers.py`
generates synthetic frames/video in-memory instead, so the suite runs
with zero binary assets checked into git and zero camera required.

Drop real day/night/empty-forest clips here when available and point
`tests/test_pipeline.py`'s integration test at them via
`FileFrameSource` instead of the synthetic generator, to catch issues
the synthetic frames can't (real noise, compression artifacts, actual
animal shapes).
