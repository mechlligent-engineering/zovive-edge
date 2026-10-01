"""pytest fixtures, for the case where you *do* have pytest available
(your laptop, CI) and want fixture-style tests instead of unittest
classes. Not required by any test in this repo — every test here uses
stdlib unittest so `python3 -m unittest discover -s tests` always
works with zero installs — but pytest will pick these up automatically
if you add fixture-style tests later.
"""

from __future__ import annotations

import pytest

from tests.helpers import blank_frame, frame_with_box


@pytest.fixture
def sample_frame():
    return blank_frame()


@pytest.fixture
def sample_frame_with_box():
    return frame_with_box()
