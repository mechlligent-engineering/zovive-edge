"""Trigger arbitration tests.

- camera motion and software motion within cooldown produce ONE event
- the safety sweep fires during cooldown and is never suppressed
- gate_source is recorded correctly for each path
- a remote trigger preempts everything
"""
