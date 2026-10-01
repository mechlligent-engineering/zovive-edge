"""Replay recorded wide-view footage through the mode state machine offline.

Emits the PTZ commands it would have issued, with timings, instead of driving the
camera. Lets you tune deadband, max_trigger_s and the ping-pong guards against real
footage before touching hardware.

TODO: implement.
"""
