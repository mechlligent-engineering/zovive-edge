# Testing

Everything in `pipeline/` and `storage/` must be testable on a laptop with no camera,
no NPU and no base station. That is why the fixtures exist. If a module cannot be tested
without hardware, the hardware-touching part belongs in `inference/` or `capture/`.
