# NPU Architecture

PrivacyCam is optimized for Intel Neural Processing Units (NPUs) present in Meteor Lake and newer processors.

## Why NPU?
NPUs are dedicated hardware blocks for highly efficient AI processing. By running the face detection OpenVINO model on the NPU:
- CPU utilization drops significantly.
- Battery life improves on laptops.
- Dedicated hardware guarantees stable latency, immune to heavy CPU loads.

## OpenVINO Runtime
We use the Intel OpenVINO runtime to execute models. The configuration device string is typically `"AUTO"`, which allows OpenVINO to select the best available hardware (prioritizing NPU, then GPU, then CPU). You can force NPU usage by setting `device: "NPU"` in the configuration.

## Model Compilation & Warm-up
NPU execution requires compiling the OpenVINO model specifically for the hardware on first run. This might cause a slight delay (a few seconds) on application startup. We perform a "warm-up" inference run during startup so that real-time latency is minimal once video processing begins.
