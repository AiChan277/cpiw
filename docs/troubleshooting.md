# Troubleshooting

## Common Issues

### Camera Not Found
- Ensure no other application (like Zoom) is currently using your webcam.
- Check the `device_id` in your YAML config. If `0` fails, try `1` or `2`.

### NPU Unavailable
- Ensure you have a compatible Intel CPU (Meteor Lake or newer).
- Ensure Intel NPU drivers are installed and up to date.
- OpenVINO will automatically fallback to GPU or CPU if NPU fails.

### Virtual Camera Not Showing in Apps
- Install OBS Studio and start its Virtual Camera once to ensure the driver is installed.
- On Windows, ensure you run the installation scripts for the OBS virtual cam driver if prompted.

### Performance Issues (Lag, High Latency)
- Lower the camera resolution in the config (e.g., 640x480).
- Switch the anonymization mode to something simpler like `pixelate`.
- Increase the `interval` under the `detection` config so full ML detection runs less often.
