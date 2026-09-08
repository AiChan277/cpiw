# Virtual Camera Architecture

PrivacyCam relies on a virtual camera output to pass the anonymized video stream to other applications like Zoom, Teams, Discord, and OBS.

## Implementation
We use `pyvirtualcam`, which provides a cross-platform abstraction over virtual camera drivers.
- **Windows**: Requires OBS Virtual Camera driver.
- **macOS**: Requires OBS Mac Virtual Camera.
- **Linux**: Requires `v4l2loopback`.

## Usage
The virtual camera starts automatically when the pipeline initializes if `virtual_camera: true` is set in the config. Other applications will see a camera named "PrivacyCam" (or default virtual cam names depending on the OS backend).

## Troubleshooting
If the camera is not showing up:
1. Ensure the required driver is installed for your OS.
2. Ensure no other application is locking the virtual camera device.
3. Check the PrivacyCam logs for `pyvirtualcam` initialization errors.
