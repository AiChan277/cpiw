# Privacy Architecture

PrivacyCam is built from the ground up to respect user privacy completely. 

## Design Principles
1. **Local Only**: All computation, including ML inference, is performed locally on the user's hardware.
2. **Ephemeral Data**: Video frames are processed in RAM and immediately overwritten. No video is saved to disk.
3. **No Biometrics**: The models used perform Object Detection (Faces). They do *not* perform facial recognition, embedding generation, or feature extraction that could identify an individual.
4. **No Telemetry**: We do not collect analytics, crash logs, or usage statistics over the network.

## Threat Model
The application assumes the host OS is trusted. It protects against casual observation and automated mass-surveillance from video conferencing software by anonymizing faces *before* the video stream reaches those applications.
