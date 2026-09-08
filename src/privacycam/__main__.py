"""CLI entry point for PrivacyCam."""

import click
import logging
from privacycam.config import AppConfig
from privacycam.logging_config import setup_logging


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """PrivacyCam - Real-time face anonymization."""
    if ctx.invoked_subcommand is None:
        # Default: launch GUI
        ctx.invoke(run)


@main.command()
@click.option('--device', type=int, default=0, help='Camera device index')
@click.option(
    '--inference-device',
    type=str,
    default='AUTO',
    help='Inference device (NPU, NVIDIA, GPU.1, GPU.0, CPU, AUTO)',
)
@click.option(
    '--mode',
    type=click.Choice(['blur', 'pixelate', 'solid']),
    default='blur',
    help='Anonymization mode',
)
@click.option('--config', type=click.Path(exists=True), help='Path to configuration YAML')
@click.option('--resolution', type=str, default='1280x720', help='Camera resolution WxH')
@click.option('--no-preview', is_flag=True, help='Disable preview window')
@click.option('--no-virtual-camera', is_flag=True, help='Disable virtual camera output')
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--headless', is_flag=True, help='Run without GUI (OpenCV preview only)')
def run(device, inference_device, mode, config, resolution, no_preview, no_virtual_camera, debug, headless):
    """Run PrivacyCam."""
    setup_logging(level="DEBUG" if debug else "INFO")

    if config:
        app_config = AppConfig.from_yaml(config)
    else:
        app_config = AppConfig.default()

    w, h = map(int, resolution.split('x'))
    app_config.camera.device_index = device
    app_config.camera.width = w
    app_config.camera.height = h
    app_config.detection.device = inference_device
    app_config.anonymization.mode = mode
    app_config.output.preview = not no_preview
    app_config.output.virtual_camera = not no_virtual_camera

    from privacycam.app import PrivacyCamApp
    app = PrivacyCamApp(app_config)
    try:
        if headless:
            app.run_headless()
        else:
            app.run()
    except KeyboardInterrupt:
        app.stop()


@main.command()
@click.option(
    '--device',
    type=click.Choice(['CPU', 'GPU', 'NPU', 'AUTO']),
    default='AUTO',
    help='Inference device',
)
@click.option('--resolution', type=str, default='1280x720', help='Camera resolution WxH')
@click.option('--frames', type=int, default=100, help='Number of frames to benchmark')
@click.option('--faces-count', type=int, default=1, help='Simulated face count')
def benchmark(device, resolution, frames, faces_count):
    """Run performance benchmark."""
    setup_logging(level="INFO")
    click.echo(f"Benchmarking on {device} with resolution {resolution} for {frames} frames...")
    click.echo("Benchmark not yet implemented — model must be downloaded first.")


@main.command()
def detect_cameras():
    """List available cameras."""
    setup_logging(level="WARNING")
    from privacycam.capture.camera_manager import CameraManager
    from privacycam.config import CameraConfig

    manager = CameraManager(CameraConfig())
    cameras = manager.enumerate_cameras()

    if not cameras:
        click.echo("No cameras found.")
        return

    click.echo("Available cameras:")
    for cam in cameras:
        click.echo(
            f"  [{cam['index']}] {cam['name']} - "
            f"{cam['resolution'][0]}x{cam['resolution'][1]}"
        )


if __name__ == '__main__':
    main()
