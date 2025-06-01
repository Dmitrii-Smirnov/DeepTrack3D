import subprocess
import os
from utils.video_utils import get_video_fps_ffprobe  # Assuming utils is in PYTHONPATH or same level


class VideoConverter:
    def __init__(self, preset: str = 'slow', crf: int = 18,
                 audio_codec: str = 'aac', audio_bitrate: str = '192k'):
        self.preset = preset
        self.crf = crf
        self.audio_codec = audio_codec
        self.audio_bitrate = audio_bitrate

    def convert(self, input_video_path: str, output_video_path: str, target_fps: int = None) -> str | None:
        """Converts videos to .MP4, optionally adjusting FPS."""
        print(f"Starting conversion for: {input_video_path} to {output_video_path}")

        if not os.path.exists(input_video_path):
            print(f"Error: Input video for conversion not found at {input_video_path}")
            return None

        original_fps = get_video_fps_ffprobe(input_video_path)  # From video_utils
        if original_fps > 0:
            print(f"Original video FPS: {original_fps:.2f}")
        else:
            print(f"Warning: Could not determine original FPS for {input_video_path}. Conversion will proceed.")

        command = [
            'ffmpeg', '-y',  # Overwrite output without asking
            '-i', input_video_path,
            '-c:v', 'libx264',
            '-preset', self.preset,
            '-crf', str(self.crf),
            '-c:a', self.audio_codec,
            '-b:a', self.audio_bitrate,
            '-strict', 'experimental'  # For some AAC versions/FFmpeg builds
        ]

        if target_fps and isinstance(target_fps, (int, float)) and target_fps > 0:
            command.extend(['-r', str(target_fps)])
            print(f"Target FPS set to: {target_fps}")
        elif target_fps:
            print(f"Warning: Invalid target_fps value ({target_fps}). Original FPS will be used if possible.")

        command.append(output_video_path)

        try:
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            # Using capture_output=True to get stderr if needed
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Conversion successful: {output_video_path}")
            return output_video_path
        except subprocess.CalledProcessError as e:
            print(f"Error during FFMPEG conversion for '{input_video_path}':")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"FFMPEG Stderr: {e.stderr}")
            return None
        except FileNotFoundError:
            print("Error: ffmpeg command not found. Please ensure ffmpeg is installed and in your system's PATH.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during video conversion: {e}")
            return None