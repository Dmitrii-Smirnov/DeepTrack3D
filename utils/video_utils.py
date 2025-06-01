import subprocess
import os

def get_video_fps_ffprobe(input_video_path: str) -> float:
    """
    Retrieves the frame rate (FPS) of a given video file using ffprobe.
    Returns 0.0 if FPS cannot be determined or an error occurs.
    """
    if not os.path.exists(input_video_path):
        print(f"Error: Video file not found at {input_video_path}")
        return 0.0

    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate', '-of',
        'default=noprint_wrappers=1:nokey=1', input_video_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        fps_str = result.stdout.strip()
        if '/' in fps_str:
            num, denom = map(int, fps_str.split('/'))
            return num / denom if denom != 0 else 0.0
        return float(fps_str)
    except subprocess.CalledProcessError as e:
        print(f"Error getting FPS for {input_video_path} via ffprobe. Command: '{' '.join(e.cmd)}'")
        print(f"FFprobe stderr: {e.stderr}")
        return 0.0
    except FileNotFoundError:
        print("Error: ffprobe command not found. Please ensure ffmpeg (which includes ffprobe) is installed and in your system's PATH.")
        return 0.0
    except ValueError as e:
        print(f"Error parsing FPS string '{fps_str}' for {input_video_path}: {e}")
        return 0.0
    except Exception as e:
        print(f"An unexpected error occurred while getting FPS for {input_video_path}: {e}")
        return 0.0

# You can add other video-related utility functions here.