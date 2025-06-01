import yaml  # PyYAML for loading config.yaml
import os
import sys
import subprocess  # For creating dummy files/dirs if needed for first run
import torch  # To check for CUDA availability for default half_precision
import argparse  # For command-line arguments

# Add the project root to Python's module search path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can use direct imports from our packages
from core_logic.pipeline import VideoProcessingPipeline


def load_config(config_path="config.yaml") -> dict:
    """Loads configuration from a YAML file."""
    abs_config_path = os.path.join(project_root, config_path)
    if not os.path.exists(abs_config_path):
        print(f"Error: Configuration file not found at {abs_config_path}")
        print("Please create a 'config.yaml' file in the application root directory.")
        print("See documentation or previous examples for its structure.")
        sys.exit(1)  # Exit if no config file

    try:
        with open(abs_config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from {abs_config_path}")
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file {abs_config_path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading config {abs_config_path}: {e}")
        sys.exit(1)


def setup_dummy_data_for_first_run(config: dict):
    """
    Creates dummy input video (at config path) and tracker config if they don't exist.
    This helps with a smoother first run if the user hasn't set up their own data yet,
    using the paths defined in the config file.
    """
    print("\n--- First Run Setup Check (using config paths) ---")

    # 1. Input Video (based on config path)
    # ... (this part remains the same) ...
    config_video_path_rel = config.get("input_video_path", "data/input_videos/default_dummy.mov")
    config_video_path_abs = os.path.join(project_root, config_video_path_rel)
    os.makedirs(os.path.dirname(config_video_path_abs), exist_ok=True)

    if not os.path.exists(config_video_path_abs):
        print(f"Configured input video '{config_video_path_abs}' not found.")
        try:
            print(f"Attempting to create a dummy video at: {config_video_path_abs} (requires ffmpeg in PATH)")
            subprocess.run([
                'ffmpeg', '-y', '-f', 'lavfi', '-i', 'testsrc=duration=3:size=320x240:rate=10',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-t', '3', config_video_path_abs
            ], check=True, capture_output=True, text=True)
            print(f"Dummy video successfully created: {config_video_path_abs}")
            print(f"Note: This dummy video is based on the 'input_video_path' in your config.yaml.")
        except FileNotFoundError:
            print(
                "ffmpeg not found. Cannot create dummy video. Please ensure ffmpeg is installed or place your video at the configured path.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to create dummy video using ffmpeg. Error: {e.stderr}")
            print(f"Please manually place a video file at: {config_video_path_abs} (as defined in config.yaml)")
        except Exception as e:
            print(f"An unexpected error occurred creating dummy video: {e}")
    else:
        print(f"Configured input video found: {config_video_path_abs}")


    # 2. Tracker Configuration (e.g., bytetrack.yaml)
    tracker_config_rel = config.get("tracker_config", "data/bytetrack.yaml")
    tracker_config_abs = os.path.join(project_root, tracker_config_rel)
    os.makedirs(os.path.dirname(tracker_config_abs), exist_ok=True)

    if not os.path.exists(tracker_config_abs):
        print(f"Tracker config '{tracker_config_abs}' not found. Creating a minimal dummy one.")
        # MODIFIED DUMMY CONTENT:
        dummy_tracker_content = """tracker_type: bytetrack
                                   track_high_thresh: 0.50
                                   track_low_thresh: 0.10
                                   new_track_thresh: 0.60
                                   track_buffer: 30
                                   match_thresh: 0.80
                                   fuse_score: true 
                                """
        try:
            with open(tracker_config_abs, "w") as f:
                f.write(dummy_tracker_content)
            print(f"Dummy tracker config created with 'fuse_score: true': {tracker_config_abs}")
        except IOError as e:
            print(f"Could not write dummy tracker config to {tracker_config_abs}: {e}")
    else:
        print(f"Tracker config found: {tracker_config_abs}")

    # Ensure base_output_folder exists
    # ... (this part remains the same) ...
    base_output_folder_rel = config.get("base_output_folder", "output_data")
    base_output_folder_abs = os.path.join(project_root, base_output_folder_rel)
    os.makedirs(base_output_folder_abs, exist_ok=True)
    print(f"Output directory ensured: {base_output_folder_abs}")
    print("--- First Run Setup Check Complete ---\n")



def main():
    parser = argparse.ArgumentParser(description="Video Processing and Analysis Pipeline.")
    parser.add_argument(
        "-i", "--input_video",
        type=str,
        help="Path to the input video file. Overrides 'input_video_path' in config.yaml.",
        default=None  # Default is None, meaning we'll use the config file's path
    )
    parser.add_argument(
        "-c", "--config_file",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (relative to project root). Default: config.yaml"
    )
    args, unknown_args = parser.parse_known_args()  # MODIFIED LINE

    # Optional: You can print the unknown arguments if you want to see what they are
    if unknown_args:
        print(f"Ignoring unrecognized arguments: {unknown_args}")

    print("Starting Video Processing Application...")

    # Load configuration
    config = load_config(args.config_file)

    # (Optional) Setup dummy data based on config paths if running for the first time
    setup_dummy_data_for_first_run(config)

    # Determine the input video path: CLI argument takes precedence
    actual_input_video_path = None
    if args.input_video:
        print(f"Command-line argument for input video provided: {args.input_video}")
        # Convert CLI path to absolute path, assuming it might be relative to CWD
        cli_video_path_abs = os.path.abspath(args.input_video)
        if not os.path.exists(cli_video_path_abs):
            print(f"Error: Input video specified via command line not found: {cli_video_path_abs}")
            sys.exit(1)
        actual_input_video_path = cli_video_path_abs
        config["input_video_path_original_config"] = config.get(
            "input_video_path")  # Store original for reference if needed
        config["input_video_path"] = actual_input_video_path  # Override config
        print(f"Using video from command line: {actual_input_video_path}")
    else:
        print("No command-line input video provided. Using path from config file.")
        # Use path from config, ensure it's absolute relative to project root
        config_video_path_rel = config.get("input_video_path")
        if not config_video_path_rel:
            print(f"Error: 'input_video_path' not found in {args.config_file} and no CLI argument provided.")
            sys.exit(1)
        actual_input_video_path = os.path.join(project_root, config_video_path_rel)
        if not os.path.exists(actual_input_video_path):
            print(f"Error: Input video specified in config file not found: {actual_input_video_path}")
            print(
                f"Please ensure the video exists or run the script once to allow dummy video creation at this location.")
            sys.exit(1)
        config["input_video_path"] = actual_input_video_path  # Ensure it's absolute
        print(f"Using video from config: {actual_input_video_path}")

    # Update other config paths to be absolute for robustness
    config["base_output_folder"] = os.path.join(project_root, config.get("base_output_folder", "output_data"))
    config_tracker_path = config.get("tracker_config", "data/bytetrack.yaml")
    config["tracker_config"] = os.path.join(project_root, config_tracker_path)
    if not os.path.exists(config["tracker_config"]):
        print(f"Warning: Tracker configuration file not found at {config['tracker_config']}. Analysis might fail.")

    # Auto-set half_precision based on CUDA if not explicitly false in config
    # Check if 'half_precision' key exists and is explicitly set to False
    if config.get("half_precision") is False:
        config["half_precision"] = False
        print("half_precision explicitly set to False in config.")
    else:  # If true, null/None, or not present, base it on CUDA availability
        config["half_precision"] = torch.cuda.is_available()
        if config["half_precision"]:
            print("CUDA available, half_precision enabled.")
        else:
            print("CUDA not available, half_precision disabled.")

    # Create and run the pipeline
    try:
        pipeline = VideoProcessingPipeline(config=config)  # Pass the potentially modified config
        pipeline.run()
    except RuntimeError as e:
        print(f"A critical runtime error occurred in the pipeline: {e}")
        print("This might be due to model loading failures or other essential components not initializing.")
    except Exception as e:
        print(f"An unexpected error occurred during pipeline execution: {e}")
        import traceback
        traceback.print_exc()

    print("\nVideo Processing Application finished.")


if __name__ == "__main__":
    main()