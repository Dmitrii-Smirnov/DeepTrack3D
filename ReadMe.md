# Video Processing and Analysis Pipeline (DeepTrack3D)

## Table of Contents
1.  [Description](#description)
2.  [Features](#features)
3.  [Project Structure](#project-structure)
4.  [Prerequisites](#prerequisites)
5.  [Setup and Installation](#setup-and-installation)
6.  [Configuration (`config.yaml`)](#configuration-configyaml)
7.  [Usage](#usage)
8.  [Output](#output)
9.  [Modules Overview](#modules-overview)
10. [Troubleshooting](#troubleshooting)
11. [Contributing](#contributing)
12. [License](#license)
13. [Future Enhancements](#future-enhancements)

## 1. Description
This project provides an automated pipeline for processing video files. It performs a series of operations including video format conversion, slicing into segments, video stabilization, and detailed frame-by-frame analysis. The analysis extracts object tracking information, pose estimation keypoints, and depth information for detected objects (primarily persons), saving the results to CSV files. It also generates visualized videos from the analysis step.

The application is designed with an Object-Oriented approach for modularity and maintainability.

## 2. Features
* **Video Conversion**: Converts various video formats (e.g., `.MOV`, `.MTS`) to `.MP4` and allows FPS adjustment.
* **Video Slicing**: Splits videos into smaller, manageable segments of a configurable duration (e.g., 5-minute clips).
* **Video Stabilization**: Stabilizes shaky video segments using the `vidstab` library.
* **Object Tracking & Pose Estimation**: Utilizes YOLO (e.g., `yolov11x-pose`) for detecting and tracking objects (persons by default) and estimating their pose keypoints.
* **Depth Estimation**: Employs MiDaS models to estimate depth maps for video frames.
* **Data Extraction**: Calculates bounding boxes, centroids (with depth), and 3D keypoints (x, y, depth) for tracked objects.
* **CSV Output**: Saves extracted data frame-by-frame into structured CSV files for further analysis.
* **Visualized Output**: YOLO tracking saves processed videos/frames with visualizations.
* **Configurable Pipeline**: All major parameters, paths, and model choices are configurable via a `config.yaml` file.
* **Modular Design**: Built with Python classes for each processing step, promoting reusability and extensibility.
* **Command-Line Interface**: Allows specifying the input video file and configuration file directly via CLI.
* **Automated Dummy Setup**: `main.py` can include helpers to create dummy input files for easier first-time setup and testing.

## 3. Project Structure
```
DeepTrack3D/
│
├── main.py                     # Main script to run the pipeline
├── config.yaml                 # Configuration file for pipeline parameters
├── requirements.txt            # Python package dependencies
│
├── core_logic/                 # Central classes for pipeline orchestration and analysis
│   ├── init.py
│   ├── pipeline.py             # Defines VideoProcessingPipeline
│   ├── video_analyzer.py       # Defines VideoAnalyzer (pose/depth extraction)
│   └── model_manager.py        # Defines ModelManager (ML model loading)
│
├── processing_steps/           # Modules for individual video processing operations
│   ├── init.py
│   ├── video_converter.py      # Defines VideoConverter
│   ├── video_slicer.py         # Defines VideoSlicer
│   └── video_stabilizer.py     # Defines VideoStabilizer
│
├── utils/                      # Utility modules and helper functions
│   ├── init.py
│   ├── data_writer.py          # Defines CSVDataWriter
│   └── video_utils.py          # Video-related helper functions (e.g., get_fps)
│
├── data/                       # Directory for input data and non-code assets
│   ├── input_videos/           # Place your input videos here
│   │   └── your_video.mov      # Example placeholder (or default_dummy.mov)
│   ├── bytetrack.yaml          # Tracker configuration (e.g., for ByteTrack)
│   └── models/                 # Optional: For manually downloaded ML models
│       └── yolov11x-pose.pt     # Example model
│
├── output_data/                # Default base output folder for all generated files
│   ├── 1_converted_videos/
│   ├── 2_sliced_videos/
│   ├── 3_stabilized_videos/
│   ├── 4_csv_data/
│   └── 5_yolo_processed_videos/ # Or similar name based on YOLO output structure
│
└── README.md                   # This file
└── https://www.google.com/search?q=LICENSE                     # Project license file
```


## 4. Prerequisites
* **Python**: Version 3.8 or newer (Python 3.10.x recommended).
* **FFmpeg**: Must be installed and accessible in your system's PATH (for video conversion, slicing, and FPS detection). `ffprobe` (usually included with FFmpeg) is also required.
* **Python Libraries**: All required Python packages are listed in `requirements.txt`.

## 5. Setup and Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Dmitrii-Smirnov/DeepTrack3D.git # Replace with your actual repository URL
    cd DeepTrack3D
    ```

2.  **Install FFmpeg**:
    * **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install ffmpeg`
    * **macOS (using Homebrew)**: `brew install ffmpeg`
    * **Windows**: Download binaries from the [FFmpeg official website](https://ffmpeg.org/download.html) and add the `bin` directory to your system's PATH.
    Verify installation by running `ffmpeg -version` and `ffprobe -version` in your terminal.

3.  **Create and Activate a Python Virtual Environment (Recommended)**:
    ```bash
    python3 -m venv venv  # Or python -m venv venv
    ```
    #### On Windows:
    ```
    venv\Scripts\activate
    ```
    ##### On macOS/Linux:
    ```
    source venv/bin/activate
    ```

4.  **Install Python Dependencies**:
    Ensure you have a `requirements.txt` file in the project root. It should contain:
    ```txt
    # requirements.txt
    PyYAML>=6.0
    torch>=2.0.0
    torchvision>=0.15.0
    ultralytics>=8.0.190 # Example version, pin to a known working version if issues arise
    moviepy>=1.0.3
    opencv-python>=4.7.0
    numpy>=1.23.0
    vidstab>=1.0.2
    timm>=0.9.0 # Often a dependency for MiDaS or related vision models
    ```
    With your virtual environment activated, install the packages:
    ```bash
    pip install -r requirements.txt
    ```
    **PyTorch Note**: For GPU support, ensure you have the correct NVIDIA drivers and CUDA toolkit installed. Then, install the PyTorch version compatible with your CUDA setup by visiting the [PyTorch website](https://pytorch.org/get-started/locally/) and using their recommended `pip` command.

5.  **Prepare Data and Configuration Files**:
    * **Input Videos**: Place your videos in the `data/input_videos/` directory. The script can create a dummy video here for a first run if the configured video is not found.
    * **Tracker Configuration**: Ensure your tracker configuration file (e.g., `bytetrack.yaml`, as specified in `config.yaml`) is present in the `data/` directory. A minimal version, **critically including `fuse_score: true` (or `false`)**, will be auto-generated by `main.py` if not found. This parameter is vital for certain Ultralytics versions.
        Example `data/bytetrack.yaml`:
        ```yaml
        tracker_type: bytetrack
        track_high_thresh: 0.50
        track_low_thresh: 0.10
        new_track_thresh: 0.60
        track_buffer: 30 # Corresponds to args.track_buffer in tracker code
        match_thresh: 0.80
        # For Ultralytics versions that require it (e.g. >8.0.190), ensure fuse_score is present.
        # Check Ultralytics documentation for your specific version if errors occur.
        fuse_score: true
        ```
    * **ML Models**: YOLO and MiDaS models are typically downloaded automatically if standard names are used in `config.yaml`. For manual model placement (e.g., in `data/models/`), update paths in `config.yaml`.
    * **Configuration File**: Review and modify `config.yaml` in the project root to suit your needs.

## 6. Configuration (`config.yaml`)
The main behavior of the pipeline is controlled by `config.yaml`. Key parameters:
* `input_video_path`: Default input video path (can be overridden by CLI). Path relative to project root or absolute.
* `base_output_folder`: Root directory for all generated outputs.
* `tracker_config_file`: Path to the tracker configuration file (e.g., `data/bytetrack.yaml`).
* `target_fps`: Target FPS for converted videos. Use `null` or omit the key to keep original FPS if the input is MP4 and no FPS change is desired.
* `segment_duration_seconds`: Video segment duration in seconds.
* `yolo_model_name`: YOLO model name or path (e.g., `yolov11x-pose.pt`).
* `midas_model_type`: MiDaS model type (e.g., `DPT_Large`).
* `yolo_confidence`, `yolo_iou_threshold`, `yolo_classes`: YOLO detection and NMS parameters.
* `use_half_precision`: Boolean for FP16 inference on compatible GPUs.
* `frame_downscale_factor`: Factor to downscale frames for MiDaS processing (e.g., `1.0` for original, `0.5` for half).
* `cleanup_intermediate_folders`: Optional boolean to delete intermediate folders (converted, sliced, stabilized) after successful pipeline completion.

*(Suggestion: For local path overrides without modifying the main `config.yaml`, consider copying it to `config.local.yaml` and adding `config.local.yaml` to your `.gitignore`. You would then run `python main.py --config_file config.local.yaml`.)*

## 7. Usage
Ensure your Python virtual environment is activated. Navigate to the project's root directory (`DeepTrack3D/`) in your terminal.

**A. Using Default Configuration (and default video path from `config.yaml`):**
```bash
python main.py
```
B. Specifying an Input Video via Command Line:
To process a specific video file, overriding the input_video_path in config.yaml, use the -i or --input_video argument:

```bash
python main.py -i /path/to/your/specific_video.mov
```
##### or
```bash
python main.py --input_video data/input_videos/another_video.mp4
```
C. Specifying a Custom Configuration File:
Use the ```-c``` or ```--config_file argument (path relative to project root or absolute)```:

```bash
python main.py -c path/to/my_custom_settings.yaml
```
D. Combining Command-Line Arguments:

```bash
python main.py -i /path/to/your/specific_video.mov -c my_custom_settings.yaml
```
The script will print progress messages, and outputs will be saved in the directory specified by base_output_folder.

8. Output
The pipeline generates outputs organized into subdirectories within the base_output_folder (default output_data/):

1_converted_videos/: Converted MP4 videos.
2_sliced_videos/: Sliced video segments.
3_stabilized_videos/: Stabilized video segments.
4_csv_data/: CSV files with extracted pose and depth data for each processed segment.
5_yolo_processed_videos/: Visualized frames/videos from YOLO tracking. The exact structure within this folder (e.g., subfolders per run) is determined by Ultralytics.
9. Modules Overview
main.py: Application entry point, handles CLI arguments and initiates the pipeline.
config.yaml: User-configurable parameters for the pipeline.
core_logic/:
pipeline.py (VideoProcessingPipeline): Orchestrates the sequence of processing steps.
video_analyzer.py (VideoAnalyzer / PoseDepthTracker): Handles ML-based analysis (pose estimation, depth calculation).
model_manager.py (ModelManager): Manages the loading and preparation of ML models (YOLO, MiDaS).
processing_steps/:
video_converter.py (VideoConverter / VideoAsset): Handles video format/FPS conversion and basic video properties.
video_slicer.py (VideoSlicer): Responsible for video segmentation.
video_stabilizer.py (VideoStabilizer): Performs video stabilization.
utils/:
data_writer.py (CSVDataWriter): Handles writing extracted data to CSV files.
video_utils.py: Contains miscellaneous video-related helper functions.
10. Troubleshooting
AttributeError: ... has no attribute 'track_buffer' or 'fuse_score' (Ultralytics):
Ensure your tracker configuration file (e.g., data/bytetrack.yaml) explicitly contains the required parameters like track_buffer and fuse_score: true (or false). The exact parameters needed can vary between Ultralytics versions. Refer to the Ultralytics documentation for your specific version.
Update Ultralytics: pip install -U ultralytics.
If issues persist, consider replacing the default.yaml file in your ultralytics package installation with the latest from their GitHub repository (see Ultralytics documentation or previous error messages for paths). Back up your existing file first!
FFmpeg/ffprobe not found: Ensure FFmpeg is installed correctly and its bin directory is added to your system's PATH environment variable.
Model Download Issues: Check your internet connection. For MiDaS, torch.hub.load might require trust_repo=True if you encounter issues with SSL or model fetching. For manual model placement, verify paths in config.yaml.
CUDA/GPU Issues: Verify NVIDIA drivers, CUDA toolkit, and PyTorch compatibility. The script should default to CPU if CUDA isn't detected or properly configured.
File Not Found Errors: Double-check all paths specified in config.yaml and any command-line arguments. Ensure they are correct relative to the project root or are absolute paths.
Permission Errors: Ensure your user has write permissions for the base_output_folder and its subdirectories.
MoviePy errors ('NoneType' object has no attribute 'stdout'): This can indicate an underlying ffmpeg failure during video writing. Enable MoviePy's logger (logger='bar' in write_videofile calls) for more detailed ffmpeg output. Check for sufficient disk space and valid segment durations.
11. Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for bugs, feature requests, or improvements.
Before contributing, please:

Fork the repository.
Create a new branch for your feature or bug fix.
Ensure your code adheres to the project's coding style (e.g., use a linter like Flake8 or Black).
Write clear and concise commit messages.
Open a pull request with a detailed description of your changes.
12. License
This project is licensed under the MIT License.

MIT License

Copyright (c) 2025 Dmitrii Smirnov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
13. Future Enhancements
Implement a more robust logging system (e.g., Python's logging module) throughout the application.
Add more sophisticated error handling, retries, and recovery mechanisms for pipeline steps.
Support for additional output formats (e.g., JSON, Parquet).
Implement parallel processing for video segments to speed up the pipeline on multi-core systems.
Develop a more comprehensive suite of unit and integration tests.
Add options for different trackers beyond ByteTrack.
GUI for easier configuration and execution.
<!-- end list -->