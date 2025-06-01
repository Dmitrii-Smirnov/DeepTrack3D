import os
import shutil
import torch  # For torch.cuda.is_available() for default half_precision

# Assuming current file is in core_logic, and sibling directories are processing_steps, utils
from core_logic.model_manager import ModelManager
from core_logic.video_analyzer import VideoAnalyzer
from processing_steps.video_converter import VideoConverter
from processing_steps.video_slicer import VideoSlicer
from processing_steps.video_stabilizer import VideoStabilizer


# CSVDataWriter is used internally by VideoAnalyzer, so not directly here unless needed elsewhere by pipeline

class VideoProcessingPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.base_output_folder = os.path.abspath(config.get("base_output_folder", "./output_data_pipeline"))
        os.makedirs(self.base_output_folder, exist_ok=True)
        print(f"Pipeline output will be in: {self.base_output_folder}")

        # Initialize components based on config
        self.converter = VideoConverter()  # Uses default params unless specified and passed
        self.slicer = VideoSlicer(segment_duration_seconds=config.get("segment_duration", 300))
        self.stabilizer = VideoStabilizer(border_type=config.get("stabilization_border_type", "black"))

        self.model_manager = ModelManager(
            yolo_model_path=config.get("yolo_model_path", "yolo11x-pose.pt"),
            midas_model_type=config.get("midas_model_type", "DPT_Large"),
            yolo_conf=float(config.get("yolo_conf", 0.5)),
            yolo_iou=float(config.get("yolo_iou", 0.45)),
            half_precision=config.get("half_precision", torch.cuda.is_available())
        )

        self.analyzer = VideoAnalyzer(
            model_manager=self.model_manager,
            tracker_config_path=config.get("tracker_config", "data/bytetrack.yaml"),  # Ensure path is correct
            yolo_classes=config.get("yolo_classes", [0]),
            downscale_factor=float(config.get("downscale_factor", 1.0))
        )
        self.intermediate_paths_to_cleanup = []

    def _get_output_path_for_step(self, step_name: str, *subdirs) -> str:
        """Helper to create a unique path for a processing step's output."""
        path = os.path.join(self.base_output_folder, step_name, *subdirs)
        os.makedirs(path, exist_ok=True)
        return path

    def run(self):
        input_video_raw = self.config.get("input_video_path")
        if not input_video_raw or not os.path.exists(input_video_raw):
            print(
                f"Error: Input video path '{input_video_raw}' not provided or file does not exist. Aborting pipeline.")
            return

        input_video_abs = os.path.abspath(input_video_raw)
        print(f"Starting pipeline for video: {input_video_abs}")
        base_name_original = os.path.splitext(os.path.basename(input_video_abs))[0]

        current_processed_video_path = input_video_abs
        is_original_file_being_processed = True  # Flag if current_processed_video_path is the initial input

        # --- 1. Convert Video (if necessary, or to standardize format/fps) ---
        target_fps = self.config.get("target_fps")
        needs_conversion = not input_video_abs.lower().endswith(".mp4") or \
                           (target_fps is not None and isinstance(target_fps, (int, float)) and target_fps > 0)

        if needs_conversion:
            print("\nStep 1: Converting video...")
            conversion_output_dir = self._get_output_path_for_step("1_converted_videos")
            converted_video_filename = f"{base_name_original}_converted.mp4"
            converted_video_path = os.path.join(conversion_output_dir, converted_video_filename)

            conversion_result_path = self.converter.convert(
                current_processed_video_path, converted_video_path, target_fps=target_fps
            )
            if conversion_result_path:
                current_processed_video_path = conversion_result_path
                is_original_file_being_processed = False
                self.intermediate_paths_to_cleanup.append(conversion_output_dir)  # Cleanup entire dir later
            else:
                print(
                    f"Video conversion failed for {current_processed_video_path}. Attempting to proceed with original/previous version.")
        else:
            print("\nStep 1: Skipping video conversion (input is MP4 and no target_fps, or target_fps is invalid).")

        # --- 2. Slice Video ---
        print("\nStep 2: Slicing video...")
        # Sliced videos will be based on the name of the video entering this step
        slicing_input_basename = os.path.splitext(os.path.basename(current_processed_video_path))[0]
        sliced_videos_base_dir = self._get_output_path_for_step("2_sliced_videos", slicing_input_basename)

        segment_paths = self.slicer.slice_video(current_processed_video_path, sliced_videos_base_dir)

        if not segment_paths:
            print(
                f"Video slicing failed or produced no segments for {current_processed_video_path}. Aborting further processing for this video.")
            if not is_original_file_being_processed:  # If the input to slicing was a converted file
                self.intermediate_paths_to_cleanup.append(
                    os.path.dirname(current_processed_video_path))  # Its parent dir
            self._perform_cleanup()
            return

        # If slicing was successful and the input was an intermediate converted file, mark its dir for cleanup
        if not is_original_file_being_processed:
            self.intermediate_paths_to_cleanup.append(os.path.dirname(current_processed_video_path))
        # The main directory containing all segments of this video
        self.intermediate_paths_to_cleanup.append(os.path.dirname(sliced_videos_base_dir))  # e.g. .../2_sliced_videos/

        # --- Process each segment ---
        for i, segment_path in enumerate(segment_paths):
            segment_basename = os.path.splitext(os.path.basename(segment_path))[0]
            print(f"\n--- Processing Segment {i + 1}/{len(segment_paths)}: {segment_basename} ---")

            current_segment_path_for_analysis = segment_path

            # --- 3. Stabilize Segment ---
            print(f"Step 3: Stabilizing segment '{segment_basename}'...")
            # Stabilized videos go into a subdir named after the original video, then the segment name
            stabilization_output_dir = self._get_output_path_for_step("3_stabilized_videos", slicing_input_basename)
            stabilized_segment_filename = f"{segment_basename}_stabilized.mp4"
            stabilized_segment_path = os.path.join(stabilization_output_dir, stabilized_segment_filename)

            stabilized_output_path = self.stabilizer.stabilize(current_segment_path_for_analysis,
                                                               stabilized_segment_path)
            if stabilized_output_path:
                current_segment_path_for_analysis = stabilized_output_path
                # Don't add individual stabilized files for cleanup, add the parent dir later if all segments are done.
            else:
                print(
                    f"Stabilization failed for segment '{segment_path}'. Analyzing original/previous segment version.")

            # --- 4. Analyze Segment (Pose & Depth Extraction) ---
            print(f"Step 4: Analyzing segment '{os.path.basename(current_segment_path_for_analysis)}'...")
            # CSVs go into a subdir named after the original video
            csv_output_dir = self._get_output_path_for_step("4_csv_data", slicing_input_basename)
            analysis_input_basename = os.path.splitext(os.path.basename(current_segment_path_for_analysis))[0]
            csv_filename = f"{analysis_input_basename}_analysis.csv"
            csv_output_path = os.path.join(csv_output_dir, csv_filename)

            # YOLO's processed video output (visualizations)
            # Project dir for all YOLO outputs from this pipeline run
            yolo_processed_video_project_dir = self._get_output_path_for_step("5_yolo_processed_frames")

            self.analyzer.analyze_video(
                video_path=current_segment_path_for_analysis,
                output_csv_path=csv_output_path,
                processed_video_output_project_dir=yolo_processed_video_project_dir  # YOLO will create name subdir
            )
            print(f"--- Finished processing segment: {segment_basename} ---")

        # Add parent directories of stabilized videos for cleanup after all segments are processed
        self.intermediate_paths_to_cleanup.append(self._get_output_path_for_step("3_stabilized_videos"))
        self.intermediate_paths_to_cleanup.append(self._get_output_path_for_step("5_yolo_processed_frames"))

        self._perform_cleanup()
        print("\nVideo processing pipeline finished successfully.")

    def _perform_cleanup(self):
        if self.config.get("cleanup_intermediate_folders", False):
            print("\nCleaning up intermediate folders/files...")
            # Use set to avoid duplicate removal attempts, sort for somewhat predictable order (mostly for logs)
            unique_paths = sorted(list(set(self.intermediate_paths_to_cleanup)), reverse=True)

            for path_to_remove in unique_paths:
                # Critical safety: ensure it's within the pipeline's base_output_folder
                # AND not the base_output_folder itself
                # AND not one of the designated final output folders (like the main CSV data folder)
                abs_path_to_remove = os.path.abspath(path_to_remove)
                abs_base_output_folder = os.path.abspath(self.base_output_folder)

                is_safe_to_remove = abs_path_to_remove.startswith(abs_base_output_folder) and \
                                    abs_path_to_remove != abs_base_output_folder and \
                                    not abs_path_to_remove.startswith(os.path.abspath(
                                        self._get_output_path_for_step("4_csv_data")))  # Protect final CSVs

                if os.path.exists(abs_path_to_remove) and is_safe_to_remove:
                    try:
                        if os.path.isdir(abs_path_to_remove):
                            shutil.rmtree(abs_path_to_remove)
                            print(f"Removed intermediate directory: {abs_path_to_remove}")
                        # elif os.path.isfile(abs_path_to_remove): # If we were tracking individual files
                        # os.remove(abs_path_to_remove)
                        # print(f"Removed intermediate file: {abs_path_to_remove}")
                    except OSError as e:
                        print(f"Error cleaning up {abs_path_to_remove}: {e}")
                elif not is_safe_to_remove and os.path.exists(abs_path_to_remove):
                    print(
                        f"Skipping cleanup of '{abs_path_to_remove}' (outside designated cleanup scope or a final data folder).")
            self.intermediate_paths_to_cleanup.clear()  # Clear list after attempting cleanup