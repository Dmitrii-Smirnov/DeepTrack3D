import cv2
import torch
import numpy as np
import time
import os

# Assuming ModelManager and CSVDataWriter will be imported from their respective locations
# For example, if main.py is run from video_processing_app's parent directory:
# from video_processing_app.core_logic.model_manager import ModelManager
# from video_processing_app.utils.data_writer import CSVDataWriter
# If main.py is IN video_processing_app, and video_processing_app is in PYTHONPATH:
from core_logic.model_manager import ModelManager
from utils.data_writer import CSVDataWriter


class VideoAnalyzer:
    def __init__(self, model_manager: ModelManager,
                 tracker_config_path: str = "bytetrack.yaml",
                 yolo_classes: list = [0],
                 downscale_factor: float = 1.0):

        self.model_manager = model_manager
        self.yolo_model = self.model_manager.get_yolo_model()
        self.midas_model = self.model_manager.get_midas_model()
        self.midas_transform = self.model_manager.get_midas_transform()
        self.device = self.model_manager.device

        self.tracker_config_path = tracker_config_path
        self.yolo_classes = yolo_classes
        self.downscale_factor = max(0.1, min(2.0, downscale_factor))  # Clamp between 0.1 and 2.0

        if not all([self.yolo_model, self.midas_model, self.midas_transform]):
            raise RuntimeError(
                "VideoAnalyzer: One or more critical models (YOLO, MiDaS, MiDaS Transform) failed to load from ModelManager.")

        self.keypoint_names = [  # COCO 17 keypoints order, ensure this matches YOLO output
            "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
            "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
            "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
            "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
        ]
        self.csv_header = self._generate_csv_header()

    def _generate_csv_header(self) -> list[str]:
        header = [
            "FrameID", "ObjectID", "BoxX1", "BoxY1", "BoxX2", "BoxY2",
            "CentroidX", "CentroidY", "CentroidDepth"
        ]
        for name in self.keypoint_names:
            header.extend([f"{name}X", f"{name}Y", f"{name}Depth"])
        return header

    def _estimate_depth(self, frame_rgb_for_depth_estimation: np.ndarray) -> np.ndarray | None:
        """Estimates depth map for the input frame (RGB)."""
        if self.midas_model is None or self.midas_transform is None:
            print("Error: MiDaS model or transform not available for depth estimation.")
            return None
        try:
            input_batch = self.midas_transform(frame_rgb_for_depth_estimation).to(self.device)
            if self.model_manager.half_precision:  # Already checked for CUDA in ModelManager
                input_batch = input_batch.half()

            with torch.no_grad():
                prediction = self.midas_model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame_rgb_for_depth_estimation.shape[:2],  # Output depth at this resolution
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            return prediction.cpu().numpy()
        except Exception as e:
            print(f"Error during MiDaS depth estimation: {e}")
            # import traceback; traceback.print_exc() # For more detailed debug
            return None

    def _get_depth_at_original_coords(self, depth_map_scaled: np.ndarray,
                                      x_orig: int, y_orig: int,
                                      original_frame_shape: tuple,
                                      scaled_frame_shape_for_depth: tuple) -> float:
        """Safely retrieve depth value, scaling coordinates if necessary."""
        if original_frame_shape[1] == 0 or original_frame_shape[0] == 0:  # Avoid division by zero
            return np.nan

        scale_x = scaled_frame_shape_for_depth[1] / original_frame_shape[1]
        scale_y = scaled_frame_shape_for_depth[0] / original_frame_shape[0]

        x_coord_in_depth_map = int(x_orig * scale_x)
        y_coord_in_depth_map = int(y_orig * scale_y)

        # Clamp coordinates to be within the bounds of the depth_map_scaled
        y_coord_in_depth_map = max(0, min(y_coord_in_depth_map, depth_map_scaled.shape[0] - 1))
        x_coord_in_depth_map = max(0, min(x_coord_in_depth_map, depth_map_scaled.shape[1] - 1))

        return depth_map_scaled[y_coord_in_depth_map, x_coord_in_depth_map]

    def _process_tracked_object(self, frame_id: int, original_frame_shape: tuple,
                                scaled_frame_shape_for_depth: tuple,
                                depth_map_at_scaled_res: np.ndarray,
                                yolo_box_result,  # ultralytics.engine.results.Box object
                                yolo_keypoints_result  # ultralytics.engine.results.Keypoints object for one person
                                ) -> list:
        """Processes a single tracked object to extract required data."""

        object_id_tensor = yolo_box_result.id
        object_id = int(
            object_id_tensor.item()) if object_id_tensor is not None and object_id_tensor.numel() > 0 else np.nan

        # Bounding box from original image coordinates
        bbox_orig_tensor = yolo_box_result.xyxy[0]
        bbox_orig = bbox_orig_tensor.cpu().tolist()  # Ensure it's on CPU and converted to list

        img_h_orig, img_w_orig = original_frame_shape[:2]

        # Clamp bounding box to image dimensions
        # Ensure coordinates are numbers before min/max
        clamped_x1 = max(0, min(float(bbox_orig[0]), img_w_orig - 1))
        clamped_y1 = max(0, min(float(bbox_orig[1]), img_h_orig - 1))
        clamped_x2 = max(0, min(float(bbox_orig[2]), img_w_orig - 1))
        clamped_y2 = max(0, min(float(bbox_orig[3]), img_h_orig - 1))
        bbox_clamped_orig = [clamped_x1, clamped_y1, clamped_x2, clamped_y2]

        centroid_x_orig = int((bbox_clamped_orig[0] + bbox_clamped_orig[2]) / 2)
        centroid_y_orig = int((bbox_clamped_orig[1] + bbox_clamped_orig[3]) / 2)

        centroid_depth = self._get_depth_at_original_coords(
            depth_map_at_scaled_res, centroid_x_orig, centroid_y_orig,
            original_frame_shape, scaled_frame_shape_for_depth
        )

        keypoints_data_flat = []
        num_expected_keypoints = len(self.keypoint_names)

        if yolo_keypoints_result and hasattr(yolo_keypoints_result, 'xy') and yolo_keypoints_result.xy.numel() > 0:
            # keypoints_xy_orig_tensor should be for a single person, shape [1, num_kpts, 2] or [num_kpts, 2]
            keypoints_xy_orig_tensor = yolo_keypoints_result.xy[
                0].cpu()  # Get data for the first (only) person, move to CPU

            for i in range(num_expected_keypoints):
                if i < keypoints_xy_orig_tensor.shape[0]:  # Check if this keypoint index exists
                    kp_x_orig, kp_y_orig = int(keypoints_xy_orig_tensor[i, 0]), int(keypoints_xy_orig_tensor[i, 1])

                    kp_x_clamped = max(0, min(kp_x_orig, img_w_orig - 1))
                    kp_y_clamped = max(0, min(kp_y_orig, img_h_orig - 1))

                    kp_depth = self._get_depth_at_original_coords(
                        depth_map_at_scaled_res, kp_x_clamped, kp_y_clamped,
                        original_frame_shape, scaled_frame_shape_for_depth
                    )
                    keypoints_data_flat.extend([kp_x_clamped, kp_y_clamped, kp_depth])
                else:  # Keypoint not present in detection
                    keypoints_data_flat.extend([np.nan, np.nan, np.nan])
        else:  # No keypoints detected for this object or keypoints data is empty
            keypoints_data_flat.extend([np.nan] * (num_expected_keypoints * 3))

        # Final check for length, should not be necessary if logic above is correct
        expected_len = num_expected_keypoints * 3
        if len(keypoints_data_flat) < expected_len:
            keypoints_data_flat.extend([np.nan] * (expected_len - len(keypoints_data_flat)))
        elif len(keypoints_data_flat) > expected_len:
            keypoints_data_flat = keypoints_data_flat[:expected_len]

        row_data = [frame_id, object_id] + bbox_clamped_orig + \
                   [centroid_x_orig, centroid_y_orig, centroid_depth] + \
                   keypoints_data_flat
        return row_data

    def analyze_video(self, video_path: str, output_csv_path: str,
                      processed_video_output_project_dir: str):
        """Analyzes the video, extracts data, and saves to CSV."""
        print(f"Starting analysis for video: {video_path}")

        if not os.path.exists(video_path):
            print(f"Error: Video for analysis not found at {video_path}")
            return
        if not os.path.exists(self.tracker_config_path):
            print(f"Error: Tracker configuration file not found at {self.tracker_config_path}")
            print("Analysis cannot proceed without a valid tracker configuration.")
            return

        data_writer = CSVDataWriter(output_csv_path)
        if not data_writer.writer:
            print(
                f"Critical: CSV writer for {output_csv_path} could not be initialized. Aborting analysis for {video_path}.")
            return
        data_writer.write_header(self.csv_header)

        try:
            # Ensure the project directory for YOLO outputs exists
            os.makedirs(processed_video_output_project_dir, exist_ok=True)

            # Define a unique name for this specific video's output subfolder within the project dir
            video_base_name = os.path.splitext(os.path.basename(video_path))[0]
            yolo_experiment_name = f"{video_base_name}_analysis_frames"

            results_generator = self.yolo_model.track(
                source=video_path,
                show=False,
                save=True,  # Saves visualized frames/video by YOLO
                tracker=self.tracker_config_path,
                classes=self.yolo_classes,
                conf=self.model_manager.yolo_conf,  # Pass configured confidence
                iou=self.model_manager.yolo_iou,  # Pass configured IoU
                project=processed_video_output_project_dir,  # Main output directory for YOLO
                name=yolo_experiment_name,  # Subdirectory for this run's outputs
                stream=True,
                persist=True,  # Persist tracks across frames
                verbose=False  # Reduce YOLO verbosity
            )
        except Exception as e:
            print(f"Error initiating YOLO tracking for {video_path}: {e}")
            data_writer.close_file()
            return

        frame_id_counter = 0
        print(f"Analyzing frames from {video_path}...")
        for frame_results in results_generator:  # frame_results is ultralytics.engine.results.Results
            frame_id_counter += 1
            start_time_frame = time.time()

            original_frame_bgr = frame_results.orig_img  # NumPy array (H, W, C) in BGR
            original_frame_shape = original_frame_bgr.shape

            # Prepare frame for depth estimation
            frame_for_depth_bgr = original_frame_bgr
            if self.downscale_factor != 1.0 and 0.1 < self.downscale_factor < 2.0:
                new_w = int(original_frame_shape[1] * self.downscale_factor)
                new_h = int(original_frame_shape[0] * self.downscale_factor)
                if new_w > 0 and new_h > 0:  # Ensure valid dimensions
                    frame_for_depth_bgr = cv2.resize(original_frame_bgr, (new_w, new_h),
                                                     interpolation=cv2.INTER_AREA if self.downscale_factor < 1.0 else cv2.INTER_LINEAR)

            scaled_frame_shape_for_depth = frame_for_depth_bgr.shape
            frame_rgb_for_depth = cv2.cvtColor(frame_for_depth_bgr, cv2.COLOR_BGR2RGB)

            depth_map_at_scaled_res = self._estimate_depth(frame_rgb_for_depth)
            if depth_map_at_scaled_res is None:
                # print(f"Frame {frame_id_counter}: Depth estimation failed. Skipping object processing.")
                # Optionally, write a row with FrameID and NaNs for all other fields
                nan_row = [frame_id_counter] + [np.nan] * (len(self.csv_header) - 1)
                data_writer.write_row(nan_row)
                continue

            if frame_results.boxes:  # ultralytics.engine.results.Boxes object
                for i in range(len(frame_results.boxes)):
                    box_result = frame_results.boxes[i]  # Box object for one instance

                    keypoints_result_for_box = None
                    if frame_results.keypoints and hasattr(frame_results.keypoints, 'xy') and i < len(
                            frame_results.keypoints.xy):
                        # Assuming keypoints are ordered same as boxes, and keypoints[i] corresponds to boxes[i]
                        keypoints_result_for_box = frame_results.keypoints[i]  # Keypoints object for one instance

                    # Process only if it has a tracking ID from the tracker
                    if box_result.id is not None and box_result.id.numel() > 0:
                        row_data = self._process_tracked_object(
                            frame_id=frame_id_counter,
                            original_frame_shape=original_frame_shape,
                            scaled_frame_shape_for_depth=scaled_frame_shape_for_depth,
                            depth_map_at_scaled_res=depth_map_at_scaled_res,
                            yolo_box_result=box_result,
                            yolo_keypoints_result=keypoints_result_for_box
                        )
                        data_writer.write_row(row_data)
                    # else:
                    # print(f"Frame {frame_id_counter}: Object detected without tracking ID, skipping.")
            else:  # No boxes detected in this frame
                # print(f"Frame {frame_id_counter}: No objects detected.")
                # Write a row with FrameID and NaNs for other fields to indicate frame was processed
                nan_row = [frame_id_counter] + [np.nan] * (len(self.csv_header) - 1)
                data_writer.write_row(nan_row)

            if frame_id_counter % 50 == 0:  # Log progress every 50 frames
                elapsed_time_frame = time.time() - start_time_frame
                print(f"Processed Frame ID: {frame_id_counter} (took {elapsed_time_frame:.3f}s)")

        data_writer.close_file()
        print(f"Analysis complete for '{video_path}'. Data saved to {output_csv_path}")
        yolo_output_full_path = os.path.join(processed_video_output_project_dir, yolo_experiment_name)
        print(f"Processed YOLO video/frame visualizations saved in: {yolo_output_full_path}")