from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.tools import subprocess_call  # For better logging/error handling if needed
import os


class VideoSlicer:
    def __init__(self, segment_duration_seconds: int = 300):
        self.segment_duration = segment_duration_seconds

    def slice_video(self, input_video_path: str, output_folder: str) -> list[str]:
        """Splits a video into segments."""
        print(f"Slicing video: {input_video_path} into segments of {self.segment_duration}s")

        if not os.path.exists(input_video_path):
            print(f"Error: Input video for slicing not found at {input_video_path}")
            return []

        os.makedirs(output_folder, exist_ok=True)
        segment_paths = []
        video = None  # Initialize video to None for finally block

        try:
            video = VideoFileClip(input_video_path)
            video_duration = int(video.duration)

            if video_duration == 0:
                print(f"Warning: Video '{input_video_path}' has zero duration. Cannot slice.")
                return []

            for start_time in range(0, video_duration, self.segment_duration):
                end_time = min(start_time + self.segment_duration, video_duration)

                # Ensure segment has a meaningful duration
                if end_time - start_time < 0.5:  # Avoid extremely short clips
                    continue

                segment = video.subclip(start_time, end_time)
                base_name = os.path.splitext(os.path.basename(input_video_path))[0]
                # Sanitize filename components if necessary, though os.path.join handles separators
                output_filename = f"{base_name}_segment_{start_time}s_to_{end_time}s.mp4"
                output_path = os.path.join(output_folder, output_filename)

                print(f"Writing segment: {output_path} ({start_time}s - {end_time}s)")
                # Consider adding threads=4 or other ffmpeg_params for performance
                segment.write_videofile(output_path, codec="libx264",
                                        logger=None)  # logger=None to reduce moviepy verbosity
                segment_paths.append(output_path)

            print(
                f"Video '{input_video_path}' split into {len(segment_paths)} segments successfully in '{output_folder}'.")
            return segment_paths
        except Exception as e:
            print(f"Error during slicing of '{input_video_path}': {e}")
            # You might want to log the full traceback here for debugging
            import traceback
            traceback.print_exc()
            return []
        finally:
            if video:
                video.close()  # Release resources