import os


class VideoStabilizer:
    def __init__(self, border_type: str = 'black', border_size: str = 'auto'):
        try:
            from vidstab import VidStab
            self.VidStab = VidStab
            self.vidstab_available = True
        except ImportError:
            print("Warning: vidstab library not found. Stabilization feature will be unavailable.")
            self.VidStab = None
            self.vidstab_available = False

        self.border_type = border_type
        self.border_size = border_size

    def stabilize(self, input_video_path: str, output_video_path: str) -> str | None:
        """Stabilizes a video using VidStab."""
        if not self.vidstab_available:
            print(f"Cannot stabilize video '{input_video_path}': vidstab library is not available.")
            return None

        if not os.path.exists(input_video_path):
            print(f"Error: Input video for stabilization not found at {input_video_path}")
            return None

        print(f"Stabilizing video: {input_video_path} to {output_video_path}")
        stabilizer = self.VidStab()  # Instantiate VidStab object
        try:
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            # Note: VidStab can be verbose.
            stabilizer.stabilize(input_path=input_video_path,
                                 output_path=output_video_path,
                                 border_type=self.border_type,
                                 border_size=self.border_size)
            print(f"Stabilization successful: {output_video_path}")
            return output_video_path
        except Exception as e:
            print(f"Error during VidStab stabilization of '{input_video_path}': {e}")
            import traceback
            traceback.print_exc()
            return None