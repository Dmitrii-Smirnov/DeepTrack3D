import torch
from ultralytics import YOLO
import torch.backends.cudnn as cudnn
import os


class ModelManager:
    def __init__(self, yolo_model_path: str = "yolov11x-pose.pt",
                 midas_model_type: str = "DPT_Large",
                 yolo_conf: float = 0.5, yolo_iou: float = 0.45,
                 half_precision: bool = True):

        self.yolo_model_path = yolo_model_path
        self.midas_model_type = midas_model_type
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou

        self.device, self.num_gpus = self._detect_device()
        self.half_precision = half_precision if self.device.type == 'cuda' else False  # Only on GPU

        if self.device.type == 'cuda':
            cudnn.benchmark = True  # Enable CuDNN autotuning for performance if using CUDA

        self.yolo_model = self._load_yolo_model()
        self.midas_model = self._load_midas_model()

        self.midas_transform = None
        if self.midas_model:  # Only load transform if MiDaS model loaded
            try:
                midas_transforms_hub = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True, verbose=False)
                if self.midas_model_type in ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]:
                    self.midas_transform = midas_transforms_hub.dpt_transform
                elif self.midas_model_type == "MiDaS_small_Transformer":  # Example for specific model
                    self.midas_transform = midas_transforms_hub.beit512_transform  # Check MiDaS repo for correct transform
                else:  # Fallback or default, check MiDaS documentation
                    self.midas_transform = midas_transforms_hub.small_transform
                print(f"Loaded MiDaS transform suitable for {self.midas_model_type}.")
            except Exception as e:
                print(f"Error loading MiDaS transforms: {e}. Depth estimation might not work.")
                self.midas_transform = None

    def _detect_device(self):
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            device = torch.device("cuda")
            print(f"CUDA is available. Using {torch.cuda.get_device_name(0)}. Number of GPUs: {num_gpus}")
        else:
            num_gpus = 0
            device = torch.device("cpu")
            print("CUDA is not available. Using CPU.")
        return device, num_gpus

    def _load_yolo_model(self):
        try:
            print(f"Loading YOLO model: {self.yolo_model_path} with conf={self.yolo_conf}, iou={self.yolo_iou}")
            # YOLO automatically tries to download if a standard name like 'yolov11x-pose.pt' is given and not found.
            model = YOLO(self.yolo_model_path)

            # For task-specific models (like pose), conf/iou are often part of the model's internal state.
            # For general detection models, they might be arguments to predict/track.
            # The YOLO object itself might not have a direct .model.conf attribute for all versions/tasks.
            # It's safer to rely on passing conf/iou to the track/predict methods if the direct attribute setting fails
            # or is not standard across all YOLO model types.
            # However, the provided PoseDepthTracker had this, so we keep it, but with a check.
            if hasattr(model, 'model') and hasattr(model.model, 'yaml') and 'pose' in model.model.yaml.get('task', ''):
                # This is a more specific check for pose models if direct conf/iou setting is needed.
                # Ultralytics models usually handle this internally or via track/predict args.
                pass  # For now, assume YOLO handles conf/iou passed to track method or its internal defaults.

            model.to(self.device)
            if self.half_precision:
                model.half()
                print("YOLO model set to half precision (FP16).")

            # Multi-GPU with DataParallel can be complex for some models like YOLO track.
            # Often better to run on a single GPU or let Ultralytics handle it.
            # if self.num_gpus > 1:
            # model = torch.nn.DataParallel(model)
            # print(f"Wrapped YOLO model with DataParallel for {self.num_gpus} GPUs.")
            print(f"YOLO model '{self.yolo_model_path}' loaded successfully on {self.device}.")
            return model
        except Exception as e:
            print(f"Error loading YOLO model '{self.yolo_model_path}': {e}")
            if "trust_repo=True" in str(e):
                print("Hint: Try adding `trust_repo=True` if using torch.hub for a custom model source.")
            import traceback
            traceback.print_exc()
            return None

    def _load_midas_model(self):
        try:
            print(f"Loading MiDaS model: {self.midas_model_type}")
            # `force_reload=True` can be useful for debugging, but remove for production.
            midas = torch.hub.load("intel-isl/MiDaS", self.midas_model_type, trust_repo=True, verbose=False)
            midas.to(self.device)
            midas.eval()  # Set to evaluation mode
            if self.half_precision:
                midas.half()
                print("MiDaS model set to half precision (FP16).")

            # if self.num_gpus > 1:
            # midas = torch.nn.DataParallel(midas)
            # print(f"Wrapped MiDaS model with DataParallel for {self.num_gpus} GPUs.")
            print(f"MiDaS model '{self.midas_model_type}' loaded successfully on {self.device}.")
            return midas
        except Exception as e:
            print(f"Error loading MiDaS model '{self.midas_model_type}': {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_yolo_model(self):
        return self.yolo_model

    def get_midas_model(self):
        return self.midas_model

    def get_midas_transform(self):
        return self.midas_transform