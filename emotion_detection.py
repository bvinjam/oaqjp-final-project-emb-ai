import importlib.util
from pathlib import Path

_module_path = Path(__file__).resolve().parent / "oaqjp-final-project-emb-ai" / "EmotionDetection" / "emotion_detection.py"
_spec = importlib.util.spec_from_file_location("emotion_detection_compat", _module_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
emotion_detector = _module.emotion_detector

__all__ = ["emotion_detector"]
