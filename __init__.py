"""
ComfyUI-RayNodes
Custom nodes for image processing and face analysis
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_dependencies():
    """Auto-install dependencies on ComfyUI startup if missing."""
    # Common dependencies (won't trigger torch reinstall)
    common_deps = [
        "numpy",
        "pillow",
        "requests",
        "onnx",
        "onnxruntime",
        "opencv-python",
    ]

    # Packages that need --no-deps to avoid torch reinstall
    no_deps_packages = [
        "facenet-pytorch",
        "emotiefflib",
    ]

    def is_package_installed(package_name):
        """Check if a package is installed."""
        try:
            __import__(package_name.replace("-", "_").split("[")[0])
            return True
        except ImportError:
            return False

    def install_package(package, no_deps=False):
        """Install a package using pip."""
        try:
            cmd = [sys.executable, "-m", "pip", "install", package]
            if no_deps:
                cmd.append("--no-deps")
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"[Comfyui-RayNodes] Installed: {package}")
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"[Comfyui-RayNodes] Failed to install {package}: {e}")
            return False

    # Check and install common dependencies
    missing_common = []
    for dep in common_deps:
        pkg_name = dep.replace("-", "_").split("[")[0]
        if pkg_name == "opencv_python":
            pkg_name = "cv2"
        if not is_package_installed(pkg_name):
            missing_common.append(dep)

    if missing_common:
        logger.info(f"[Comfyui-RayNodes] Installing missing dependencies: {missing_common}")
        for dep in missing_common:
            install_package(dep)

    # Check and install no-deps packages
    for pkg in no_deps_packages:
        pkg_name = pkg.replace("-", "_")
        if not is_package_installed(pkg_name):
            logger.info(f"[Comfyui-RayNodes] Installing {pkg} (with --no-deps to preserve CUDA torch)...")
            install_package(pkg, no_deps=True)

# Auto-install dependencies on import
ensure_dependencies()

from .ray_nodes import *

# Node class mapping
NODE_CLASS_MAPPINGS = {
    "BracketedTagIndexMerger": BracketedTagIndexMerger,
    "Florence2TagProcessor": Florence2TagProcessor,
    "ImageListConverter": ImageListConverter,
    "ImageSelector": ImageSelector,
    "MaskBlackener": MaskBlackener,
    "MaskApplierAndCombiner": MaskApplierAndCombiner,
    "MaskProcessor": MaskProcessor,
    "TagArrayToLines": TagArrayToLines,
    "TagIndexMerger": TagIndexMerger,
    "GrabberTagProcessor": GrabberTagProcessor,
    "ImageResizer": ImageResizer,
    "SaveImageWebsocket": SaveImageWebsocket,
    "BorderMask": BorderMask,
    "SaturationAdjuster": SaturationAdjuster,
    "HighlightOverlay": HighlightOverlay,
    "MaskMerger": MaskMerger,
    # Face Analysis Nodes (DeepFace)
    "FaceAnalysis": FaceAnalysisNode,
    "EmotionAnalysis": EmotionAnalysisNode,
    "FaceExtract": FaceExtractNode,
    "DescriptionGen": DescriptionGenNode,
    # Face Analysis Nodes (Alternative Engines)
    "EmotiEffLibAnalysis": EmotiEffLibAnalysisNode,
    "HSEmotionAnalysis": HSEmotionAnalysisNode,
    "MultiEngineEmotion": MultiEngineEmotionNode,
}

# Node display name mapping
NODE_DISPLAY_NAME_MAPPINGS = {
    "BracketedTagIndexMerger": "🏷️ Bracketed Tag-Index Merger",
    "Florence2TagProcessor": "🏷️ Florence2 Tag Processor",
    "ImageListConverter": "🖼️Image List Converter",
    "ImageSelector": "🖼️ Image Selector",
    "MaskBlackener": "🖤Mask Blackener",
    "MaskApplierAndCombiner": "🎭Mask Applier and Combiner",
    "MaskProcessor": "🎭Mask Processor",
    "TagArrayToLines": "📄 Tag Array to Lines",
    "TagIndexMerger": "🏷️ Tag-Index Merger",
    "GrabberTagProcessor": "Grabber Tag Processor",
    "ImageResizer": "🖼️Image Resizer",
    "SaveImageWebsocket": "Save Image Websocket",
    "BorderMask": "🎭Border Mask",
    "SaturationAdjuster": "🌈SaturationAdjuster",
    "HighlightOverlay": "✨亮部覆盖 (Highlight Overlay)",
    "MaskMerger": "🎭Mask Merger",
    # Face Analysis Nodes (DeepFace)
    "FaceAnalysis": "😊 Face Analysis (DeepFace)",
    "EmotionAnalysis": "😊 Emotion Analysis",
    "FaceExtract": "😊 Face Extract",
    "DescriptionGen": "😊 Description Generator",
    # Face Analysis Nodes (Alternative Engines)
    "EmotiEffLibAnalysis": "😊 EmotiEffLib Analysis (推荐)",
    "HSEmotionAnalysis": "😊 HSEmotion Analysis",
    "MultiEngineEmotion": "😊 Multi-Engine Emotion",
}
