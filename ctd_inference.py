"""
ctd_inference.py - Minimal ONNX inference shell for comic-text-detector.

This module does NOT vendor the upstream training repository
(https://github.com/dmMaze/comic-text-detector, GPL-3.0). It only loads the
published ONNX weights and reproduces the standard inference pre/post-processing
needed to consume them (letterbox resize, NMS, mask thresholding).

The model produces three heads. We use two:
  - "blk": text-block detections [cx, cy, w, h, conf, lang_eng, lang_ja, ...]
           (the cls columns here encode language, NOT text-vs-balloon)
  - "seg": single-channel text pixel mask (used for inpainting)

Note: This model does NOT detect speech balloons. Balloons must be derived
separately via flood-fill post-processing (see manga_text_removal.py).

Inspiration: mayocream/koharu (https://github.com/mayocream/koharu)
"""
import cv2
import numpy as np


def _letterbox(img, new_shape=1024, stride=64):
    """[Unused] Aspect-preserving letterbox kept for reference.

    The comic-text-detector ONNX export requires a FIXED 1024x1024 input, so
    detect() below uses a plain resize_exact instead (matching koharu's
    comic-text-detector crate) and maps coordinates back per-axis.
    """
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = new_shape - new_w
    pad_h = new_shape - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, (r, r), (pad_w, pad_h), (top, left)


def _xywh2xyxy(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def _nms(boxes_xyxy, scores, iou_thr):
    """Pure-numpy NMS (avoids depending on torchvision). Returns kept indices."""
    if len(scores) == 0:
        return np.array([], dtype=np.int32)
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)


class SimpleTextDetector:
    """Minimal ONNX runtime wrapper for the comic-text-detector model.

    Loads the published ONNX weights (comic-text-detector by dmMaze, GPL-3.0)
    and exposes a single `detect()` method returning (text_mask, bboxes).
    """

    INPUT_SIZE = 1024
    MASK_THRESHOLD = 30  # 0-255, below this the seg head output is considered background

    def __init__(self, model_path, device="auto"):
        import onnxruntime as ort

        # Probe available providers. Prefer CUDA EP only when requested/available
        # so the node keeps working on CPU-only setups.
        providers = []
        if device in ("cuda", "auto"):
            try:
                available = ort.get_available_providers()
                if "CUDAExecutionProvider" in available:
                    providers.append("CUDAExecutionProvider")
            except Exception:
                pass
        providers.append("CPUExecutionProvider")

        self.sess = ort.InferenceSession(
            model_path,
            providers=providers,
            sess_options=ort.SessionOptions(),
        )
        self.input_name = self.sess.get_inputs()[0].name
        self.input_shape = self.sess.get_inputs()[0].shape  # e.g. [1,3,1024,1024]
        # Output names vary across exports; locate blk (detection) and seg (mask) heads.
        out_names = [o.name for o in self.sess.get_outputs()]
        self.blk_name = self._pick(out_names, ["blk", "det", "output0", "boxes"])
        self.seg_name = self._pick(out_names, ["seg", "mask", "segment"])

    @staticmethod
    def _pick(names, candidates):
        for c in candidates:
            for n in names:
                if c.lower() in n.lower():
                    return n
        return names[0]

    def detect(self, img_bgr, conf_threshold=0.4, nms_threshold=0.35):
        """Run detection.

        Args:
            img_bgr: HxWx3 uint8 BGR image.
            conf_threshold: detection confidence threshold.
            nms_threshold: IoU threshold for NMS.

        Returns:
            text_mask: HxW uint8 (0 or 255) text pixel mask at original resolution.
            bboxes: list of dicts {x, y, w, h, confidence, language}.
        """
        h0, w0 = img_bgr.shape[:2]

        # The comic-text-detector ONNX export has a FIXED 1024x1024 input shape,
        # so we resize_exact (stretch aspect ratio) rather than letterbox. We
        # then scale x/y independently when mapping boxes and the mask back to
        # the original resolution. This matches koharu's
        # comic-text-detector crate (resize_exact + w_ratio/h_ratio).
        resized = cv2.resize(img_bgr, (self.INPUT_SIZE, self.INPUT_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        inp = img_rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        outputs = self.sess.run([self.blk_name, self.seg_name], {self.input_name: inp})

        w_ratio = w0 / float(self.INPUT_SIZE)
        h_ratio = h0 / float(self.INPUT_SIZE)

        # ---- detection head: [cx, cy, w, h, conf, lang_eng, lang_ja, ...] ----
        blk = outputs[0]
        # The export may carry a leading batch index; collapse to (N, fields).
        blk = np.asarray(blk).reshape(-1, blk.shape[-1])
        # remove batch-index column if present (first col == 0 for all rows)
        if blk.shape[1] >= 8 and np.allclose(blk[:, 0], 0):
            blk = blk[:, 1:]
        # now expect [cx, cy, w, h, conf, cls0, cls1, ...]
        conf = blk[:, 4]
        keep_conf = conf >= conf_threshold
        blk = blk[keep_conf]
        if blk.shape[0] > 0:
            xywh = blk[:, :4].copy()
            # map center/size from 1024-space to original via per-axis ratio
            xywh[:, 0] *= w_ratio
            xywh[:, 2] *= w_ratio
            xywh[:, 1] *= h_ratio
            xywh[:, 3] *= h_ratio
            xyxy = _xywh2xyxy(xywh)
            scores = blk[:, 4]
            # language = argmax of remaining cls columns
            cls_cols = blk[:, 5:]
            lang_idx = cls_cols.argmax(axis=1) if cls_cols.shape[1] > 0 else np.zeros(len(blk), dtype=int)
            keep = _nms(xyxy, scores, nms_threshold)
            xyxy = xyxy[keep]
            scores = scores[keep]
            lang_idx = lang_idx[keep]
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            lang_idx = np.zeros((0,), dtype=int)

        bboxes = []
        lang_names = ["eng", "ja", "unknown"]
        for box, s, li in zip(xyxy, scores, lang_idx):
            x1, y1, x2, y2 = box
            bboxes.append({
                "x": float(np.clip(x1, 0, w0 - 1)),
                "y": float(np.clip(y1, 0, h0 - 1)),
                "w": float(x2 - x1),
                "h": float(y2 - y1),
                "confidence": float(s),
                "language": lang_names[int(li)] if int(li) < len(lang_names) else "unknown",
            })

        # ---- segmentation head: single-channel mask ----
        seg = np.asarray(outputs[1]).squeeze()
        if seg.ndim == 3:  # (C,H,W) -> take first channel
            seg = seg[0]
        seg = (seg * 255.0).clip(0, 255).astype(np.uint8)
        # resize mask back to original resolution, then threshold
        text_mask = cv2.resize(seg, (w0, h0), interpolation=cv2.INTER_LINEAR)
        _, text_mask = cv2.threshold(text_mask, self.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)

        return text_mask, bboxes
