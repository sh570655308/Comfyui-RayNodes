"""
MangaTextRemoval - Manga/comic text detection, balloon mask derivation and inpainting.

This ComfyUI node takes an image and returns:
  - the image with text erased (LaMa inpainting),
  - a binary mask of detected text pixels,
  - a binary mask of speech balloons (derived via flood-fill),
  - a JSON string of detected text-block bounding boxes.

==============================================================================
License / attribution notes
==============================================================================
- The detection weights are from comic-text-detector by dmMaze
  (https://github.com/dmMaze/comic-text-detector, GPL-3.0). This file only
  loads those published ONNX weights via a self-contained inference shell
  (ctd_inference.py); it does NOT reproduce upstream training/model code.
- The inpainting model is LaMa by advimman (Apache-2.0), loaded through the
  `simple-lama-inpainting` package.
- The tiled-inference scheme (split into overlapping tiles, blend with a
  raised-cosine window) and the flood-fill balloon-derivation approach are
  inspired by mayocream/koharu (https://github.com/mayocream/koharu, GPL-3.0).
  The code here is written from scratch in Python; no Rust source from koharu
  is translated verbatim.

Users of the published ONNX weights remain bound by the comic-text-detector
license (GPL-3.0).
==============================================================================
"""
import os
import json
import math
import logging

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models", "manga_text")

# Hugging Face URL for the comic-text-detector ONNX weights
_CTD_ONNX_URL = "https://huggingface.co/mayocream/comic-text-detector-onnx/resolve/main/comic-text-detector.onnx"
_CTD_FILENAME = "comic-text-detector.onnx"


def _to_bgr(image_tensor):
    """ComfyUI IMAGE [B,H,W,C] float 0-1 RGB -> first frame as HxWx3 uint8 BGR."""
    arr = image_tensor[0].detach().cpu().numpy() if image_tensor.dim() == 4 else image_tensor.detach().cpu().numpy()
    arr = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _mask_to_tensor(mask_uint8):
    """HxW uint8 (0/255) -> [1,H,W] float 0-1 tensor."""
    return torch.from_numpy((mask_uint8 > 0).astype(np.float32)).unsqueeze(0)


def _raised_cosine_weights(w, h, overlap):
    """Centered raised-cosine blending window over a tile of size (w,h).

    Weight == 1 in the interior, smoothly drops to 0 across a band of width
    overlap/2 at the borders. Used to blend overlapping inpainted tiles
    without visible seams.
    Inspired by mayocream/koharu's lama crate.
    """
    half = overlap / 2.0
    if overlap <= 0 or half <= 1e-3:
        return np.ones((h, w), dtype=np.float32)
    weights = np.ones((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dx = min(x, w - 1 - x)
            dy = min(y, h - 1 - y)
            d = min(dx, dy)
            if d < half:
                t = (d / half)
                # 0 at border -> 1 at distance half
                weights[y, x] = 0.5 * (1.0 - math.cos(math.pi * t))
    return weights


class MangaTextRemoval:
    """Detect manga/comic text, derive balloon masks, and erase the text."""

    def __init__(self):
        self.detector = None
        self._lama = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "conf_threshold": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 0.9, "step": 0.05}),
                "nms_threshold": ("FLOAT", {"default": 0.35, "min": 0.1, "max": 0.9, "step": 0.05}),
                "dilate_kernel": ("INT", {"default": 5, "min": 1, "max": 31}),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 1024}),
                "tile_overlap": ("INT", {"default": 128, "min": 32, "max": 256}),
                "detect_balloon": ("BOOLEAN", {"default": True}),
                "return_inpaint": ("BOOLEAN", {"default": True}),
                "extract_text_only": ("BOOLEAN", {"default": True}),
                "diff_threshold": ("INT", {"default": 8, "min": 1, "max": 60}),
                "diff_dilate": ("INT", {"default": 2, "min": 0, "max": 15}),
                "bg_color": ("INT", {"default": 0, "min": 0, "max": 255}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("image", "text_mask", "balloon_mask", "bboxes", "text_only")
    FUNCTION = "remove_text"
    CATEGORY = "image"
    OUTPUT_NODE = True

    # ---------------------------------------------------------------- model IO
    def _download_if_missing(self, filename, url):
        os.makedirs(_MODELS_DIR, exist_ok=True)
        dest = os.path.join(_MODELS_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return dest
        logger.info("[MangaTextRemoval] Downloading %s ...", filename)
        try:
            from huggingface_hub import hf_hub_download
            # prefer hf_hub for robustness
            local = hf_hub_download(
                repo_id="mayocream/comic-text-detector-onnx",
                filename="comic-text-detector.onnx",
                local_dir=_MODELS_DIR,
            )
            return local
        except Exception as e:
            logger.warning("[MangaTextRemoval] hf_hub_download failed (%s), falling back to direct download", e)
        import requests
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        tmp = dest + ".tmp"
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        os.replace(tmp, dest)
        logger.info("[MangaTextRemoval] Saved %s (%d bytes)", dest, downloaded)
        return dest

    def _ensure_detector(self, device):
        if self.detector is None:
            from .ctd_inference import SimpleTextDetector
            model_path = self._download_if_missing(_CTD_FILENAME, _CTD_ONNX_URL)
            self.detector = SimpleTextDetector(model_path, device=device)
        return self.detector

    def _ensure_lama(self):
        if self._lama is None:
            # Package name is `simple_lama_inpainting`; the top-level class is
            # `SimpleLama`. (Older docs sometimes reference `import simple_lama`,
            # which is incorrect for current versions.)
            from simple_lama_inpainting import SimpleLama
            self._lama = SimpleLama()
        return self._lama

    # -------------------------------------------------------------- processing
    def _derive_balloon_mask(self, img_bgr, text_mask, bboxes):
        """Derive a speech-balloon mask via flood-fill from text-block seeds.

        The comic-text-detector model does not directly detect balloons; we
        recover balloon regions by flood-filling from the center of each text
        block with a tolerance on luminance, then taking the area that is
        brighter than the text mask alone. This is a standard OpenCV technique.

        Approach inspired by mayocream/koharu's ctd_inference.py experiment.
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # flood-fill needs a 2-pixel wider canvas to stop at the image border
        fill_canvas = np.zeros((h + 2, w + 2), np.uint8)
        # we will paint filled regions onto this mask
        region = np.zeros((h, w), np.uint8)

        # tolerance: balloons are usually near-uniform backgrounds (white-ish)
        lo_diff, up_diff = 12, 12

        for bb in bboxes:
            cx = int(bb["x"] + bb["w"] / 2)
            cy = int(bb["y"] + bb["h"] / 2)
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            seed_val = int(gray[cy, cx])
            # only seed on bright-ish backgrounds typical of balloons
            if seed_val < 128:
                continue
            tmp = gray.copy()
            mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(
                tmp, mask, (cx, cy),
                newVal=128,
                loDiff=(lo_diff, lo_diff, lo_diff),
                upDiff=(up_diff, up_diff, up_diff),
                flags=4 | (255 << 8),
            )
            # mask has 255 where filled (excluding the 1px border)
            filled = mask[1:h + 1, 1:w + 1]
            # keep only regions that are reasonably balloon-sized
            if filled.sum() // 255 > 200:
                region[filled == 255] = 255

        # balloon = filled background minus the text pixels themselves
        balloon = cv2.subtract(region, text_mask)
        # tidy up: close small holes, remove tiny specks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        balloon = cv2.morphologyEx(balloon, cv2.MORPH_CLOSE, kernel)
        return balloon

    def _inpaint_tile(self, img_bgr, mask_uint8, x0, y0, tw, th, tile_size):
        """Inpaint one reflected-padded tile via simple-lama.

        Returns the inpainted pixels for the effective (tw x th) region.
        """
        h, w = img_bgr.shape[:2]
        # reflect-pad to tile_size x tile_size on the right/bottom edges
        pad_w = tile_size - tw
        pad_h = tile_size - th
        x1, y1 = min(x0 + tw, w), min(y0 + th, h)
        img_tile = img_bgr[y0:y1, x0:x1]
        msk_tile = mask_uint8[y0:y1, x0:x1]
        if pad_w > 0:
            img_tile = cv2.copyMakeBorder(img_tile, 0, 0, 0, pad_w, cv2.BORDER_REFLECT)
            msk_tile = cv2.copyMakeBorder(msk_tile, 0, 0, 0, pad_w, cv2.BORDER_REFLECT)
        pad_h_eff = tile_size - img_tile.shape[0]
        if pad_h_eff > 0:
            img_tile = cv2.copyMakeBorder(img_tile, 0, pad_h_eff, 0, 0, cv2.BORDER_REFLECT)
            msk_tile = cv2.copyMakeBorder(msk_tile, 0, pad_h_eff, 0, 0, cv2.BORDER_REFLECT)

        pil_img = Image.fromarray(cv2.cvtColor(img_tile, cv2.COLOR_BGR2RGB))
        pil_msk = Image.fromarray(msk_tile).convert("L")
        # mask pixels = 255 -> erase
        lama = self._ensure_lama()
        result = lama(pil_img, pil_msk)
        out = cv2.cvtColor(np.array(result.convert("RGB")), cv2.COLOR_RGB2BGR)
        return out[:th, :tw]

    def _inpaint_tiled(self, img_bgr, mask_uint8, tile_size, overlap):
        """Tile-based LaMa inpainting with raised-cosine blending.

        Strategy mirrors mayocream/koharu's lama crate: scan the image in tiles
        of `tile_size` with `overlap` px overlap, skip tiles that contain no
        mask pixels, blend overlapping regions with a raised-cosine window, and
        keep the original pixels outside the mask.
        """
        h, w = img_bgr.shape[:2]
        tile_size = max(32, int(tile_size))
        overlap = min(int(overlap), tile_size - 1)
        stride = tile_size - overlap

        acc = np.zeros((h, w, 3), dtype=np.float32)
        acc_w = np.zeros((h, w), dtype=np.float32)

        y0 = 0
        while y0 < h:
            x0 = 0
            while x0 < w:
                tw = min(tile_size, w - x0)
                th = min(tile_size, h - y0)
                # skip if no mask in this effective region
                if mask_uint8[y0:y0 + th, x0:x0 + tw].sum() == 0:
                    x0 += stride
                    continue
                out = self._inpaint_tile(img_bgr, mask_uint8, x0, y0, tw, th, tile_size)
                weights = _raised_cosine_weights(tw, th, overlap)
                for c in range(3):
                    acc[y0:y0 + th, x0:x0 + tw, c] += out[:, :, c] * weights
                acc_w[y0:y0 + th, x0:x0 + tw] += weights
                x0 += stride
            y0 += stride

        # compose: keep original outside mask, blended inside
        out_img = img_bgr.astype(np.float32).copy()
        m = mask_uint8 > 0
        # where acc_w > 0 inside mask, use blended; where mask but no tile hit (edge),
        # fall back to nearest neighbor fill of the inpainted accumulator
        valid = (acc_w > 0) & m
        for c in range(3):
            layer = acc[:, :, c] / np.maximum(acc_w, 1e-6)
            out_img[:, :, c] = np.where(valid, layer, out_img[:, :, c])
        return out_img.clip(0, 255).astype(np.uint8)

    def _extract_text_only(self, orig_bgr, inpainted_bgr, diff_threshold=8, diff_dilate=2, bg_color=0):
        """Isolate the text layer by pixel-diffing the original and inpainted images.

        `inpainted_bgr` must be produced from a mask that covers *only* the real
        text pixels (NOT a dilated mask). Otherwise LaMa's resynthesis of the
        dilated ring around the text shows up as spurious "text" in the diff.

        Unlike the model's segmentation mask (which is approximate), the pixel
        difference precisely captures every location that actually changed
        during inpainting - i.e. the real text strokes. We composite the
        original pixels at those locations onto a solid background, producing a
        layer that contains only the text.

        Args:
            orig_bgr: HxWx3 uint8 BGR original image.
            inpainted_bgr: HxWx3 uint8 BGR image with ONLY text erased
                (raw mask, no dilation).
            diff_threshold: per-pixel change (summed over BGR) above which a
                pixel is considered "text". Low value = more sensitive.
            diff_dilate: optional dilation of the diff mask (px) to recover
                text-stroke interiors that inpainting reproduced nearly
                verbatim (e.g. faint strokes). 0 disables.
            bg_color: 0-255 background value for the output layer.

        Returns:
            text_only_bgr: HxWx3 uint8 BGR image (text on solid background).
            diff_mask: HxW uint8 (0/255) precise text-pixel mask.
        """
        # per-pixel absolute difference, summed across channels
        diff = cv2.absdiff(orig_bgr, inpainted_bgr)
        diff_sum = diff.sum(axis=2).astype(np.uint16)  # range 0..765
        # threshold -> binary mask of changed pixels
        _, diff_mask = cv2.threshold(diff_sum, int(diff_threshold), 255, cv2.THRESH_BINARY)

        if diff_dilate and diff_dilate > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (int(diff_dilate) * 2 + 1, int(diff_dilate) * 2 + 1)
            )
            diff_mask = cv2.dilate(diff_mask, k, iterations=1)

        # morphological close to fill thin gaps inside glyph strokes
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, close_k)

        # composite: original pixels where mask, solid bg elsewhere
        bg = np.full_like(orig_bgr, int(bg_color))
        m3 = (diff_mask > 0)[..., None]
        text_only = np.where(m3, orig_bgr, bg).astype(np.uint8)
        return text_only, diff_mask

    # ------------------------------------------------------------------ entry
    def remove_text(
        self,
        image,
        conf_threshold=0.4,
        nms_threshold=0.35,
        dilate_kernel=5,
        tile_size=512,
        tile_overlap=128,
        detect_balloon=True,
        return_inpaint=True,
        extract_text_only=True,
        diff_threshold=8,
        diff_dilate=2,
        bg_color=0,
        device="auto",
    ):
        img_bgr = _to_bgr(image)

        detector = self._ensure_detector(device)
        text_mask, bboxes = detector.detect(img_bgr, conf_threshold, nms_threshold)

        # Keep the raw (un-dilated) mask for accurate text_only extraction later.
        # The dilated mask is used for the main inpaint pass to ensure text is
        # fully erased; but the dilated ring is *not* real text and must be
        # excluded from the diff used to isolate the text layer.
        raw_text_mask = text_mask

        # dilate mask to widen the erased area (koharu applies dilate+erode before inpaint)
        if dilate_kernel and dilate_kernel > 1:
            kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
            text_mask = cv2.dilate(text_mask, kernel, iterations=1)

        balloon_mask = (
            self._derive_balloon_mask(img_bgr, text_mask, bboxes)
            if detect_balloon
            else np.zeros_like(text_mask)
        )

        # ---- main inpaint pass (uses the DILATED mask) ----
        need_inpaint = (return_inpaint or extract_text_only) and text_mask.sum() > 0
        if need_inpaint:
            inpainted = self._inpaint_tiled(img_bgr, text_mask, tile_size, tile_overlap)
            out_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
            out_tensor = torch.from_numpy(out_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        else:
            inpainted = None
            out_tensor = image

        if extract_text_only and inpainted is not None:
            # For text isolation we want the diff to reflect ONLY the real text
            # pixels, not the dilated ring that LaMa also resynthesized.
            # So we diff against a reference inpaint that used the RAW mask.
            # When dilate_kernel <= 1 the dilated mask equals the raw mask, so
            # we reuse the main pass without a second LaMa run.
            if raw_text_mask is text_mask:
                ref_inpainted = inpainted
            else:
                ref_inpainted = self._inpaint_tiled(
                    img_bgr, raw_text_mask, tile_size, tile_overlap
                )
            text_only_bgr, _diff_mask = self._extract_text_only(
                img_bgr, ref_inpainted, diff_threshold, diff_dilate, bg_color
            )
            text_only_rgb = cv2.cvtColor(text_only_bgr, cv2.COLOR_BGR2RGB)
            text_only_t = torch.from_numpy(text_only_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        else:
            text_only_t = torch.zeros_like(image)

        text_mask_t = _mask_to_tensor(text_mask)
        balloon_mask_t = _mask_to_tensor(balloon_mask)

        return (out_tensor, text_mask_t, balloon_mask_t,
                json.dumps(bboxes, ensure_ascii=False), text_only_t)
