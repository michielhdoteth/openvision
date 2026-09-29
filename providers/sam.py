"""SAM segmentation provider with on-demand loading."""
import logging
import numpy as np
from pathlib import Path
from typing import Optional

from core.gpu import get_device, unload_model, log_vram

logger = logging.getLogger(__name__)


def create_segmenter(config: dict):
    """Create a text-prompt segmenter matching the configured provider."""
    default_provider = "sam3" if str(config.get("model", "")).lower() in {"sam3", "sam-3"} else "sam"
    provider = str(config.get("provider", default_provider)).lower()
    if provider in {"sam3", "sam-3"} or str(config.get("model", "")).lower() == "sam3":
        return Sam3Provider(config)
    if provider == "sam":
        return SamProvider(config)
    raise ValueError(f"Unknown segmentation provider '{provider}'. Choose 'sam3' or 'sam'.")


class SamProvider:
    """
    SAM segmentation provider.

    Model is loaded on demand (lazy) to avoid occupying VRAM unnecessarily.
    Loading sequence:
    1. Unload VLM models
    2. Clear CUDA cache
    3. Load SAM
    4. Segment
    5. Unload SAM if configured

    Supports: SAM 2.1 (transformers), MobileSAM (ultralytics), FastSAM (ultralytics)
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = config.get("model", "sam-2.1")
        self.fallback = config.get("fallback_model", "mobile-sam")
        self.load_on_demand = config.get("load_on_demand", True)
        self._model = None
        self._processor = None
        self._ultralytics_model = None
        self._loaded = False

    def segment(self, image: np.ndarray, prompt: str) -> list[dict]:
        """
        Segment objects in an image matching the text prompt.

        Args:
            image: RGB numpy array (H, W, 3)
            prompt: Text description of object to segment

        Returns:
            List of dicts: {mask, bbox, score}
        """
        self._ensure_loaded()

        try:
            return self._segment_impl(image, prompt)
        except Exception as e:
            logger.warning(f"SAM segmentation failed: {e}")
            logger.info(f"Attempting fallback model: {self.fallback}")
            self.unload()
            # Try fallback
            original_model = self.model
            self.model = self.fallback
            try:
                self._ensure_loaded()
                return self._segment_impl(image, prompt)
            except Exception as e2:
                logger.error(f"Fallback segmentation also failed: {e2}")
                raise RuntimeError(
                    f"Segmentation failed for configured model '{original_model}' and fallback '{self.fallback}': {e2}"
                ) from e2
            finally:
                self.model = original_model
                if self.config.get("load_on_demand", True):
                    self.unload()

    def segment_frames(
        self,
        frames: list[tuple[float, np.ndarray]],
        prompt: str,
        run_dir: Optional[Path] = None,
    ) -> dict:
        """
        Segment an object across multiple frames.

        Args:
            frames: List of (timestamp, image) tuples
            prompt: Object description
            run_dir: Optional directory to save masks

        Returns:
            dict with: frames_processed, objects (list of object data)
        """
        results = []
        prompt_slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in prompt).strip("_")[:24] or "object"

        for timestamp, image in frames:
            objects = self.segment(image, prompt)
            if objects:
                for object_index, obj in enumerate(objects):
                    obj_data = {
                        "timestamp": f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}",
                        "time_seconds": timestamp,
                        "bbox": obj.get("bbox", []),
                        "score": obj.get("score", 0.0),
                    }

                    if run_dir and "mask" in obj:
                        mask_path = run_dir / "masks" / f"mask_{prompt_slug}_{int(timestamp):08d}_{object_index:03d}.png"
                        self._save_mask(obj["mask"], str(mask_path))
                        obj_data["mask_path"] = str(mask_path)

                    results.append(obj_data)

        return {
            "frames_processed": len(frames),
            "objects": results,
        }

    def _ensure_loaded(self):
        """Load SAM model if not already loaded."""
        if self._loaded:
            return

        log_vram("before_sam_load")

        # Clear CUDA cache before loading
        unload_model()

        logger.info(f"Loading SAM model: {self.model}")

        try:
            if self.model.startswith("sam-2"):
                self._load_sam2()
            else:
                self._load_ultralytics()

            self._loaded = True
            log_vram("after_sam_load")
        except ImportError as e:
            logger.error(f"Failed to load {self.model}: {e}")
            logger.error("Install with: pip install transformers ultralytics")
            raise
        except Exception as e:
            logger.error(f"Error loading {self.model}: {e}")
            raise

    def _load_sam2(self):
        """Load SAM 2.x from transformers."""
        from transformers import SamModel, SamProcessor

        device = get_device()
        model_id = "facebook/sam-vit-huge"

        self._processor = SamProcessor.from_pretrained(model_id)
        self._model = SamModel.from_pretrained(model_id).to(device)
        self._model.eval()
        logger.info(f"SAM 2 loaded on {device}")

    def _load_ultralytics(self):
        """Load MobileSAM or FastSAM from ultralytics."""
        from ultralytics import SAM

        device = get_device()
        if "mobile" in self.model:
            model_name = "mobile_sam.pt"
        elif "fast" in self.model.lower():
            model_name = "FastSAM-x.pt"
        else:
            model_name = "mobile_sam.pt"

        self._ultralytics_model = SAM(model_name)
        self._ultralytics_model.to(device)
        logger.info(f"Ultralytics SAM loaded on {device}")

    def _segment_impl(self, image: np.ndarray, prompt: str) -> list[dict]:
        """Run segmentation on image with given prompt."""
        if hasattr(self, '_ultralytics_model') and self._ultralytics_model is not None:
            return self._segment_ultralytics(image, prompt)
        elif hasattr(self, '_model') and self._model is not None:
            return self._segment_transformers(image, prompt)
        else:
            logger.error("No SAM model loaded")
            return []

    def _segment_ultralytics(self, image: np.ndarray, prompt: str) -> list[dict]:
        """Segment using ultralytics SAM."""
        results = self._ultralytics_model(image, text=prompt)

        objects = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            masks = results[0].masks

            if boxes is not None:
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].tolist()
                    score = float(boxes.conf[i]) if boxes.conf is not None else 1.0
                    obj = {
                        "bbox": bbox,
                        "score": score,
                    }
                    if masks is not None and i < len(masks):
                        obj["mask"] = masks.data[i].cpu().numpy()
                    objects.append(obj)

        return objects

    def _segment_transformers(self, image: np.ndarray, prompt: str) -> list[dict]:
        """Segment using transformers SAM."""
        import torch

        inputs = self._processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Process outputs
        predicted_masks = outputs.pred_masks
        iou_scores = outputs.iou_scores

        objects = []
        for i in range(min(5, predicted_masks.shape[1])):
            mask = predicted_masks[0, i].cpu().numpy()
            score = float(iou_scores[0, i].cpu())

            # Convert mask to binary
            binary_mask = (mask > 0.0).astype(np.uint8)

            # Get bounding box from mask
            rows = np.any(binary_mask, axis=1)
            cols = np.any(binary_mask, axis=0)
            if rows.any() and cols.any():
                y1, y2 = np.where(rows)[0][[0, -1]]
                x1, x2 = np.where(cols)[0][[0, -1]]
                bbox = [float(x1), float(y1), float(x2), float(y2)]
            else:
                bbox = [0, 0, 0, 0]

            objects.append({
                "mask": binary_mask,
                "bbox": bbox,
                "score": score,
            })

        return objects

    def _save_mask(self, mask: np.ndarray, path: str):
        """Save a binary mask as PNG."""
        import cv2
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        # Scale to 0-255 for saving
        if mask.max() <= 1:
            mask_img = (mask * 255).astype(np.uint8)
        else:
            mask_img = mask.astype(np.uint8)
        cv2.imwrite(str(path_obj), mask_img)

    def unload(self):
        """Unload SAM model from GPU memory."""
        self._model = None
        self._processor = None
        self._ultralytics_model = None
        self._loaded = False
        unload_model()
        logger.info("SAM unloaded from GPU")

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class Sam3Provider:
    """
    SAM 3 segmentation provider using the official Meta SAM 3 repo.

    SAM 3 uses text prompts for segmentation.
    Requires: Python 3.12+, torch 2.7+, and the sam3 package installed.
    Falls back to SamProvider if SAM 3 is not available.

    Note: SAM 3 checkpoint requires Hugging Face access at:
    https://huggingface.co/facebook/sam3
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.processor = None
        self._loaded = False

    def segment(self, image: np.ndarray, prompt: str) -> list[dict]:
        """
        Segment using SAM 3 with text prompt.

        Args:
            image: RGB numpy array (H, W, 3)
            prompt: Text description of object to segment

        Returns:
            List of dicts: {mask, bbox, score}
        """
        self._ensure_loaded()

        # Convert numpy to PIL
        from PIL import Image
        pil_image = Image.fromarray(image)

        inference_state = self.processor.set_image(pil_image)
        output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)

        masks = output.get("masks", [])
        boxes = output.get("boxes", [])
        scores = output.get("scores", [])

        results = []
        for i in range(len(masks)):
            mask = masks[i]
            if hasattr(mask, "detach"):
                mask = mask.detach().cpu().numpy()
            elif hasattr(mask, "cpu"):
                mask = mask.cpu().numpy()
            bbox = boxes[i].detach().cpu().tolist() if hasattr(boxes[i], "detach") else boxes[i].tolist() if hasattr(boxes[i], "tolist") else boxes[i]
            score_value = scores[i]
            score = float(score_value.detach().cpu().item() if hasattr(score_value, "detach") else score_value)

            results.append({
                "mask": mask,
                "bbox": bbox,
                "score": score,
            })

        return results

    def segment_frames(self, frames: list[tuple[float, np.ndarray]], prompt: str,
                       run_dir: Optional[Path] = None) -> dict:
        """Apply text-prompt segmentation to timestamped frames and persist masks."""
        prompt_slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in prompt).strip("_")[:24] or "object"
        objects = []
        for timestamp, image in frames:
            for index, item in enumerate(self.segment(image, prompt)):
                obj = {"timestamp": f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}",
                       "time_seconds": timestamp, "bbox": item.get("bbox", []),
                       "score": item.get("score", 0.0)}
                if run_dir and item.get("mask") is not None:
                    mask_path = run_dir / "masks" / f"mask_{prompt_slug}_{int(timestamp):08d}_{index:03d}.png"
                    self._save_mask(np.asarray(item["mask"]), mask_path)
                    obj["mask_path"] = str(mask_path)
                objects.append(obj)
        return {"frames_processed": len(frames), "objects": objects}

    @staticmethod
    def _save_mask(mask: np.ndarray, path: Path) -> None:
        import cv2
        path.parent.mkdir(parents=True, exist_ok=True)
        mask = np.asarray(mask).squeeze()
        if mask.ndim != 2:
            raise ValueError(f"Expected a 2D segmentation mask, got shape {mask.shape}")
        image = (mask * 255).astype(np.uint8) if mask.max(initial=0) <= 1 else mask.astype(np.uint8)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not save segmentation mask to {path}")

    def _ensure_loaded(self):
        """Load SAM 3 model if not already loaded."""
        if self._loaded:
            return

        from core.gpu import unload_model as gpu_unload, log_vram as gpu_log_vram

        gpu_log_vram("before_sam3_load")

        # Clear CUDA cache before loading
        gpu_unload()

        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model)
            self._loaded = True

            gpu_log_vram("after_sam3_load")
            logger.info("SAM 3 loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to load SAM 3: {e}")
            logger.error("SAM 3 is not bundled: its upstream NumPy constraint conflicts with OpenVision's NumPy 2 requirement.")
            logger.error("Install a compatible SAM 3 build in a separate environment and configure its checkpoint/access first.")
            raise
        except Exception as e:
            logger.error(f"Error loading SAM 3 model: {e}")
            raise

    def unload(self):
        """Unload SAM 3 model from GPU memory."""
        self.model = None
        self.processor = None
        self._loaded = False
        from core.gpu import unload_model
        unload_model()
        logger.info("SAM 3 unloaded from GPU")

    @property
    def is_loaded(self) -> bool:
        return self._loaded
