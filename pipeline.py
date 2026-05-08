"""
pipeline.py
-----------
Core end-to-end pipeline for animal image analysis.
Modules:
  1. AnimalDetector   — YOLOv8 bounding-box detection
  2. AnimalClassifier — EfficientNet-B0 species classifier
  3. GradCAMExplainer — Grad-CAM saliency map
  4. AnimalAnalysisPipeline — orchestrates all three, returns a unified result dict
"""

import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── constants ──────────────────────────────────────────────────────────────
IMG_SIZE       = 224
CONF_THRESHOLD = 0.25   # YOLO minimum confidence
IOU_THRESHOLD  = 0.45   # YOLO NMS IoU

CLASS_NAMES = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra']

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════
# 1. DETECTION
# ═══════════════════════════════════════════════════════════════════════════
class AnimalDetector:
    """
    Thin wrapper around YOLOv8n for animal detection.

    Parameters
    ----------
    weights : str
        Path to a fine-tuned YOLO weights file, or 'yolov8n.pt' to use the
        COCO-pretrained nano model (already includes 80 COCO classes; animal
        classes are filtered after inference).
    device : torch.device | str
    """

    COCO_ANIMAL_CLASSES = {
        14: "bird", 15: "cat", 16: "dog", 17: "horse",
        18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
        22: "zebra", 23: "giraffe",
    }

    def __init__(self, weights: str = "yolov8n.pt", device=DEVICE):
        self.model = YOLO(weights)
        self.device = device

    def detect(self, image_bgr: np.ndarray):
        """
        Run YOLO on a BGR image.

        Returns
        -------
        list[dict]
            Each dict has keys: 'box' (x1,y1,x2,y2), 'conf', 'label'
        """
        t0 = time.perf_counter()
        results = self.model(
            image_bgr,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )[0]
        dt = (time.perf_counter() - t0) * 1000  # ms

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls.item())
            label  = self.COCO_ANIMAL_CLASSES.get(cls_id)
            if label is None:               # skip non-animal COCO classes
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            detections.append({
                "box":   (x1, y1, x2, y2),
                "conf":  float(box.conf.item()),
                "label": label,
            })

        return detections, dt

    def best_detection(self, image_bgr: np.ndarray):
        """Return only the highest-confidence detection (or None)."""
        dets, dt = self.detect(image_bgr)
        if not dets:
            return None, dt
        return max(dets, key=lambda d: d["conf"]), dt


# ═══════════════════════════════════════════════════════════════════════════
# 2. CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════
def build_efficientnet(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Construct EfficientNet-B0 with a replaced classification head.

    Parameters
    ----------
    num_classes : int
    pretrained  : bool  load ImageNet weights before replacing the head

    Returns
    -------
    nn.Module  ready to train / infer
    """
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


class AnimalClassifier:
    """
    EfficientNet-B0 species classifier.

    Parameters
    ----------
    weights_path : str | None
        Path to a saved state_dict (.pth). If None, uses random weights —
        useful for testing; in production always supply trained weights.
    num_classes  : int
    class_names  : list[str]
    device       : torch.device
    """

    def __init__(
        self,
        weights_path: str | None = None,
        num_classes: int = len(CLASS_NAMES),
        class_names: list[str] = CLASS_NAMES,
        device=DEVICE,
    ):
        self.device      = device
        self.class_names = class_names
        self.model       = build_efficientnet(num_classes, pretrained=(weights_path is None)).to(device)

        if weights_path:
            state = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(state)
            print(f"[Classifier] Loaded weights from {weights_path}")

        self.model.eval()

        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])

    def predict(self, image_rgb: np.ndarray | Image.Image):
        """
        Classify a single image (full image or cropped ROI).

        Returns
        -------
        dict  with keys: 'class', 'class_idx', 'confidence', 'probabilities', 'latency_ms'
        """
        if isinstance(image_rgb, np.ndarray):
            pil_img = Image.fromarray(image_rgb)
        else:
            pil_img = image_rgb

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
        dt = (time.perf_counter() - t0) * 1000

        probs     = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        class_idx = int(np.argmax(probs))

        return {
            "class":         self.class_names[class_idx],
            "class_idx":     class_idx,
            "confidence":    float(probs[class_idx]),
            "probabilities": probs,
            "latency_ms":    dt,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 3. GRAD-CAM EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════
class GradCAMExplainer:
    """
    Wraps pytorch-grad-cam for EfficientNet-B0.
    Target layer is the last convolutional block (conv_head).
    """

    def __init__(self, classifier: AnimalClassifier):
        self.classifier = classifier
        # EfficientNet-B0 target layer: last conv before pooling
        target_layers   = [classifier.model.conv_head]
        self.cam        = GradCAM(
            model=classifier.model,
            target_layers=target_layers,
        )

    def generate(
        self,
        image_rgb: np.ndarray,
        class_idx: int | None = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap overlay.

        Parameters
        ----------
        image_rgb  : H×W×3 uint8 RGB
        class_idx  : target class (None = predicted class)

        Returns
        -------
        np.ndarray  H×W×3 uint8 heatmap blended on the input image
        """
        transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])
        pil_img = Image.fromarray(image_rgb)
        tensor  = transform(pil_img).unsqueeze(0).to(self.classifier.device)

        targets = [ClassifierOutputTarget(class_idx)] if class_idx is not None else None
        grayscale_cam = self.cam(input_tensor=tensor, targets=targets)[0]

        # Resize Grad-CAM mask back to original image size
        h, w = image_rgb.shape[:2]
        grayscale_cam_resized = cv2.resize(grayscale_cam, (w, h))

        # Normalise image to [0,1] float for blending
        img_float = image_rgb.astype(np.float32) / 255.0
        visualization = show_cam_on_image(img_float, grayscale_cam_resized, use_rgb=True)
        return visualization  # uint8 RGB


# ═══════════════════════════════════════════════════════════════════════════
# 4. FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
class AnimalAnalysisPipeline:
    """
    Orchestrates detection → ROI extraction → classification → Grad-CAM.

    Usage
    -----
    >>> pipeline = AnimalAnalysisPipeline(classifier_weights="weights/clf.pth")
    >>> result   = pipeline.run("path/to/image.jpg")
    >>> print(result["predicted_class"], result["confidence"])
    """

    def __init__(
        self,
        yolo_weights: str = "yolov8n.pt",
        classifier_weights: str | None = None,
        num_classes: int = len(CLASS_NAMES),
        class_names: list[str] = CLASS_NAMES,
        device=DEVICE,
    ):
        print(f"[Pipeline] Device: {device}")
        self.detector   = AnimalDetector(yolo_weights, device)
        self.classifier = AnimalClassifier(classifier_weights, num_classes, class_names, device)
        self.explainer  = GradCAMExplainer(self.classifier)
        self.device     = device

    def run(self, image_input: str | np.ndarray) -> dict:
        """
        Full inference pass.

        Parameters
        ----------
        image_input : file path (str) or BGR numpy array

        Returns
        -------
        dict with keys:
            original_rgb, annotated_rgb, roi_rgb, heatmap_rgb,
            detection, predicted_class, confidence, probabilities,
            detection_ms, classification_ms, total_ms
        """
        t_total = time.perf_counter()

        # ── load image ──────────────────────────────────────────────────
        if isinstance(image_input, str):
            bgr = cv2.imread(image_input)
            if bgr is None:
                raise FileNotFoundError(f"Cannot open image: {image_input}")
        else:
            bgr = image_input  # already BGR

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # ── detect ──────────────────────────────────────────────────────
        detection, det_ms = self.detector.best_detection(bgr)

        if detection is not None:
            x1, y1, x2, y2 = detection["box"]
            # Clamp coordinates
            h_img, w_img = rgb.shape[:2]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w_img, x2); y2 = min(h_img, y2)
            roi_rgb = rgb[y1:y2, x1:x2]
        else:
            # Fallback: classify full image
            roi_rgb = rgb
            detection = {"box": (0, 0, rgb.shape[1], rgb.shape[0]), "conf": 1.0, "label": "unknown"}

        # ── classify ────────────────────────────────────────────────────
        clf_result = self.classifier.predict(roi_rgb)
        clf_ms     = clf_result["latency_ms"]

        # ── Grad-CAM ────────────────────────────────────────────────────
        heatmap_rgb = self.explainer.generate(roi_rgb, class_idx=clf_result["class_idx"])

        # ── annotate original image ─────────────────────────────────────
        annotated = rgb.copy()
        x1, y1, x2, y2 = detection["box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (34, 197, 94), 2)
        label_text = f"{clf_result['class']} {clf_result['confidence']:.1%}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), (34, 197, 94), -1)
        cv2.putText(annotated, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        total_ms = (time.perf_counter() - t_total) * 1000

        return {
            "original_rgb":      rgb,
            "annotated_rgb":     annotated,
            "roi_rgb":           roi_rgb,
            "heatmap_rgb":       heatmap_rgb,
            "detection":         detection,
            "predicted_class":   clf_result["class"],
            "class_idx":         clf_result["class_idx"],
            "confidence":        clf_result["confidence"],
            "probabilities":     clf_result["probabilities"],
            "detection_ms":      det_ms,
            "classification_ms": clf_ms,
            "total_ms":          total_ms,
        }

    def run_batch(self, image_paths: list[str]) -> list[dict]:
        """Run the pipeline on a list of image paths."""
        return [self.run(p) for p in image_paths]


# ── quick smoke test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    pipeline = AnimalAnalysisPipeline()
    result   = pipeline.run(img_path)
    print(f"Predicted : {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Detection : {result['detection_ms']:.1f} ms")
    print(f"Classify  : {result['classification_ms']:.1f} ms")
    print(f"Total     : {result['total_ms']:.1f} ms")
