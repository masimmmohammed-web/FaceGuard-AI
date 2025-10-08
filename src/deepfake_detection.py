from pathlib import Path
import argparse
import math

import cv2
import numpy as np
from PIL import Image

from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import torch.nn.functional as F

MODEL_ID = "prithivMLmods/Deepfake-vs-Real-8000"
LABELS = {0: "Deepfake", 1: "Real one"}

def load_model(model_id: str = MODEL_ID):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    return processor, model

@torch.no_grad()
def predict_image(pil_img: Image.Image, processor, model):
    inputs = processor(images=pil_img.convert("RGB"), return_tensors="pt")
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1).cpu().numpy().squeeze().tolist()
    return {LABELS[i]: float(probs[i]) for i in range(len(probs))}

def classify_with_threshold(probs: dict, fake_threshold: float = 0.90, uncertain_band: float = 0.60):
    p_fake = probs["Deepfake"]
    if p_fake >= fake_threshold:
        return "Deepfake (high-conf)"
    if p_fake >= uncertain_band:
        return "Likely deepfake (uncertain)"
    if (1 - p_fake) >= fake_threshold:
        return "Real (high-conf)"
    return "Likely real (uncertain)"

def detect_faces_cv2(pil_img: Image.Image,
                     scale_factor: float = 1.1,
                     min_neighbors: int = 5,
                     min_size: int = 60):
    """Return list of face boxes (x1,y1,x2,y2,score≈1.0) using OpenCV Haar cascade."""
    gray = np.array(pil_img.convert("L"))
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade at {cascade_path}")

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    boxes = []
    for (x, y, w, h) in faces:
        boxes.append((int(x), int(y), int(x + w), int(y + h), 1.0))
    return boxes

def expand_box(box, img_w, img_h, margin=0.25):
    x1, y1, x2, y2, score = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2, y1 + bh / 2
    nw, nh = bw * (1 + 2 * margin), bh * (1 + 2 * margin)
    nx1 = int(max(0, math.floor(cx - nw / 2)))
    ny1 = int(max(0, math.floor(cy - nh / 2)))
    nx2 = int(min(img_w, math.ceil(cx + nw / 2)))
    ny2 = int(min(img_h, math.ceil(cy + nh / 2)))
    return (nx1, ny1, nx2, ny2, score)

def annotate_and_save(pil_img: Image.Image, whole_pred: str, whole_probs: dict,
                      face_infos: list, out_path: Path):
    img = np.array(pil_img.convert("RGB"))[:, :, ::-1].copy()  # RGB->BGR
    header = f"[Whole] {whole_pred} | Deepfake={whole_probs['Deepfake']:.2%} Real={whole_probs['Real one']:.2%}"
    cv2.rectangle(img, (5, 5), (5 + 10 + 8 * len(header), 35), (0, 0, 0), -1)
    cv2.putText(img, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    for i, info in enumerate(face_infos, 1):
        x1, y1, x2, y2, score = info["box"]
        p_fake = info["probs"]["Deepfake"]
        label = info["label_str"]
        color = (0, 0, 255) if p_fake >= 0.5 else (0, 200, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        txt = f"[Face {i}] {label} | Fake={p_fake:.2%}"
        cv2.rectangle(img, (x1, max(0, y1 - 24)), (x1 + 8 * len(txt), y1), color, -1)
        cv2.putText(img, txt, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return out_path

def process_image(image_path: Path, output_path: Path, fake_threshold=0.90, uncertain_band=0.60,
                  face_margin=0.25, min_face_px=60, haar_scale=1.1, haar_neighbors=5):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    pil_img = Image.open(image_path).convert("RGB")
    W, H = pil_img.size

    processor, model = load_model()

    # Whole image
    whole_probs = predict_image(pil_img, processor, model)
    whole_pred = classify_with_threshold(whole_probs, fake_threshold, uncertain_band)

    # Faces (OpenCV Haar)
    boxes = detect_faces_cv2(pil_img, scale_factor=haar_scale, min_neighbors=haar_neighbors, min_size=min_face_px)
    face_infos = []
    for box in boxes:
        x1, y1, x2, y2, score = expand_box(box, W, H, margin=face_margin)
        crop = pil_img.crop((x1, y1, x2, y2))
        probs = predict_image(crop, processor, model)
        label_str = classify_with_threshold(probs, fake_threshold, uncertain_band)
        face_infos.append({"box": (x1, y1, x2, y2, score), "probs": probs, "label_str": label_str})

    any_face_high_fake = any(info["probs"]["Deepfake"] >= fake_threshold for info in face_infos)
    overall = "Deepfake (high-conf via face)" if any_face_high_fake else whole_pred

    print(f"\n=== RESULTS for {image_path.name} ===")
    print(f"[Whole image] {whole_pred} | Deepfake={whole_probs['Deepfake']:.2%}  Real={whole_probs['Real one']:.2%}")
    if not face_infos:
        print("No faces detected with OpenCV Haar.")
    else:
        for i, info in enumerate(face_infos, 1):
            p = info["probs"]
            print(f"[Face {i}] {info['label_str']} | Deepfake={p['Deepfake']:.2%}  Real={p['Real one']:.2%}  "
                  f"Box={info['box'][:4]}")

    print(f"\nOverall decision: {overall}")

    out = annotate_and_save(pil_img, whole_pred, whole_probs, face_infos, output_path)
    print(f"Annotated output saved to: {out}")

if _name_ == "_main_":
    parser = argparse.ArgumentParser(description="Detect deepfakes on whole image and cropped faces (OpenCV Haar).")
    parser.add_argument("image", type=str, help="Path to an input image (jpg/png).")
    parser.add_argument("--out", type=str, default=None, help="Path to save annotated image.")
    parser.add_argument("--fake-threshold", type=float, default=0.90, help="Confidence to call deepfake.")
    parser.add_argument("--uncertain-band", type=float, default=0.60, help="Uncertain band for 'likely' messages.")
    parser.add_argument("--face-margin", type=float, default=0.25, help="Extra margin around face crop (fraction).")
    parser.add_argument("--haar-scale", type=float, default=1.1, help="Haar scaleFactor (lower finds more faces).")
    parser.add_argument("--haar-neighbors", type=int, default=5, help="Haar minNeighbors (lower finds more, noisier).")
    parser.add_argument("--min-face-px", type=int, default=60, help="Minimum face size (pixels).")
    args = parser.parse_args()

    img_path = Path(args.image)
    out_path = Path(args.out) if args.out else img_path.with_name(img_path.stem + "_annotated.jpg")

    process_image(
        image_path=img_path,
        output_path=out_path,
        fake_threshold=args.fake_threshold,
        uncertain_band=args.uncertain_band,
        face_margin=args.face_margin,
        min_face_px=args.min_face_px,
        haar_scale=args.haar_scale,
        haar_neighbors=args.haar_neighbors,
    )
