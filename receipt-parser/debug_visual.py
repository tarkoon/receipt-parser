"""debug_visual.py — Bounding box drawing, field overlay, pipeline trace."""

from dataclasses import dataclass, field
from pathlib import Path
import time
import json

import cv2
import numpy as np

from schema import get_debug_color_map


@dataclass
class PipelineTrace:
    """Records each pipeline step with name, timestamp, elapsed time."""
    steps: list[dict] = field(default_factory=list)
    debug_dir: Path | None = None
    _start_time: float = field(default_factory=time.time)
    _last_time: float = field(default_factory=time.time)

    def log_step(self, name: str, data=None, image: np.ndarray | None = None):
        """Log a pipeline step. Saves artifacts to debug_dir if set."""
        now = time.time()
        elapsed = now - self._last_time
        step_num = len(self.steps) + 1

        self.steps.append({
            "step": step_num,
            "name": name,
            "elapsed": elapsed,
            "timestamp": now,
        })
        self._last_time = now

        if self.debug_dir:
            prefix = f"{step_num:02d}_{name}"
            if image is not None:
                cv2.imwrite(str(self.debug_dir / f"{prefix}.png"), image)
            if data is not None:
                if isinstance(data, str):
                    (self.debug_dir / f"{prefix}.txt").write_text(
                        data, encoding="utf-8"
                    )
                elif isinstance(data, (dict, list)):
                    (self.debug_dir / f"{prefix}.json").write_text(
                        json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

    def summary(self) -> str:
        """Returns a formatted multi-line timing string."""
        lines = ["Pipeline Trace:"]
        total = 0.0
        for step in self.steps:
            elapsed = step["elapsed"]
            total += elapsed
            if step["step"] == 1:
                lines.append(f"  [  start] {step['name']}")
            else:
                lines.append(f"  [+{elapsed:.3f}s] {step['name']}")
        lines.append(f"  Total: {total:.3f}s")
        return "\n".join(lines)


def draw_ocr_bboxes(
    image: np.ndarray,
    ocr_blocks: list[dict],
    output_path: Path,
) -> None:
    """Draw color-coded bounding boxes on the preprocessed image.
    Green (conf >= 0.9), Yellow (>= 0.7), Red (< 0.7).
    """
    # Convert grayscale to BGR for colored drawing
    if len(image.shape) == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    for block in ocr_blocks:
        conf = block["confidence"]
        if conf >= 0.9:
            color = (0, 255, 0)    # Green
        elif conf >= 0.7:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)    # Red

        bbox = block["bbox"]
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)

        # Label with text + confidence
        label = f"{block['text'][:20]} ({conf:.0%})"
        text_x = int(min(p[0] for p in bbox))
        text_y = int(min(p[1] for p in bbox)) - 5
        cv2.putText(canvas, label, (text_x, max(text_y, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def _fuzzy_match_bbox(
    value: str,
    text_to_bbox: dict[str, list],
    threshold: float = 0.6,
) -> list | None:
    """Match an extracted value to the best OCR bounding box.
    Strategy: exact match → substring match → character overlap ratio.
    """
    if value is None:
        return None

    value_str = str(value).strip()
    if not value_str:
        return None

    # Exact match
    if value_str in text_to_bbox:
        return text_to_bbox[value_str]

    # Substring match
    for text, bbox in text_to_bbox.items():
        if value_str in text or text in value_str:
            return bbox

    # Character overlap ratio
    best_ratio = 0.0
    best_bbox = None
    for text, bbox in text_to_bbox.items():
        common = set(value_str) & set(text)
        ratio = len(common) / max(len(set(value_str)), 1)
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_bbox = bbox

    return best_bbox


def draw_field_overlay(
    image: np.ndarray,
    ocr_blocks: list[dict],
    extracted: dict,
    output_path: Path,
) -> None:
    """Map extracted field values back to OCR bboxes using fuzzy matching.
    Colors from schema.get_debug_color_map(). Draws legend in top-left.
    """
    # Convert grayscale to BGR for colored drawing
    if len(image.shape) == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    color_map = get_debug_color_map()
    text_to_bbox = {b["text"]: b["bbox"] for b in ocr_blocks}

    legend_y = 20
    matched_fields = []

    for field_name, color in color_map.items():
        value = extracted.get(field_name)
        if value is None:
            continue

        # Handle list fields (line_items, taxes)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for v in item.values():
                        bbox = _fuzzy_match_bbox(str(v), text_to_bbox)
                        if bbox:
                            pts = np.array(bbox, dtype=np.int32)
                            cv2.polylines(canvas, [pts], True, color, 2)
            matched_fields.append((field_name, color))
        else:
            bbox = _fuzzy_match_bbox(str(value), text_to_bbox)
            if bbox:
                pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(canvas, [pts], True, color, 2)
                # Label
                text_x = int(min(p[0] for p in bbox))
                text_y = int(max(p[1] for p in bbox)) + 15
                cv2.putText(canvas, field_name, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                matched_fields.append((field_name, color))

    # Draw legend in top-left corner
    for field_name, color in matched_fields:
        cv2.rectangle(canvas, (5, legend_y - 12), (20, legend_y), color, -1)
        cv2.putText(canvas, field_name, (25, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        legend_y += 20

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
