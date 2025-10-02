import os
import re
from pathlib import Path
from typing import List


def txt_from_each_subdir_sorted(folder_path: str) -> List[str]:
    base = Path(folder_path)
    txt_files: List[Path] = []

    for subdir in base.iterdir():
        if subdir.is_dir():
            matches = list(subdir.glob("*.txt"))
            if matches:
                txt_files.append(matches[0])

    # Sort by timestamp from folder name
    txt_files_sorted = sorted(
        txt_files,
        key=lambda p: p.parent.name  # assumes folder name = timestamp
    )

    # Convert PosixPath to str
    txt_files_str = [str(p) for p in txt_files_sorted]

    return txt_files_str

def txt_from_each_subdir_sorted_personlization(folder_path: str) -> List[str]:
    base = Path(folder_path)
    txt_files: List[Path] = []

    for subdir in base.iterdir():
        if subdir.is_dir():
            matches = list(subdir.glob("summary.txt"))
            if matches:
                txt_files.append(matches[0])

    # Sort by timestamp from folder name
    txt_files_sorted = sorted(
        txt_files,
        key=lambda p: p.parent.name  # assumes folder name = timestamp
    )

    # Convert PosixPath to str
    txt_files_str = [str(p) for p in txt_files_sorted]

    return txt_files_str

def history_lookup(history_file_path: str):
    """
    Extract reference image path, reference mask path, object name, and visual characteristics
    from a saved structured history file.

    Args:
        history_file_path (str): Path to the saved history .txt file.

    Returns:
        dict: {
            "found": bool,
            "object_name": str or None,
            "reference_image_path": str or None,
            "reference_mask_path": str or None,
            "visual_characteristics": str or None
        }
    """
    if not os.path.exists(history_file_path):
        return {
            "found": False,
            "object_name": None,
            "reference_image_path": None,
            "reference_mask_path": None,
            "visual_characteristics": None
        }

    reference_image_path = None
    reference_mask_path = None
    visual_characteristics = None
    object_name = None

    with open(history_file_path, 'r') as f:
        content = f.read()

    # Extract reference image path
    img_path_match = re.search(r"\*\*Reference Image Path:\*\* `(.*?)`", content)
    if img_path_match:
        reference_image_path = img_path_match.group(1)

    # Extract reference mask path
    mask_path_match = re.search(r"\*\*Reference Mask Path:\*\* `(.*?)`", content)
    if mask_path_match:
        reference_mask_path = mask_path_match.group(1)

    # Extract object name and visual characteristics
    object_vis_match = re.search(r"\*\*Object Name & Visual Characteristics:\*\* (.*?)\n", content)
    if object_vis_match:
        full_text = object_vis_match.group(1).strip()
        visual_characteristics = full_text
        # Assume object name is the first part before the first comma
        if ',' in full_text:
            object_name = full_text.split(',', 1)[0].strip()
        else:
            object_name = full_text.strip()

    found = bool(reference_image_path or reference_mask_path or visual_characteristics)

    return {
        "found": found,
        "object_name": object_name,
        "reference_image_path": reference_image_path,
        "reference_mask_path": reference_mask_path,
        "visual_characteristics": visual_characteristics
    }


def load_last_summary(output_base_dir):
    summary_path = txt_from_each_subdir_sorted_personlization(output_base_dir)[-1]
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            content = f.read().strip()
            if content:
                return content
    else:
        return ""
    

def extract_hitl_summaries(output_base_dir):
    results = []

    for folder in sorted(os.listdir(output_base_dir)):
        summary_path = os.path.join(output_base_dir, folder, "summary.txt")

        if not os.path.exists(summary_path):
            continue

        with open(summary_path, "r") as f:
            content = f.read()

        # Use regex to extract HITL Mode and Reason
        match = re.search(
                r"\[CURRENT RUN\]\s*Recommended HITL Mode:\s*(.*?)\s*Reason:\s*(.*?)(?:\n---|\[OVERALL RECOMMENDATION\]|$)",
                content,
                re.DOTALL | re.IGNORECASE
            )

        if match:
            mode = match.group(1).strip()
            reason = match.group(2).strip()
            results.append(f"Timestamp: {folder}\nRecommended HITL Mode: {mode}\nReason: {reason}\n")

    return "\n".join(results)