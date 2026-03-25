import os

# Root of the cloned repository (two levels up from src/config/paths.py)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Icons are stored in src/config/
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_IMAGES_DIR = os.path.join(REPO_ROOT, "output_images")
IMG_EXAMPLE_DIR = os.path.join(REPO_ROOT, "examples")
PLANNING_ICON_PATH = os.path.join(_CONFIG_DIR, "planning_icon.png")
SEGMENTATION_ICON_PATH = os.path.join(_CONFIG_DIR, "segmentation_icon.png")
MITOCHONDRIA_ICON_PATH = os.path.join(_CONFIG_DIR, "mitochondria_icon.png")
PROMPT_TEMPLATE_PATH = os.path.join(REPO_ROOT, "prompt", "react.txt")
PLANNING_PROMPT_TEMPLATE_PATH = os.path.join(REPO_ROOT, "prompt", "planning.txt")
SUMMARIZER_PROMPT_TEMPLATE_PATH = os.path.join(REPO_ROOT, "prompt", "summarizer_hitl_recommendation.txt")
KNOWLEDGE_BASE_PATH = os.path.join(REPO_ROOT, "output", "memory", "knowledge_base.json")
