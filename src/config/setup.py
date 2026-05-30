import yaml
import os
from src.config.logging import logger


class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, config_path: str = None):
        if self.__initialized:
            return
        self.__initialized = True

        if config_path is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(repo_root, "config", "config.yml")

        self.__config = self._load_config(config_path) or {}
        models = self.__config.get('models', {})
        checkpoints = self.__config.get('checkpoints', {})

        # LLM model names
        self.MODEL_AGENT = models.get('agent', 'gemini-3-flash-preview')
        self.MODEL_SAM3_SEGMENT = models.get('sam3_segment', 'gemini-3-flash-preview')
        self.MODEL_VLM_EVAL = models.get('vlm_eval', 'gemini-3-flash-preview')
        self.MODEL_SEG_EVAL = models.get('seg_eval', 'gemini-3-flash-preview')
        self.MODEL_SUMMARIZER = models.get('summarizer', 'gemini-3-flash-preview')
        self.MODEL_SEARCH = models.get('search', 'gemini-3-flash-preview')

        # Checkpoints (resolve relative to repo root)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sam3_path = checkpoints.get('sam3', 'src/sam3/checkpoints/sam3/sam3.pt')
        self.SAM3_CHECKPOINT = os.path.join(repo_root, sam3_path) if not os.path.isabs(sam3_path) else sam3_path

    @staticmethod
    def _load_config(config_path: str):
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return None


config = Config()
