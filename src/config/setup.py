from src.config.logging import logger
from typing import Dict
from typing import Any
import yaml
import os


class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls)
            # The following line ensures that the __init__ method is only called once.
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, config_path: str = None):
        """
        Initialize the Config class.

        Args:
        - config_path (str): Path to the YAML configuration file.
        """
        if self.__initialized:
            return
        self.__initialized = True

        # Use absolute path based on this file's location
        if config_path is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(repo_root, "config", "config.yml")

        self.__config = self._load_config(config_path)
        if self.__config is None:
            # Use default values if config file is missing
            logger.warning("Using default configuration values")
            self.__config = {
                'project_id': os.getenv('GOOGLE_PROJECT_ID', 'your-project-id'),
                'region': os.getenv('GOOGLE_REGION', 'us-central1'),
                'credentials_json': os.getenv('GOOGLE_APPLICATION_CREDENTIALS', ''),
                'model_name': os.getenv('MODEL_NAME', 'gemini-3-flash-preview')
            }

        self.PROJECT_ID = self.__config.get('project_id', 'your-project-id')
        self.REGION = self.__config.get('region', 'us-central1')
        self.CREDENTIALS_PATH = self.__config.get('credentials_json', '')
        if self.CREDENTIALS_PATH:
            self._set_google_credentials(self.CREDENTIALS_PATH)
        self.MODEL_NAME = self.__config.get('model_name', 'gemini-3-flash-preview')

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """
        Load the YAML configuration from the given path.

        Args:
        - config_path (str): Path to the YAML configuration file.

        Returns:
        - dict: Loaded configuration data.
        """
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load the configuration file. Error: {e}")

    @staticmethod
    def _set_google_credentials(credentials_path: str) -> None:
        """
        Set the Google application credentials environment variable.

        Args:
        - credentials_path (str): Path to the Google credentials file.
        """
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path


config = Config()