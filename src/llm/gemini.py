from google import genai
from google.genai import types
from src.config.logging import logger
from typing import Optional
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
client = genai.Client(api_key=GOOGLE_API_KEY)


def generate(model_name: str, contents) -> Optional[str]:
    try:
        logger.info("Generating response from Gemini")
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                top_p=1.0,
                candidate_count=1,
                max_output_tokens=8192,
            ),
        )

        if not response.text:
            logger.error("Empty response from the model")
            return None

        logger.info("Successfully generated response")
        return response.text
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return None
