import subprocess
from src.config.logging import logger
import google.generativeai as genai
from pathlib import Path
import os
from PIL import Image
import json


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def Prompt_generated(process_text: str) -> str:
    """
    Generates a prompt report by incorporating the provided process text and instructing the model
    to extract reference details, including the image path, mask path, and object name.

    Args:
        process_text (str): A detailed text including the process history, visual characteristics,
                            segmentation process, evaluation outcomes, and any embedded reference details.

    Returns:
        str: The complete prompt with instructions to extract required details.
    """
    prompt = f"""
Documentation & Reporting

Process History
{process_text}

Reference Detail Extraction

Identify the reference image path in the process history.
Identify the reference mask path in the process history.
Identify the object name that captures the visual characteristics.

Additional Context

Visual characteristics of the objective.
Names and execution order of all tools used and the performance of each tool.
Outcomes of the segmentation evaluation.

This report template ensures reproducibility and transparency of the entire segmentation workflow.
    """
    return prompt


def summarizer_report(save_dir: str, process_text: str) -> str:
    """
    Generates a summary report based on the provided process text. The prompt instructs the LLM to
    extract reference image and mask paths as well as the object name from within the process text.

    Args:
        process_text (str): A detailed summary including visual characteristics, segmentation process,
                            evaluation outcomes, and possibly embedded reference details.

    Returns:
        str: The generated report text or an error message in case of failure.
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        prompt = Prompt_generated(process_text)
        response = model.generate_content([prompt])
        
        # Save to file
        output_path = save_dir +"/summarized_report.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        
        
        return response.text

    except Exception as e:
        logger.error(f"Error generating content: {e}")
        return {"error": str(e)}
