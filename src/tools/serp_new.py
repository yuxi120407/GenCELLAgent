from src.config.logging import logger
from src.config.paths import KNOWLEDGE_BASE_PATH
from src.utils.io import load_yaml
from typing import Tuple
from typing import Union
from typing import Dict
from typing import List
from typing import Any 
import requests
import json
import os
import google.generativeai as genai
import re
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    logger.error("GOOGLE_API_KEY environment variable is not set in .env.")

class SerpAPIClient:
    """
    A client for interacting with the SERP API for performing search queries.
    """

    def __init__(self, api_key: str):
        """
        Initialize the SerpAPIClient with the provided API key.

        Parameters:
        -----------
        api_key : str
            The API key for authenticating with the SERP API.
        """
        self.api_key = api_key
        self.base_url = "https://serpapi.com/search.json"

    def __call__(self, query: str, engine: str = "google", location: str = "") -> Union[Dict[str, Any], Tuple[int, str]]:
        """
        Perform Google search using the SERP API.

        Parameters:
        -----------
        query : str
            The search query string.
        engine : str, optional
            The search engine to use (default is "google").
        location : str, optional
            The location for the search query (default is an empty string).

        Returns:
        --------
        Union[Dict[str, Any], Tuple[int, str]]
            The search results as a JSON dictionary if successful, or a tuple containing the HTTP status code
            and error message if the request fails.
        """
        params = {
            "engine": engine,
            "q": query,
            "api_key": self.api_key,
            "location": location
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to SERP API failed: {e}")
            return response.status_code, str(e)


def format_top_search_results(results: Dict[str, Any], top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Format the top N search results into a list of dictionaries with updated key names.

    Parameters:
    -----------
    results : Dict[str, Any]
        The search results returned from the SERP API.
    top_n : int, optional
        The number of top search results to format (default is 10).

    Returns:
    --------
    List[Dict[str, Any]]
        A list of dictionaries containing the formatted top search results with updated key names.
    """
    return [
        {
            "position": result.get('position'),
            "title": result.get('title'),
            "link": result.get('link'),
            "snippet": result.get('snippet')
        }
        for result in results.get('organic_results', [])[:top_n]
    ]


def summarize_with_gemini(top_results: List[Dict[str, Any]], api_key: str) -> str:
    """
    Use Gemini to summarize the top search results.
    """

    model = genai.GenerativeModel('gemini-2.5-flash')

    content = "\n".join(
        f"{i+1}. {r['title']}\n{r['snippet']}" for i, r in enumerate(top_results) if r.get("snippet")
    )

    #prompt = f"Summarize the following search results:\n\n{content}\n\nSummary:"
    prompt = (
        "Analyze the following search results and do two things:\n"
        "1. Summarize the Visual Characteristics described across the content.\n"
        "2. Generate a Segmentation Prompt that could be used to guide a visual segmentation tool "
        "based on those characteristics.\n\n"
        f"{content}\n\n"
        "Output format:\n"
        "### Visual Characteristics Summary ###\n"
        "[your summary here]\n\n"
        "### Segmentation Prompt ###\n"
        "[your segmentation prompt here]"
    )

    try:
        response = model.generate_content(prompt)
        
        # Handle cases where response might be empty or blocked by safety settings
        if not response or not response.parts:
            logger.error("Gemini returned an empty response or was blocked by safety settings.")
            return {
                "visual_characteristics": "Failed to extract characteristics due to an API error or safety block.",
                "segmentation_prompt": "Please manually define a segmentation prompt."
            }
            
        output = response.text.strip()
        
        # Extract both sections using regex
        visual_summary_match = re.search(r"### Visual Characteristics Summary ###\n(.+?)\n###", output, re.DOTALL)
        segmentation_prompt_match = re.search(r"### Segmentation Prompt ###\n(.+)", output, re.DOTALL)

        visual_characteristics = visual_summary_match.group(1).strip() if visual_summary_match else ""
        segmentation_prompt = segmentation_prompt_match.group(1).strip() if segmentation_prompt_match else ""

        # Fallback if regex failed to match the exact format
        if not visual_characteristics and not segmentation_prompt:
            visual_characteristics = output
            segmentation_prompt = output

        return {
            "visual_characteristics": visual_characteristics,
            "segmentation_prompt": segmentation_prompt
        }
    except Exception as e:
        logger.error(f"Error during Gemini summarization in SERP tool: {e}")
        return {
            "visual_characteristics": "Error generating summary.",
            "segmentation_prompt": "Error generating prompt."
        }

 #"top search content":content,


def search(search_query: str, location: str = "") -> str:
    """
    Main function to execute the Google search using SERP API and return the top results as a JSON string.
    Saves the generated visual characteristics to a local memory file.

    Parameters:
    -----------
    search_query : str
        The search query to be executed using the SERP API.
    location : str, optional
        The location to include in the search query (default is an empty string).

    Returns:
    --------
    str
        A JSON string containing the top search results or an error message, with updated key names.
    """

    api_key = os.getenv("SERPAPI_API_KEY")
    
    if not api_key:
        error_json = json.dumps({"error": "SERPAPI_API_KEY environment variable is not set."})
        logger.error(error_json)
        return error_json

    # Initialize clients
    serp_client = SerpAPIClient(api_key)

    # Fetch search results
    results = serp_client(search_query, location=location)

    if isinstance(results, dict):
        try:
            top_results = format_top_search_results(results)
            summary = summarize_with_gemini(top_results, api_key)

            # --- Save to Knowledge Base Memory ---
            try:
                if os.path.exists(KNOWLEDGE_BASE_PATH):
                    with open(KNOWLEDGE_BASE_PATH, 'r') as f:
                        kb = json.load(f)
                else:
                    kb = {}
                
                # Use the raw query as the key, or extract the organelle name if possible
                organelle_key = search_query.replace("visual characteristics of", "").replace("shape texture location structure", "").strip().lower()
                
                kb[organelle_key] = {
                    "visual_characteristics": summary.get("visual_characteristics", ""),
                    "segmentation_prompt": summary.get("segmentation_prompt", "")
                }
                
                with open(KNOWLEDGE_BASE_PATH, 'w') as f:
                    json.dump(kb, f, indent=4)
                logger.info(f"Saved visual characteristics for '{organelle_key}' to knowledge base.")
            except Exception as mem_err:
                logger.error(f"Failed to save to knowledge base: {mem_err}")
            # ------------------------------------

            return json.dumps({
                "top_results": top_results,
                "summary": summary,
            }, indent=2)
        except Exception as e:
            error_json = json.dumps({"error": f"Failed to process search results: {e}"})
            logger.error(error_json)
            return error_json
    else:
        status_code, error_message = results
        error_json = json.dumps({"error": f"Search failed with status code {status_code}: {error_message}"})
        logger.error(error_json)
        return error_json


if __name__ == "__main__":
    search_query = "Best gyros in Barcelona, Spain"
    result_json = search(search_query)
    print(result_json)
