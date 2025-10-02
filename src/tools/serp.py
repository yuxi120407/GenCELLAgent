from src.config.logging import logger
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
# Static paths
CREDENTIALS_PATH = './credentials/key.yml'

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

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


def load_api_key(credentials_path: str) -> str:
    """
    Load the API key from the specified YAML file.

    Parameters:
    -----------
    credentials_path : str
        The path to the YAML file containing the API credentials.

    Returns:
    --------
    str
        The API key extracted from the YAML file.

    Raises:
    -------
    KeyError
        If the 'serp' or 'key' keys are missing in the YAML file.
    """
    config = load_yaml(credentials_path)
    return config['serp']['key']


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

    model = genai.GenerativeModel('gemini-2.0-flash-exp')

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

    response = model.generate_content(prompt)
    output = response.text.strip()
    
    # Extract both sections using regex
    visual_summary_match = re.search(r"### Visual Characteristics Summary ###\n(.+?)\n###", output, re.DOTALL)
    segmentation_prompt_match = re.search(r"### Segmentation Prompt ###\n(.+)", output, re.DOTALL)

    visual_characteristics = visual_summary_match.group(1).strip() if visual_summary_match else ""
    segmentation_prompt = segmentation_prompt_match.group(1).strip() if segmentation_prompt_match else ""

    return {
        "visual_characteristics": visual_characteristics,
        "segmentation_prompt": segmentation_prompt
    }

 #"top search content":content,





def search(search_query: str, location: str = "") -> str:
    """
    Main function to execute the Google search using SERP API and return the top results as a JSON string.

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

    api_key = load_api_key(CREDENTIALS_PATH)

    # Initialize clients
    serp_client = SerpAPIClient(api_key)

    # Fetch search results
    results = serp_client(search_query, location=location)

    if isinstance(results, dict):
        top_results = format_top_search_results(results)
        summary = summarize_with_gemini(top_results, api_key)


        return json.dumps({
            "top_results": top_results,
            "summary": summary,
        }, indent=2)
    else:
        status_code, error_message = results
        error_json = json.dumps({"error": f"Search failed with status code {status_code}: {error_message}"})
        logger.error(error_json)
        return error_json


if __name__ == "__main__":
    search_query = "Best gyros in Barcelona, Spain"
    result_json = search(search_query)
    print(result_json)

