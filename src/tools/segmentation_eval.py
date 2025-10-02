import subprocess
from src.config.logging import logger
import google.generativeai as genai
from pathlib import Path
import os
from PIL import Image
import json
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())  # auto-detects nearest .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# def Prompt_generated(visual_characteristics):
#     """
#     Generates a prompt with the specified visual characteristics inserted 
#     after the phrase 'visual characteristics:'.

#     Args:
#         visual_characteristics (str): The text to insert after 'visual characteristics:'.

#     Returns:
#         str: The complete prompt with the visual characteristics inserted.
#     """
#     prompt = f"""
# Please evaluate the segmentation results (shown in red areas) and quantify the performance with a score between 0 and 100, based on the following visual characteristics: {visual_characteristics}. The evaluation criteria are divided into four perspectives:
# 1. **Shape Accuracy**: How closely the segmented regions match the true shape of the objects.  
# 2. **Localization**: How accurately the segmentation identifies the correct locations of the objects.  
# 3. **Completeness**: How well the segmentation captures the full extent of the objects, avoiding under- or over-segmentation.  
# 4. **Boundary Precision**: How precise and smooth the boundaries of the segmented regions are, avoiding jagged or inaccurate edges.

# **Respond in the following structure:**

# {{
#   "ReviewScore": {{
#     "ShapeAccuracy": <score out of 100>,
#     "Localization": <score out of 100>,
#     "Completeness": <score out of 100>,
#     "BoundaryPrecision": <score out of 100>,
#     "OverallScore": "(score of ShapeAccuracy + score of Localization + score of Completeness + score of BoundaryPrecision) / 4",
#   }},
#   "Reasons": {{
#     "ShapeAccuracy": "<specific reasoning for the Shape Accuracy score>",
#     "Localization": "<specific reasoning for the Localization score>",
#     "Completeness": "<specific reasoning for the Completeness score>",
#     "BoundaryPrecision": "<specific reasoning for the Boundary Precision score>"
#   }},
#   "SummaryOfReasons": "<Overall details evaluation summarizing the reasons behind the scores>"
# }}
#         """
#     return prompt

# def criterion_prompt_generate(visual_characteristics):
#     contents =  f"""
#     Since we do not have ground truth, please define a segmentation performance evaluation criterion exclusively based on qualitative visual characteristics: {visual_characteristics}. The criteria should be formulated specifically for a Large Visual-Language Model (LVLM) to assess segmentation quality based on perceptual and interpretative reasoning.

#     The evaluation should consist of four or five qualitative perspectives, each assigned a weight (0-1) that represents its relative importance in the overall assessment. The sum of all weights should equal 1 to ensure a balanced criterion distribution.
#     **Respond in the following structure:**

#      {{
#       "Perspectives": {{
#         "<Perspective 1>": {{
#           "Description": "<Qualitative criterion and how it helps assess segmentation quality>",
#           "Weight": <Weight between 0-1, based on importance>
#         }},
#         "<Perspective 2>": {{
#           "Description": "<Qualitative criterion and how it helps assess segmentation quality>",
#           "Weight": <Weight between 0-1, based on importance>
#         }},
#         "<Perspective 3>": {{
#           "Description": "<Qualitative criterion and how it helps assess segmentation quality>",
#           "Weight": <Weight between 0-1, based on importance>
#         }},
#         "<Perspective 4>": {{
#           "Description": "<Qualitative criterion and how it helps assess segmentation quality>",
#           "Weight": <Weight between 0-1, based on importance>
#         }},
#         "<Perspective 5 (optional)>": {{
#           "Description": "<Qualitative criterion and how it helps assess segmentation quality>",
#           "Weight": <Weight between 0-1, based on importance>
#         }}
#       }},
#     ,
#     """
#     return contents

def criterion_prompt_generate(visual_characteristics):
    contents = f"""
    We don’t have ground truth labels for segmentation, so your task is to create a way to judge how good a segmentation result is based only on how it looks. Focus on these visual aspects: {visual_characteristics}.

    Please create **4 to 5 simple and clear judging points** (called "perspectives") that help decide if the segmentation looks good. These points should be easy to understand and focus on things a person might notice, such as neatness, completeness, or how well the parts match what you expect.

    For each perspective:
    - Give it a short, clear name.
    - Explain in plain language what it means and why it matters.
    - Give it a number between 0 and 1 (called a weight) to show how important it is. All weights must add up to exactly 1.

    **Please reply using this format:**

    {{
      "Perspectives": {{
        "<Perspective 1 Name>": {{
          "Description": "<Simple explanation of what this means and how it helps judge quality>",
          "Weight": <Number from 0 to 1>
        }},
        "<Perspective 2 Name>": {{
          "Description": "<Simple explanation of what this means and how it helps judge quality>",
          "Weight": <Number from 0 to 1>
        }},
        "<Perspective 3 Name>": {{
          "Description": "<Simple explanation of what this means and how it helps judge quality>",
          "Weight": <Number from 0 to 1>
        }},
        "<Perspective 4 Name>": {{
          "Description": "<Simple explanation of what this means and how it helps judge quality>",
          "Weight": <Number from 0 to 1>
        }},
        "<Perspective 5 (Optional)>": {{
          "Description": "<Simple explanation of what this means and how it helps judge quality>",
          "Weight": <Number from 0 to 1>
        }}
      }}
    }}
    """
    return contents


def generate_response_prompt(criterion_dict):
    perspectives = criterion_dict["Perspectives"]

    review_score_lines = []
    reasons_lines = []

    for name, props in perspectives.items():
        weight = props.get("Weight", "N/A")
        review_score_lines.append(f'    "{name}" (Weight: {weight}): <score out of 100>,')
        reasons_lines.append(f'    "{name}" (Weight: {weight}): "<specific reasoning for the {name} score>",')

    review_score_block = "\n".join(review_score_lines)
    reasons_block = "\n".join(reasons_lines)

    response_prompt = f"""
Based on the previously defined evaluation criterion, please now provide an evaluation for a segmentation result.

**Respond in the following structure:**

{{
  "ReviewScore": {{
{review_score_block}
    "OverallScore": <weighted average score calculated from individual perspective scores and weights>
  }},
  "Reasons": {{
{reasons_block}
  }},
  "SummaryOfReasons": "<Overall detailed evaluation summarizing the reasons behind the scores>"
}}
"""
    return response_prompt

# def Prompt_generated(criteria):
#     """
#     Generates a prompt with the specified visual characteristics inserted 
#     after the phrase 'visual characteristics:'.

#     Args:
#         visual_characteristics (str): The text to insert after 'visual characteristics:'.

#     Returns:
#         str: The complete prompt with the visual characteristics inserted.
#     """
#     prompt = f"""
# Please evaluate the segmentation results (shown in red areas) and quantify the performance with a score between 0 and 100, based on the following evaluation criteria: {criteria}
#         """
#     return prompt

def Prompt_generated(criteria: str) -> str:
    """
    Generates a detailed and consistent evaluation prompt using the provided criteria,
    with a clear instruction to differentiate scores based on input quality.
    
    Args:
        criteria (str): Evaluation guidelines to be inserted into the prompt.
    
    Returns:
        str: A prompt for scoring segmentation performance with emphasis on meaningful score variation.
    """
    prompt = (
        "You are provided with a segmentation result, where the segmented areas are shown in red. "
        "Carefully evaluate the quality of this result and assign a score between 0 and 100. "
        "Use the following evaluation criteria:\n\n"
        f"{criteria}\n\n"
        "Your score should reflect the actual quality of the result. "
        "Please avoid assigning similar or generic scores to all inputs — "
        "make sure your score meaningfully distinguishes between high-quality, average, and poor segmentations."
    )
    return prompt


# def refine_segmentation_prompt(original_prompt: str, evaluation_summary: str) -> str:
#     """
#     Refines the segmentation prompt based on the evaluation summary.
#     """
#     return f"""The original segmentation prompt was:
# "{original_prompt}"

# Please output only a refined segmentation prompt (better for segmentation tool) based on Evaluation Summary {evaluation_summary} in **3 sentences or less**, clearly and concisely describing how to segment the target structure.

# ### Refined Segmentation Prompt ###
# """

def refine_segmentation_prompt(original_prompt: str, evaluation_summary: str) -> str:
    """
    Generates a simplified and clearer version of the original segmentation prompt,
    using feedback from the evaluation summary to improve clarity and usability.
    """
    return f"""You are revising a segmentation instruction.

Original Segmentation Prompt:
"{original_prompt}"

Evaluation Feedback:
"{evaluation_summary}"

Instructions:
- Rewrite the prompt to be clearer and easier to follow.
- Avoid technical jargon or complex phrasing.
- Limit your revision to 3 concise sentences.
- Emphasize how to correctly identify and segment the target structure.
- Fix any issues or confusion mentioned in the feedback.

### Refined Segmentation Prompt ###
"""


# def refine_segmentation_prompt(original_prompt: str, evaluation_summary: str) -> str:
#     """
#     Refines the segmentation prompt using insights from the evaluation summary.
#     """
#     return f"""You are tasked with improving a segmentation prompt for better performance by the segmentation tool.

# Original Prompt:
# "{original_prompt}"

# Evaluation Summary:
# {evaluation_summary}

# Please revise the original prompt based on the evaluation summary. Your refined prompt should address any identified issues or weaknesses, and improve clarity, precision, or structure. 

# ### Output ###
# Return only the **refined segmentation prompt**, written in **3 sentences or fewer**.
# """


def segmentation_evaluator(image_path: str, segmentation_prompt: str, evaluation_prompt_path: str, image_example_1_path: str, image_example_2_path: str) -> dict:
    """
    Evaluates a segmented image and provides a refined segmentation prompt.
    """
    try:
        # Load image
        image = Image.open(Path(image_path))
        
        # Load image
        example_image_1 = Image.open(Path(image_example_1_path))
        example_image_2 = Image.open(Path(image_example_2_path))


        # Initialize Gemini
        model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        
        # Step 1: Generate or load evaluation prompt
        eval_path = Path(evaluation_prompt_path)

        with open(eval_path, "r") as f:
            evaluation_prompt = json.load(f)["evaluation_prompt"]

            
        #print(evaluation_prompt)


        # Step 3: Evaluate segmentation
        evaluation_response = model.generate_content(
            [evaluation_prompt, example_image_1, example_image_2, image],
             generation_config={"temperature": 0.2,})
        

        evaluation_text = evaluation_response.text.strip()

        # --- FIX: remove markdown fences like ```json ... ```
        if evaluation_text.startswith("```"):
            evaluation_text = evaluation_text.strip("`")
            # drop first line if it says "json"
            lines = evaluation_text.splitlines()
            if lines and lines[0].strip().lower().startswith("json"):
                lines = lines[1:]
            evaluation_text = "\n".join(lines)

        # Attempt to extract JSON from evaluation
        try:
            start_idx = evaluation_text.find("{")
            end_idx = evaluation_text.rfind("}") + 1
            evaluation_json = json.loads(evaluation_text[start_idx:end_idx])
            #evaluation_json = evaluation_text[start_idx:end_idx]
        except Exception as json_error:
            logger.warning("Failed to parse evaluation JSON, using raw text fallback.")
            evaluation_json = evaluation_text

        print(evaluation_json)
        # Step 3: Generate refined prompt
        if isinstance(evaluation_json, dict) and "SummaryOfReasons" in evaluation_json:
            summary = evaluation_json["SummaryOfReasons"]
        elif isinstance(evaluation_json, str):
            # if fallback -> just use the raw evaluation text directly
            summary = evaluation_json
        else:
            summary = "Evaluation data is incomplete."


        #generate a refine prompt based on summary
        refinement_instruction = refine_segmentation_prompt(segmentation_prompt, summary)
        refined_response = model.generate_content(refinement_instruction)
        refined_prompt = refined_response.text.strip()

        return json.dumps({
            "evaluation": evaluation_json,
            "refined_segmentation_prompt": refined_prompt,
        }, indent=2)


    except Exception as e:
        logger.error(f"Error in segmentation evaluation: {e}")
        return {"error": str(e)}
