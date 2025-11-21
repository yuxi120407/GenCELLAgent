import streamlit as st
import os
import time
import json
import asyncio
import datetime
from vertexai.generative_models import GenerativeModel, Part
from src.tools.serp import search as google_search
from src.tools.serp_new import search as google_search_summary
from src.tools.wiki import search as wiki_search
from src.tools.segment import segment_image 
#from src.tools.segmentation_eval_google import segmentation_evaluator
from src.tools.segmentation_eval_google_new_2_5_ICT import segmentation_evaluator
from src.tools.oneshot_segGPT import seggpt_inference_img
from src.tools.ensemble import ensemble_masks, ensemble_masks_only_text
from src.tools.mitonet import mitonet_inference
from src.tools.summarizer import summarizer_report
from src.tools.launch_human_correction import launch_sam_correction_tool

from src.utils.history_check import txt_from_each_subdir_sorted, history_lookup, extract_hitl_summaries


from src.utils.io import write_to_file, read_file
from src.config.logging import logger
from src.config.setup import config
from src.llm.gemini import generate
from pydantic import BaseModel, Field
from typing import Callable, Union, List, Dict, Any
from enum import Enum, auto
import google.generativeai as genai
import numpy as np
import cv2
import vertexai
import aiofiles
from PIL import Image
import base64
import json
from typing import Optional
import subprocess
import ast


def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")

# Initialize Vertex AI with your project and location
vertexai.init(
    project="gen-lang-client-0122908592",  # Replace with your Google Cloud project ID
    location="us-central1"                 # Replace with your GCP region (e.g., "us-central1")
)

Observation = Union[str, Exception]
PROMPT_TEMPLATE_PATH = "./data/input/react_new_v9_auto_golgi_25.txt"
OUTPUT_TRACE_PATH = "./data/output/trace_8_18.txt"
PLANNING_PROMPT_TEMPLATE_PATH = "./data/input/planning_v8_v5_golgi_auto.txt"

class AgentState(str, Enum):
    """Agent execution states"""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"
    

class Name(Enum):
    WIKIPEDIA = auto()
    GOOGLE = auto()
    IMAGESEGMENTATION = auto()
    SEGMENTATIONEVALUATION = auto()
    ONESHOTSEGMENTATION = auto()
    #ENSEMBLEREFERENCE = auto()
    #MITONET = auto()
    #ENSEMBLETEXT = auto()
    SUMMARIZER = auto()
    #HUMANCORRECTION =auto()
    NONE = auto()

    def __str__(self) -> str:
        return self.name.lower()

class Choice(BaseModel):
    name: Name = Field(..., description="The name of the tool chosen.")
    reason: str = Field(..., description="The reason for choosing this tool.")

class Message(BaseModel):
    role: str = Field(..., description="The role of the message sender.")
    content: str = Field(..., description="The content of the message.")

class Tool:
    def __init__(self, name: Name, func: Callable):
        self.name = name
        self.func = func

    async def use(self, query: Union[str, Dict[str, str], None] = None) -> Observation:
        try:
            logger.info(f"Using tool: {self.name} with query: {query}")
            if query is None or query == "" or (isinstance(query, dict) and not query):
                result = self.func()
            elif isinstance(query, dict):
                required_args = self.func.__code__.co_varnames[:self.func.__code__.co_argcount]
                missing_args = [arg for arg in required_args if arg not in query]
                if missing_args:
                    raise ValueError(f"Missing required arguments for tool {self.name}: {missing_args}")
                result = self.func(**query)
            elif isinstance(query, str):
                result = self.func(query)
            else:
                raise ValueError(f"Invalid input type for tool {self.name}: {type(query)}")
            logger.info(f"Tool {self.name} executed successfully with result: {result}")
            return result
        except Exception as e:
            error_msg = f"Error executing tool {self.name}: {e}"
            logger.error(error_msg)
            return str(e)

class Agent:
    def __init__(self, model: GenerativeModel, callback=None) -> None:
        self.model = model
        self.tools: Dict[Name, Tool] = {}
        self.messages: List[Message] = []
        self.query = ""
        self.max_iterations = 10
        self.current_iteration = 0
        self.template = self.load_template()
        self.planning_template = self.load_plan_template()
        self.callback = callback
        self.previous_tool_name: Optional[Name] = None
        self.iter_mask_path = None
        self.summary_done = False


    def load_template(self) -> str:
        return read_file(PROMPT_TEMPLATE_PATH)
    
    def load_plan_template(self) -> str:
        return read_file(PLANNING_PROMPT_TEMPLATE_PATH)

    def register(self, name: Name, func: Callable[[Any], Any], input_type: str = "text", output_type: str = "text") -> None:
        if name in self.tools:
            raise ValueError(f"Tool with name '{name}' is already registered.")
        self.tools[name] = Tool(name, func)

    def get_chat_history(self):
        return "\n".join([f"{m.role}: {m.content}" for m in self.messages])

    def clear_history(self):
        self.messages = []
        st.session_state.trace_messages = []
        if self.callback:
            self.callback({"type": "system", "content": "Chat history cleared"})
            
    def generate_summarizer_prompt(self, previous_hitl_results=""):
        query = self.query
        history = self.get_chat_history()

        prompt = f"""
You are an intelligent assistant summarizing a segmentation workflow to understand user behavior and recommend the appropriate HITL mode for the next run. Your output must be based on **both the current session and historical HITL recommendations**, with a strong focus on user behavior progression.

## Current Run
- Query: {query}
- Interaction History:
{history}

## Historical HITL Mode Recommendations:
The following recommendations were made in previous runs:
{previous_hitl_results}

---

## PART 1: Current Run Analysis
1. Describe the segmentation process used in the current run.
2. List the tools used in order and what each contributed.
3. Summarize the user’s interaction behavior:
   - Use of automatic vs manual tools
   - Use of references (e.g. one-shot segmentation)
   - Feedback frequency
   - Number of iterations

4. Based on this run alone, recommend:
[CURRENT RUN]
Recommended HITL Mode: <Fully Automatic | Reference Guided | Human Interaction>
Reason: <why this HITL mode fits this specific run>

---

## PART 2: Long-Term User Profile and Final Recommendation
5. Review the historical HITL recommendations and detect **behavioral trends**:
- Is the user becoming more or less interactive over time?
- Are they consistently using the same tools or exploring new ones?
- Are they gradually shifting from automation to correction (or vice versa)?

6. Generate a long-term **User Profile** considering both the current and past sessions.
Example profiles:
- "Consistently prefers fully automated workflows with minimal feedback."
- "Has evolved from reference-based guidance to more manual correction."
- "Initially used correction tools but now prefers faster automatic approaches."

7. Provide the final recommendation:

[OVERALL RECOMMENDATION]
Recommended HITL Mode: <Fully Automatic | Reference Guided | Human Interaction>
User Profile: <summary across runs that includes progression or consistency>
Reason: <why this mode is appropriate based on the pattern across sessions>
--

## Guidance:
- If the tool `oneshotsegmentation` was used in the current run, the user clearly provided a reference image and mask.  
  → In that case, recommend `Reference Guided` for [CURRENT RUN].
- Use other tool choices (like `humancorrection`, `mitonet`, `imagesegmentation`, etc.) and the historical HITL trends to assess consistency or change.
- Do not recommend a different HITL mode unless there is clear evidence the user's behavior has shifted.
- In the `[OVERALL RECOMMENDATION]`, the user profile must reflect their **behavior evolution** over time.
- Include progression insight in the User Profile: e.g., "The user increasingly engages with manual tools."


The final output must include:
- A `[CURRENT RUN]` block  
- A `[OVERALL RECOMMENDATION]` block  
- A thoughtful User Profile that reflects both behavior **and change over time**
"""

        return prompt



    def trace(self, role: str, content: str) -> None:
        """
        Logs the message, writes to file, appends it to a persistent list, and updates the UI container.
        """
        if role != "system":
            self.messages.append(Message(role=role, content=content))
            if self.callback:
                self.callback({"type": "message", "role": role, "content": content})
        write_to_file(path=OUTPUT_TRACE_PATH, content=f"{role}: {content}\n")
        if "trace_messages" not in st.session_state:
            st.session_state.trace_messages = []
        st.session_state.trace_messages.append((self.current_iteration, role, content))
        if "trace_container" not in st.session_state:
            st.session_state.trace_container = st.empty()
        with st.session_state.trace_container.container():
            for iter_num, r, c in st.session_state.trace_messages:
                if r == "assistant":
                    try:
                        cleaned = c.replace("Thought:", "").strip()
                        # Remove triple backticks or language tags like ```json
                        if cleaned.startswith("```json"):
                            cleaned = cleaned[7:].strip()
                        if cleaned.endswith("```"):
                            cleaned = cleaned[:-3].strip()
                  
                        parsed_content = json.loads(cleaned)
                        if "thought" in parsed_content:
                            st.markdown(f"### [Iteration {iter_num}] 🤖 **Agent Thought**")
                            st.info(f"**Thought:** {parsed_content['thought']}")
                        if "planning" in parsed_content:
                            st.markdown(f"### [Iteration {iter_num}] 🤖 **Agent Planning**")
                            st.markdown("#### **Planning Steps:**")
               
                            planning_response = parsed_content["planning"]
                            cleaned_planning_response = planning_response.strip().strip('`').strip()
                            if cleaned_planning_response.startswith('json'):
                                cleaned_planning_response = cleaned_planning_response[4:].strip()
                            parsed_planning_response = json.loads(cleaned_planning_response)
                            st.json(parsed_planning_response)
                            
                        if "action" in parsed_content:
                            st.markdown(f"### [Iteration {iter_num}] 🛠️ **Agent Action**")
                            st.markdown("#### **Action Details**")
                            st.markdown(f"- **Name:** {parsed_content['action'].get('name', 'N/A')}")
                            st.markdown(f"- **Reason:** {parsed_content['action'].get('reason', 'N/A')}")
                            if "input" in parsed_content["action"]:
                                st.markdown("##### **Input Parameters**")
                                st.json(parsed_content["action"]["input"])
                    except json.JSONDecodeError:
                        st.info(c)
                elif r == "system":
                    if ":" in c:
                        prefix, rest = c.split(":", 1)
                        rest = rest.strip()
                        try:
                            json_obj = json.loads(rest)
                            st.success(f"🔍 **{prefix.strip()}:**")
                            st.json(json_obj)
                        except Exception:
                            st.success(f"🔍 **Observation:** {c}")
                    else:
                        st.success(f"🔍 **Observation:** {c}")
                            
                    # Handle "save_path:" 
                    if "human_correction_mask_path" in c:
                        try:
                            dict_str = c.split("Observation from humancorrection:", 1)[1].strip()
                            obs = ast.literal_eval(dict_str)
                            final_mask_path = obs["human_correction_mask_path"]

                            # Parse file names
                            folder = st.session_state.timestamp_folder
                            iter_image_path = os.path.join(folder, f"human_correction_iter{iter_num}.png")
                            iter_mask_path = os.path.join(folder, f"human_correction_iter{iter_num}_mask.png")


                            # Load image and mask
                            image = cv2.imread(st.session_state.temp_path)
                            mask = np.load(final_mask_path)
                            
                            if mask.max() <= 1.0:
                                mask = (mask * 255).astype(np.uint8)
                            else:
                                mask = mask.astype(np.uint8)

                            # Save mask
                            cv2.imwrite(iter_mask_path, mask)

                            # Create overlay
                            if len(image.shape) == 2 or image.shape[2] == 1:
                                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                            red_mask = np.zeros_like(image)
                            red_mask[:, :, 2] = mask
                            overlay = cv2.addWeighted(image, 0.5, red_mask, 0.5, 0)

                            # Save overlay image
                            cv2.imwrite(iter_image_path, overlay)
                            
                            # Display in Streamlit
                            image_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                            st.image(image_rgb, caption=f"Human Corrected Overlay (Iter {iter_num})", use_column_width=True)

                            print("✅ Mask and overlay saved and displayed:")
                            print(" - Mask:", iter_mask_path)
                            print(" - Overlay:", iter_image_path)

                        except Exception as e:
                            print("❌ Failed to process:", e)
                    else:
                        print("⚠️ 'human_correction_mask_path' not found in the string.")
                        
                    if "segment_save_path:" in c:
                        image_path = c.split("segment_save_path:", 1)[1].split(",")[0].strip()
                        base, ext = os.path.splitext(os.path.basename(image_path))

                        mask_path = c.split("segment_mask_path:", 1)[1].split(",")[0].strip()
                        base_mask, ext = os.path.splitext(os.path.basename(mask_path))

                        folder = st.session_state.timestamp_folder
                        iter_image_path = os.path.join(folder, f"{base}_iter{iter_num}{ext}")
                        self.iter_mask_path = os.path.join(folder, f"{base}_iter{iter_num}_mask{ext}")
                        if os.path.exists(iter_image_path):
                            image = cv2.imread(iter_image_path)
                            image_mask = cv2.imread(self.iter_mask_path)
                        else:
                            image = cv2.imread(image_path)
                            image_mask = cv2.imread(mask_path)

                        if image is not None and image_mask is not None:
                            cv2.imwrite(iter_image_path, image)
                            cv2.imwrite(self.iter_mask_path, image_mask)
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            st.image(image_rgb, caption=f"Processed Image (Iteration {iter_num})", use_column_width=True)
                        else:
                            st.error(f"Could not load image from {image_path}")

                    # Handle "seggpt_save_path:"
                    if "seggpt_output_path:" in c:
                        image_path = c.split("seggpt_output_path:", 1)[1].split(",")[0].strip()
                        base, ext = os.path.splitext(os.path.basename(image_path))

                        mask_path = c.split("seggpt_mask_path:", 1)[1].split(",")[0].strip()
                        base_mask, ext = os.path.splitext(os.path.basename(mask_path))

                        folder = st.session_state.timestamp_folder
                        iter_image_path = os.path.join(folder, f"{base}_iter{iter_num}{ext}")
                        self.iter_mask_path = os.path.join(folder, f"{base}_iter{iter_num}_mask{ext}")
                        if os.path.exists(iter_image_path):
                            image = cv2.imread(iter_image_path)
                            image_mask = cv2.imread(self.iter_mask_path)
                        else:
                            image = cv2.imread(image_path)
                            image_mask = cv2.imread(mask_path)

                        if image is not None and image_mask is not None:
                            cv2.imwrite(iter_image_path, image)
                            cv2.imwrite(self.iter_mask_path, image_mask)
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            st.image(image_rgb, caption=f"Processed Image with one-shot segmentation (Iteration {iter_num})", use_column_width=True)
                        else:
                            st.error(f"Could not load image from {image_path}")
                    
                    if "MitoNet_image_path:" in c:
                        image_path = c.split("MitoNet_image_path:", 1)[1].split(",")[0].strip()
                        base, ext = os.path.splitext(os.path.basename(image_path))

                        mask_path = c.split("MitoNet_mask_path:", 1)[1].split(",")[0].strip()
                        base_mask, ext = os.path.splitext(os.path.basename(mask_path))

                        folder = st.session_state.timestamp_folder
                        iter_image_path = os.path.join(folder, f"{base}_iter{iter_num}{ext}")
                        self.iter_mask_path = os.path.join(folder, f"{base}_iter{iter_num}_mask{ext}")
                        if os.path.exists(iter_image_path):
                            image = cv2.imread(iter_image_path)
                            image_mask = cv2.imread(self.iter_mask_path)
                        else:
                            image = cv2.imread(image_path)
                            image_mask = cv2.imread(mask_path)

                        if image is not None and image_mask is not None:
                            cv2.imwrite(iter_image_path, image)
                            cv2.imwrite(self.iter_mask_path, image_mask)
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            st.image(image_rgb, caption=f"Processed Image with MitoNet segmentation (Iteration {iter_num})", use_column_width=True)
                        else:
                            st.error(f"Could not load image from {image_path}")
                            
                    if "ensemble_results_save_path:" in c:
                        image_path = c.split("ensemble_results_save_path:", 1)[1].split(",")[0].strip()
                        base, ext = os.path.splitext(os.path.basename(image_path))
                        
                        mask_path = c.split("ensemble_results_mask_save_path:", 1)[1].split(",")[0].strip()
                        
                        folder = st.session_state.timestamp_folder
                        iter_image_path = os.path.join(folder, f"{base}_iter{iter_num}{ext}")
                        self.iter_mask_path = os.path.join(folder, f"{base}_mask_iter{iter_num}{ext}")
                        if os.path.exists(iter_image_path):
                            image = cv2.imread(iter_image_path)
                            image_mask = cv2.imread(self.iter_mask_path)
                        else:
                            image = cv2.imread(image_path)
                            image_mask = cv2.imread(mask_path)
                        if image is not None and image_mask is not None:
                            cv2.imwrite(iter_image_path, image)
                            cv2.imwrite(self.iter_mask_path, image_mask)
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            st.image(image_rgb, caption=f"Processed Image with ensemble based on the prompt image (Iteration {iter_num})", use_column_width=True)
                        else:
                            st.error(f"Could not load image from {image_path}")
                    
                    if "ensemble_results_text_save_path:" in c:
                        image_path = c.split("ensemble_results_text_save_path:", 1)[1].split(",")[0].strip()
                        base, ext = os.path.splitext(os.path.basename(image_path))
                        
                        mask_path = c.split("ensemble_results_text_mask_save_path:", 1)[1].split(",")[0].strip()
                        
                        folder = st.session_state.timestamp_folder
                        iter_image_path = os.path.join(folder, f"{base}_iter{iter_num}{ext}")
                        self.iter_mask_path = os.path.join(folder, f"{base}_mask_iter{iter_num}{ext}")
                        
                        if os.path.exists(iter_image_path):
                            image = cv2.imread(iter_image_path)
                            image_mask = cv2.imread(self.iter_mask_path)
                        else:
                            image = cv2.imread(image_path)
                            image_mask = cv2.imread(mask_path)
                        if image is not None:
                            cv2.imwrite(iter_image_path, image)
                            cv2.imwrite(self.iter_mask_path, image_mask)
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            st.image(image_rgb, caption=f"Processed Image with ensemble based on the object text description (Iteration {iter_num})", use_column_width=True)
                        else:
                            st.error(f"Could not load image from {image_path}")

                elif r == "error":
                    st.error(f"[Iteration {iter_num}] ❌ **Error:** {c}")


    async def wait_for_human_approval(self):
        st.markdown("### Awaiting human approval. Check the **Continue** checkbox to proceed.")
        continue_checked = st.checkbox("Continue", key="wait_for_continue")
        if continue_checked:
            st.session_state.human_approval = True
        while not st.session_state.get("human_approval", False):
            await asyncio.sleep(0.1)
        st.session_state.human_approval = False    
    

    
    @staticmethod
    async def async_save_file(uploaded_file, dest_path):
        """Asynchronously save the uploaded file using aiofiles."""
        async with aiofiles.open(dest_path, "wb") as f:
            # getbuffer() is synchronous; for large files, consider processing in chunks.
            await f.write(uploaded_file.getbuffer())

    async def think(self) -> None:
        self.current_iteration += 1
        logger.info(f"Starting iteration {self.current_iteration}")
        st.markdown(f"### Iteration {self.current_iteration}")

        if self.callback:
            self.callback({"type": "iteration", "number": self.current_iteration})

        write_to_file(path=OUTPUT_TRACE_PATH, content=f"\n{'='*50}\nIteration {self.current_iteration}\n{'='*50}\n")



        if self.current_iteration == 1:
            # First check the history and use the most recent report  
            files = txt_from_each_subdir_sorted("/home/idies/workspace/Storage/xyu1/persistent/Langchain/output_images")
            if files:
                latest_file = files[-1] 
                result = history_lookup(latest_file)

                if result['found']:
                    self.saved_information = (
                        f"Object Name: {result['object_name']}\n"
                        f"Prompt Image Path: {result['reference_image_path']}\n"
                        f"Prompt Mask Path: {result['reference_mask_path']}\n"
                        f"Visual Characteristics: {result['visual_characteristics']}\n"
                    )
                else:
                    self.saved_information = "None"
            else:
                self.saved_information = "None"

            # Generate the planning prompt
            planning_prompt = self.planning_template.format(
                query=self.query,
                saved_information=self.saved_information,
                tools=', '.join([str(tool.name) for tool in self.tools.values()])
            )

            self.planning_response = await self.ask_gemini_async(planning_prompt)
            logger.info(f"Thinking => {self.planning_response}")
            self.trace("assistant", json.dumps({"planning": self.planning_response}))

            
        elif (self.current_iteration>=10 or self.previous_tool_name == Name.SUMMARIZER) and not self.summary_done: #self.current_iteration >= self.max_iterations or 
            

            logger.warning("Reached maximum iterations or summarization completed. Stopping.")
            #load the most previous summary txt
            output_base_dir = "/home/idies/workspace/Storage/xyu1/persistent/Langchain/output_images"
            previous_summaries_text = extract_hitl_summaries(output_base_dir)
            print("----- Loaded Last Summary -----")
            print(previous_summaries_text)
            summary_prompt = self.generate_summarizer_prompt(previous_summaries_text)
            final_response = await self.ask_gemini_async(summary_prompt)
            self.trace("assistant", f"Within the allowed number of iterations. Here's what I know so far: {final_response}")

            # Save summary
            try:
                summary_path = os.path.join(st.session_state.timestamp_folder, "summary.txt")
                with open(summary_path, "w") as f:
                    f.write(final_response)
                logger.info(f"Summary saved to {summary_path}")
            except Exception as e:
                logger.warning(f"Could not save summary: {e}")

            # Parse HITL recommendation
            try:
                for line in final_response.splitlines():
                    if "recommended hitl mode:" in line.lower():
                        mode = line.split(":")[1].strip()
                        st.session_state["recommended_hitl_mode"] = mode
                        hitl_path = os.path.join(st.session_state.timestamp_folder, "next_hitl_level.json")
                        with open(hitl_path, "w") as f:
                            json.dump({"hitl_mode": mode}, f)
                        break
            except Exception as e:
                logger.warning(f"Failed to extract HITL mode from summary: {e}")

            self.summary_done = True  # Prevent future summaries
            self.previous_tool_name = None
            return    
            
            
        else:
            # Use saved planning response and saved information for next prompts
            prompt = self.template.format(
                query=self.query,
                planning=self.planning_response,
                historical_information=self.saved_information,
                history=self.get_chat_history(),
                tools=', '.join([str(tool.name) for tool in self.tools.values()])
            )

            response = await self.ask_gemini_async(prompt)  #first await for the response 
            logger.info(f"Thinking => {response}")
            self.trace("assistant", f"Thought: {response}")
            await self.decide(response)

    async def decide(self, response: str) -> None:
        try:
            cleaned_response = response.strip().strip('`').strip()
            if cleaned_response.startswith('json'):
                cleaned_response = cleaned_response[4:].strip()
            parsed_response = json.loads(cleaned_response)

            if "action" in parsed_response:
                action = parsed_response["action"]
                tool_name = Name[action["name"].upper()]
                if tool_name == Name.NONE:
                    logger.info("No action needed. Proceeding to final answer.")
                    await self.think()
                else:
                    self.trace("assistant", f"Action: Using {tool_name} tool")
                    await self.act(tool_name, action.get("input", self.query))

            elif "answer" in parsed_response:
                self.trace("assistant", f"Final Answer: {parsed_response['answer']}")
                self.previous_tool_name = Name.SUMMARIZER
                await self.think()
            else:
                raise ValueError("Invalid response format")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response: {response}. Error: {str(e)}")
            self.trace("assistant", "I encountered an error in processing. Let me try again.")
            await self.think()
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            self.trace("assistant", "I encountered an unexpected error. Let me try a different approach.")
            await self.think()

    async def act(self, tool_name: Name, query: str) -> None:
        tool = self.tools.get(tool_name)
        if not tool:
            logger.error(f"No tool registered for choice: {tool_name}")
            self.trace("system", f"Error: Tool {tool_name} not found")
            await self.think()
            return
        
        else:
            # For other tools, simply use the query.
            result = await tool.use(query)  ## add await
            observation = f"Observation from {tool_name}: {result}"
            if tool_name == Name.SUMMARIZER:
                self.previous_tool_name = Name.SUMMARIZER

        # Log the observation and append it to the messages list.
        self.trace("system", observation)
        self.messages.append(Message(role="system", content=observation))
        

        
        # After logging observation
        if tool_name == Name.SUMMARIZER:
            # mark that summarizer just ran; usually no human approval gate
            self.previous_tool_name = Name.SUMMARIZER
        elif tool_name != Name.ONESHOTSEGMENTATION:
            # all other tools (except one-shot) require approval
            await self.wait_for_human_approval()
            
        #Update previous_tool_name for next time
        #self.previous_tool_name = tool_name  
        await self.think()

    async def ask_gemini_async(self, prompt: str) -> str:
        return await asyncio.to_thread(self.ask_gemini, prompt)

    def ask_gemini(self, prompt: str) -> str:
        contents = [Part.from_text(prompt)]
        response = generate(self.model, contents)
        return str(response) if response is not None else "No response from Gemini"

    async def execute(self, query: str) -> str:
        self.query = query
        self.trace(role="user", content=query)
        await self.think()
        return self.messages[-1].content



def build_query(image_path, user_query, save_dir, tools_list, reference_image_path=None, reference_mask_path=None):
    

    # Build the payload
    if reference_image_path is not None and reference_mask_path is not None: 
        final_query = {
            "image_path": image_path,
            "prompt_image_path": reference_image_path,
            "prompt_mask_path": reference_mask_path,
            "save_directory": save_dir,
            "query": f"{user_query}",
            "tool_list":tools_list,
        }   
    else:
        final_query = {
            "image_path": image_path,
            "save_directory": save_dir,
            "query": f"{user_query}",
            "tool_list":tools_list,
        }

    return json.dumps(final_query, indent=4)


# ----------------- Streamlit UI -----------------

st.title("🤖 Cell Image Segmentation Agent")

st.sidebar.title("About LLM Agent")
st.sidebar.info(
    "This LLM Agent uses the Gemini model to reason about queries and take actions using available tools. "
    "It can segment objects in images by providing a textual query."
)
st.sidebar.markdown("### Available Tools")


planning_icon_path = "/home/idies/workspace/Storage/xyu1/persistent/Langchain/ours_test/planning_icon.png"
planning_icon_img_base64 = get_img_as_base64(planning_icon_path)
planning_markdown_string = f"""
- <img src="data:image/png;base64,{planning_icon_img_base64}" alt="" width="20" height="20" /> Planning
"""
st.sidebar.markdown(planning_markdown_string, unsafe_allow_html=True)


st.sidebar.markdown("- 🌐 Web Search")

segmentation_icon_path = "/home/idies/workspace/Storage/xyu1/persistent/Langchain/ours_test/segmentation_icon.png"
segmentaion_icon_img_base64 = get_img_as_base64(segmentation_icon_path)
segmentation_markdown_string = f"""
- <img src="data:image/png;base64,{segmentaion_icon_img_base64}" alt="" width="20" height="20" /> General Image segmentation
"""
st.sidebar.markdown(segmentation_markdown_string, unsafe_allow_html=True)

st.sidebar.markdown("- 🔍 Image Segmentation Evaluation")
st.sidebar.markdown("- 🎯 Few-shot Image Segmentation")


mitochondria_icon_path = "/home/idies/workspace/Storage/xyu1/persistent/Langchain/ours_test/mitochondria_icon.png"
mitochondria_icon_img_base64 = get_img_as_base64(mitochondria_icon_path)
mitochondria_markdown_string = f"""
- <img src="data:image/png;base64,{mitochondria_icon_img_base64}" alt="" width="20" height="20" /> Mitochondrion Segmentation
"""
st.sidebar.markdown(mitochondria_markdown_string, unsafe_allow_html=True)



st.sidebar.markdown("- 📝 Ensemble Masks from Text")
st.sidebar.markdown("- 🖼️ Ensemble Masks from Image")


if "timestamp_folder" not in st.session_state: 
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
    folder = os.path.join("/home/idies/workspace/Storage/xyu1/persistent/Langchain/output_images", timestamp)
    os.makedirs(folder, exist_ok=True)
    st.session_state.timestamp_folder = folder

if "temp_path" not in st.session_state:
    st.session_state.temp_path = None

if "tool_list" not in st.session_state:
    st.session_state.tool_list = None

# Ensure these keys exist in session state so we can store optional prompt images/masks
if "human_uploaded_image" not in st.session_state:
    st.session_state.human_uploaded_image = None
if "human_uploaded_mask" not in st.session_state:
    st.session_state.human_uploaded_mask = None

# --- Main image (required) ---
uploaded_file = st.file_uploader("Upload your main image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(opencv_image, channels="BGR", caption="Uploaded Main Image", use_column_width=True)
    temp_path = os.path.join("/home/idies/workspace/Storage/xyu1/persistent/LISA/imgs", uploaded_file.name)
    st.session_state.temp_path = temp_path
    st.success(f"Uploaded file path: {temp_path}")

# --- Let the user decide if they have an example image/mask ---
st.markdown("### Do you have an example image and mask for one-shot segmentation?")

have_example = st.checkbox("I have an example image/mask")
no_example = st.checkbox("I don't have any example image/mask")

# Simple logic: if user selects "I have example," show file uploaders.
# If user selects "I don't have example," do nothing.
# If both are selected, you can handle that as an error/warning or just ignore it.
if have_example and not no_example:
    st.markdown("#### Please upload your example image and mask below:")
    
    prompt_image = st.file_uploader("Prompt Image", type=["png", "jpg", "jpeg"], key="prompt_image")
    if prompt_image:
        prompt_image_path = os.path.join(st.session_state.timestamp_folder, prompt_image.name)
        with open(prompt_image_path, "wb") as f:
            f.write(prompt_image.getbuffer())
        st.session_state.human_uploaded_image = prompt_image_path
        st.image(prompt_image_path, caption="Prompt Image", use_column_width=True)

    prompt_mask = st.file_uploader("Prompt Mask", type=["png", "jpg", "jpeg"], key="prompt_mask")
    if prompt_mask:
        prompt_mask_path = os.path.join(st.session_state.timestamp_folder, prompt_mask.name)
        with open(prompt_mask_path, "wb") as f:
            f.write(prompt_mask.getbuffer())
        st.session_state.human_uploaded_mask = prompt_mask_path
        st.image(prompt_mask_path, caption="Prompt Mask", use_column_width=True)

elif no_example and not have_example:
    st.info("You have indicated you don't have any example image/mask.")
elif have_example and no_example:
    st.warning("You selected both options. Please choose only one.")

# Query
user_query = st.text_area(
    "Enter a short query:",
    placeholder="e.g., Help me segment the mitochondrion in the provided image",
    height=120  # Adjust the height as needed (in pixels)
)



if st.session_state.temp_path is None or not user_query:
    st.warning("Please upload a main image and enter a query before running the agent.")
else:
    # Create the agent if it doesn't exist in session state
    if "agent" not in st.session_state:
        gemini = GenerativeModel(config.MODEL_NAME)
        st.session_state.agent = Agent(model=gemini, callback=lambda x: st.session_state.events.append(x))
        st.session_state.agent.register(Name.GOOGLE, google_search_summary)
        st.session_state.agent.register(Name.IMAGESEGMENTATION, segment_image)
        st.session_state.agent.register(Name.SEGMENTATIONEVALUATION, segmentation_evaluator)
        st.session_state.agent.register(Name.ONESHOTSEGMENTATION, seggpt_inference_img)
        st.session_state.agent.register(Name.SUMMARIZER, summarizer_report)

        
        
        
        st.session_state.events = []
        st.session_state.trace_messages = []
        st.session_state.human_approval = False
        
        #st.session_state.tool_list = list(st.session_state.agent.tools.keys())
        st.session_state.tool_list = [tool.name.lower() for tool in st.session_state.agent.tools.keys()]
        print(st.session_state.tool_list)
    
    user_input = build_query(st.session_state.temp_path, user_query,
                             st.session_state.timestamp_folder,
                             st.session_state.tool_list,
                             st.session_state.human_uploaded_image, 
                             st.session_state.human_uploaded_mask)

    if st.button("Send", key="send_button"):
        st.session_state.events = []
        try:
            with st.spinner("Agent is thinking..."):
                response = asyncio.run(st.session_state.agent.execute(user_input))
            st.session_state.events.append({"type": "final_answer", "content": response})
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
