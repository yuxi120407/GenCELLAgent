import streamlit as st
import os
import time
import json
import asyncio
import datetime
import ast
import base64
import subprocess
import numpy as np
import cv2
import aiofiles
from PIL import Image

import vertexai
from vertexai.generative_models import GenerativeModel, Part
import google.generativeai as genai

from pydantic import BaseModel, Field
from typing import Callable, Union, List, Dict, Any, Optional
from enum import Enum, auto

# --- Internal Tools & Utils ---
from src.tools.serp import search as google_search
from src.tools.serp_new import search as google_search_summary

from src.tools.gemini_sam3_segment import gemini_sam3_segment
from src.tools.gemini_vlm_eval import gemini_vlm_eval
from src.tools.oneshot_segGPT import seggpt_inference_img
from src.tools.cell_segmentation_models import (
    cellpose_segment,
    cellsam_segment,
    micro_sam_segment,
)

from src.tools.mitonet import mitonet_inference
from src.tools.summarizer import summarizer_report
from src.tools.launch_human_correction import launch_sam_correction_tool

from src.utils.history_check import txt_from_each_subdir_sorted, history_lookup, extract_hitl_summaries
from src.utils.io import write_to_file, read_file
from src.utils.exceptions import AgentPauseSignal
from src.config.logging import logger
from src.config.paths import (
    BASE_WORKSPACE,
    OUTPUT_IMAGES_DIR,
    IMG_EXAMPLE_DIR,
    PLANNING_ICON_PATH,
    SEGMENTATION_ICON_PATH,
    MITOCHONDRIA_ICON_PATH,
    PROMPT_TEMPLATE_PATH,
    PLANNING_PROMPT_TEMPLATE_PATH,
    SUMMARIZER_PROMPT_TEMPLATE_PATH,
    KNOWLEDGE_BASE_PATH,
)
from src.config.setup import config
from src.llm.gemini import generate

# Initialize Vertex AI
vertexai.init(
    project=config.PROJECT_ID,
    location=config.REGION
)

Observation = Union[str, Exception]


class Name(Enum):
    WIKIPEDIA = auto()
    GOOGLE = auto()
    IMAGESEGMENTATION = auto()
    CELLPOSE = auto()
    CELLSAM = auto()
    MICRO_SAM = auto()
    SEGMENTATIONEVALUATION = auto()
    ONESHOTSEGMENTATION = auto()
    SUMMARIZER = auto()
    NONE = auto()

    def __str__(self) -> str:
        return self.name.lower()

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
            import inspect
            if query is None or query == "" or (isinstance(query, dict) and not query):
                result = self.func()
            elif isinstance(query, dict):
                sig = inspect.signature(self.func)
                required_args = [
                    name for name, param in sig.parameters.items()
                    if param.default == inspect.Parameter.empty and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
                ]
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


# --- Helper Methods for UI / Image Processing ---

def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def _decode_uploaded_image(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    opencv_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if opencv_image is not None:
        return opencv_image

    uploaded_file.seek(0)
    pil_image = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def _process_standard_overlay(image_path, mask_path, folder, iter_num, caption_prefix):
    """A deduplicated helper for rendering OpenCV mask overlays in Streamlit."""
    base, ext = os.path.splitext(os.path.basename(image_path))
    iter_image_path = os.path.join(folder, f"{base}_iter{iter_num}{ext}")
    iter_mask_path = os.path.join(folder, f"{base}_iter{iter_num}_mask{ext}")
    
    if os.path.exists(iter_image_path):
        image = cv2.imread(iter_image_path)
        image_mask = cv2.imread(iter_mask_path)
    else:
        image = cv2.imread(image_path)
        image_mask = cv2.imread(mask_path)

    if image is not None:
        cv2.imwrite(iter_image_path, image)
        if image_mask is not None:
            cv2.imwrite(iter_mask_path, image_mask)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption=f"{caption_prefix} (Iteration {iter_num})", use_column_width=True)
    else:
        st.error(f"Could not load image from {image_path}")


def process_and_display_tool_output(c: str, iter_num: int):
    """Parses tool observation strings, handles image modifications, and renders to Streamlit."""
    folder = st.session_state.timestamp_folder
    
    if "human_correction_mask_path" in c:
        try:
            dict_str = c.split("Observation from humancorrection:", 1)[1].strip()
            obs = ast.literal_eval(dict_str)
            final_mask_path = obs["human_correction_mask_path"]

            iter_image_path = os.path.join(folder, f"human_correction_iter{iter_num}.png")
            iter_mask_path = os.path.join(folder, f"human_correction_iter{iter_num}_mask.png")

            image = cv2.imread(st.session_state.temp_path)
            mask = np.load(final_mask_path)
            
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)

            cv2.imwrite(iter_mask_path, mask)

            if len(image.shape) == 2 or image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            red_mask = np.zeros_like(image)
            red_mask[:, :, 2] = mask
            overlay = cv2.addWeighted(image, 0.5, red_mask, 0.5, 0)

            cv2.imwrite(iter_image_path, overlay)
            image_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption=f"Human Corrected Overlay (Iter {iter_num})", use_column_width=True)

            logger.info(f"Mask and overlay saved and displayed: {iter_mask_path}")
        except Exception as e:
            logger.error(f"Failed to process human correction: {e}")

    elif "segment_save_path:" in c:
        try:
            image_path = c.split("segment_save_path:", 1)[1].split(",")[0].strip()
            mask_path = c.split("segment_mask_path:", 1)[1].split(",")[0].strip()
            _process_standard_overlay(image_path, mask_path, folder, iter_num, "Processed Image")
        except Exception as e:
            logger.error(f"Failed processing segment_save_path: {e}")

    elif "seggpt_output_path:" in c:
        try:
            image_path = c.split("seggpt_output_path:", 1)[1].split(",")[0].strip()
            mask_path = c.split("seggpt_mask_path:", 1)[1].split(",")[0].strip()
            _process_standard_overlay(image_path, mask_path, folder, iter_num, "Processed Image with one-shot segmentation")
        except Exception as e:
            logger.error(f"Failed processing seggpt_output_path: {e}")

    elif "MitoNet_image_path:" in c:
        try:
            image_path = c.split("MitoNet_image_path:", 1)[1].split(",")[0].strip()
            mask_path = c.split("MitoNet_mask_path:", 1)[1].split(",")[0].strip()
            _process_standard_overlay(image_path, mask_path, folder, iter_num, "Processed Image with MitoNet segmentation")
        except Exception as e:
            logger.error(f"Failed processing MitoNet_image_path: {e}")

    elif "ensemble_results_save_path:" in c:
        try:
            image_path = c.split("ensemble_results_save_path:", 1)[1].split(",")[0].strip()
            mask_path = c.split("ensemble_results_mask_save_path:", 1)[1].split(",")[0].strip()
            _process_standard_overlay(image_path, mask_path, folder, iter_num, "Processed Image with ensemble based on the prompt image")
        except Exception as e:
            logger.error(f"Failed processing ensemble_results_save_path: {e}")

    elif "ensemble_results_text_save_path:" in c:
        try:
            image_path = c.split("ensemble_results_text_save_path:", 1)[1].split(",")[0].strip()
            mask_path = c.split("ensemble_results_text_mask_save_path:", 1)[1].split(",")[0].strip()
            _process_standard_overlay(image_path, mask_path, folder, iter_num, "Processed Image with ensemble based on the object text description")
        except Exception as e:
            logger.error(f"Failed processing ensemble_results_text_save_path: {e}")



def _get_last_seg_paths():
    """Scan trace_messages backwards and return (image_path, mask_path) from
    the most recent segmentation tool observation, or (None, None) if not found."""
    IMAGE_KEYS = [
        "segment_save_path:",
        "seggpt_output_path:",
        "MitoNet_image_path:",
        "ensemble_results_save_path:",
        "ensemble_results_text_save_path:",
    ]
    MASK_KEYS = [
        "segment_mask_path:",
        "seggpt_mask_path:",
        "MitoNet_mask_path:",
        "ensemble_results_mask_save_path:",
        "ensemble_results_text_mask_save_path:",
    ]
    for _, r, c in reversed(st.session_state.get("trace_messages", [])):
        if r != "system":
            continue
        img, msk = None, None
        for key in IMAGE_KEYS:
            if key in c:
                try:
                    img = c.split(key, 1)[1].split(",")[0].strip()
                    break
                except Exception:
                    pass
        for key in MASK_KEYS:
            if key in c:
                try:
                    msk = c.split(key, 1)[1].split(",")[0].strip()
                    break
                except Exception:
                    pass
        if img and msk:
            return img, msk
    return None, None


def render_trace_ui():
    """Renders the step-by-step history to the Streamlit UI."""
    for iter_num, r, c in st.session_state.get("trace_messages", []):
        if r == "assistant":
            try:
                cleaned = c.replace("Thought:", "").strip()
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
                    try:
                        st.json(json.loads(cleaned_planning_response))
                    except Exception:
                        st.markdown(cleaned_planning_response)
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
                # Strip markdown code fences (e.g. ```json ... ```) before parsing
                if rest.startswith("```json"):
                    rest = rest[7:].strip()
                elif rest.startswith("```"):
                    rest = rest[3:].strip()
                if rest.endswith("```"):
                    rest = rest[:-3].strip()
                is_summarizer = "summarizer" in prefix.lower()
                try:
                    parsed = json.loads(rest)
                    if is_summarizer:
                        st.markdown(f"### 📋 **Summarizer Report**")
                        with st.expander("View Full Report", expanded=True):
                            st.json(parsed)
                    else:
                        st.success(f"🔍 **{prefix.strip()}:**")
                        st.json(parsed)
                except Exception:
                    if is_summarizer:
                        st.markdown(f"### 📋 **Summarizer Report**")
                        with st.expander("View Full Report", expanded=True):
                            st.markdown(rest)
                    else:
                        st.success(f"🔍 **{prefix.strip()}:** {rest}")
            else:
                st.success(f"🔍 **Observation:** {c}")
            process_and_display_tool_output(c, iter_num)
        elif r == "error":
            st.error(f"[Iteration {iter_num}] ❌ **Error:** {c}")

# --- Agent Implementation ---

class Agent:
    def __init__(self, model: GenerativeModel, trace_path: str, callback=None) -> None:
        self.model = model
        self.trace_path = trace_path
        self.tools: Dict[Name, Tool] = {}
        self.messages: List[Message] = []
        self.query = ""
        self.max_iterations = 10
        self.current_iteration = 0
        self.segmentation_retry_count = 0  # Only increments on segment tool use
        self.template = self.load_template()
        self.planning_template = self.load_plan_template()
        self.callback = callback
        self.previous_tool_name: Optional[Name] = None
        self.summary_done = False
        self.planning_response = ""

    def reset(self):
        """Resets the agent's internal state for a fresh run."""
        self.messages = []
        self.query = ""
        self.current_iteration = 0
        self.segmentation_retry_count = 0
        self.previous_tool_name = None
        self.summary_done = False
        self.planning_response = ""

    def load_template(self) -> str:
        return read_file(PROMPT_TEMPLATE_PATH)
    
    def load_plan_template(self) -> str:
        return read_file(PLANNING_PROMPT_TEMPLATE_PATH)

    def load_summarizer_template(self) -> str:
        return read_file(SUMMARIZER_PROMPT_TEMPLATE_PATH)

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
        return self.load_summarizer_template().format(
            query=self.query,
            history=self.get_chat_history(),
            previous_hitl_results=previous_hitl_results
        )

    def trace(self, role: str, content: str) -> None:
        """
        Logs the message, writes to file, appends it to the persistent list,
        and immediately repaints the live trace container so the user sees each
        step as soon as it happens (during the async run).
        """
        if role != "system":
            self.messages.append(Message(role=role, content=content))
            if self.callback:
                self.callback({"type": "message", "role": role, "content": content})

        write_to_file(path=self.trace_path, content=f"{role}: {content}\n")

        if "trace_messages" not in st.session_state:
            st.session_state.trace_messages = []
        st.session_state.trace_messages.append((self.current_iteration, role, content))

        # Live-update the pre-allocated placeholder so content appears while
        # asyncio.run() is still executing (same pattern as backup_safe).
        if "trace_container" in st.session_state:
            with st.session_state.trace_container.container():
                render_trace_ui()

    async def wait_for_human_approval(self):
        # Signal the UI that we are paused, then stop execution.
        # AgentPauseSignal extends BaseException so it is NOT swallowed by
        # the `except Exception:` handlers in decide() or think().
        st.session_state.agent_status = "paused"
        st.session_state.pause_iteration = self.current_iteration
        raise AgentPauseSignal("Paused for Human Approval")

    
    @staticmethod
    async def async_save_file(uploaded_file, dest_path):
        """Asynchronously save the uploaded file using aiofiles."""
        async with aiofiles.open(dest_path, "wb") as f:
            await f.write(uploaded_file.getbuffer())

    async def think(self) -> None:
        self.current_iteration += 1
        logger.info(f"Starting iteration {self.current_iteration}")

        if self.callback:
            self.callback({"type": "iteration", "number": self.current_iteration})

        write_to_file(path=self.trace_path, content=f"\n{'='*50}\nIteration {self.current_iteration}\n{'='*50}\n")

        if self.current_iteration == 1:
            files = txt_from_each_subdir_sorted(OUTPUT_IMAGES_DIR)
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
                
            # Read the Knowledge Base (Memory)
            kb_data = "No prior knowledge saved."
            if os.path.exists(KNOWLEDGE_BASE_PATH):
                try:
                    with open(KNOWLEDGE_BASE_PATH, 'r') as f:
                        kb_json = json.load(f)
                        if kb_json:
                            kb_data = json.dumps(kb_json, indent=2)
                except Exception as e:
                    logger.warning(f"Could not read knowledge base: {e}")
                    
            self.saved_information += f"\n\n--- KNOWLEDGE BASE (Memory) ---\n{kb_data}"

            planning_prompt = self.planning_template.format(
                query=self.query,
                saved_information=self.saved_information,
                tools=', '.join([str(tool.name) for tool in self.tools.values()])
            )

            self.planning_response = await self.ask_gemini_async(planning_prompt)
            logger.info(f"Thinking => {self.planning_response}")
            self.trace("assistant", json.dumps({"planning": self.planning_response}))
            
            # Bump the iteration count so that the first actual tool action is logged as Iteration 2
            self.current_iteration += 1
            write_to_file(path=self.trace_path, content=f"\n{'='*50}\nIteration {self.current_iteration}\n{'='*50}\n")

        if (self.current_iteration >= 20) and not self.summary_done:
            logger.warning(f"Absolute max iterations (20) reached. Generating emergency HITL Profile and forcing stop.")
            
            previous_summaries_text = extract_hitl_summaries(OUTPUT_IMAGES_DIR)
            print("----- Loaded Last Summary -----")
            print(previous_summaries_text)
            
            summary_prompt = self.generate_summarizer_prompt(previous_summaries_text)
            final_response = await self.ask_gemini_async(summary_prompt)
            self.trace("assistant", f"Emergency HITL Summary: {final_response}")

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

            self.summary_done = True 
            self.previous_tool_name = None
            return    

        # Main ReAct Loop: This executes for EVERY iteration, including Iteration 1 after the plan is made.
        prompt = self.template.format(
            query=self.query,
            planning=self.planning_response,
            historical_information=self.saved_information,
            history=self.get_chat_history(),
            tools=', '.join([str(tool.name) for tool in self.tools.values()]),
            max_retries=st.session_state.get('user_max_retries', 5),
            retry_count=self.segmentation_retry_count
        )

        response = await self.ask_gemini_async(prompt)
        logger.info(f"Thinking => {response}")
        self.trace("assistant", f"Thought: {response}")
        await self.decide(response)

    async def decide(self, response: str) -> None:
        try:
            import re
            
            # Try to extract a JSON block if the model included conversational text
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                cleaned_response = json_match.group(1)
            else:
                # Fallback to basic stripping
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
                
                # Check if we still need to run the HITL summary before shutting down completely
                if not self.summary_done:
                    logger.warning("Agent gave final answer. Generating final HITL Profile.")
                    previous_summaries_text = extract_hitl_summaries(OUTPUT_IMAGES_DIR)
                    summary_prompt = self.generate_summarizer_prompt(previous_summaries_text)
                    final_response = await self.ask_gemini_async(summary_prompt)
                    self.trace("assistant", f"HITL Summary: {final_response}")
                    
                    try:
                        summary_path = os.path.join(st.session_state.timestamp_folder, "summary.txt")
                        with open(summary_path, "w") as f:
                            f.write(final_response)
                    except Exception as e:
                        logger.warning(f"Could not save summary: {e}")
                    
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
                        pass
                        
                    self.summary_done = True
                
                # We return here to END the loop, we DO NOT call self.think() again!
                return
            else:
                raise ValueError("Invalid response format")

        except AgentPauseSignal:
            raise  # Always let the pause signal propagate to the UI layer
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
            if tool_name in {Name.IMAGESEGMENTATION, Name.CELLPOSE, Name.CELLSAM, Name.MICRO_SAM}:
                self.segmentation_retry_count += 1
                
            # If it's the summarizer tool, automatically inject the full history 
            # so the LLM doesn't have to output it inside a giant JSON block.
            if tool_name == Name.SUMMARIZER:
                if isinstance(query, dict):
                    query["process_text"] = self.get_chat_history()
            
            result = await tool.use(query)
            observation = f"Observation from {tool_name}: {result}"

        self.trace("system", observation)
        self.messages.append(Message(role="system", content=observation))

        # Record the last used tool so the stopping logic knows what just happened
        self.previous_tool_name = tool_name

        # Always pause after every tool execution so the user can review the result
        await self.wait_for_human_approval()
        await self.think()

    async def ask_gemini_async(self, prompt: str) -> str:
        return await asyncio.to_thread(self.ask_gemini, prompt)

    def ask_gemini(self, prompt: str) -> str:
        contents = [Part.from_text(prompt)]
        response = generate(self.model, contents)
        return str(response) if response is not None else "No response from Gemini"

    async def execute(self, query: str) -> str:
        self.query = query
        if self.current_iteration == 0:
            # First call: trace the initial user query
            self.trace(role="user", content=query)
        else:
            # Resume after pause: inject any human feedback into the history
            feedback = st.session_state.pop("human_feedback", None)
            if feedback:
                self.trace(role="user", content=f"[Human Feedback] {feedback}")
        await self.think()
        return self.messages[-1].content if self.messages else ""


def build_query(image_path, user_query, save_dir, tools_list, reference_image_path=None, reference_mask_path=None):
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


# ----------------- Streamlit State Initialization -----------------

if "initialized" not in st.session_state:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
    folder = os.path.join(OUTPUT_IMAGES_DIR, timestamp)
    os.makedirs(folder, exist_ok=True)
    
    st.session_state.timestamp_folder = folder
    st.session_state.temp_path = None
    st.session_state.tool_list = None
    st.session_state.human_uploaded_image = None
    st.session_state.human_uploaded_mask = None
    st.session_state.trace_messages = []
    st.session_state.events = []
    st.session_state.human_approval = False
    st.session_state.correction_launched = False
    st.session_state.initialized = True

# ----------------- Streamlit UI -----------------

st.title("🤖 GenCellAgent: Generalizable, Training-Free Cellular Image Segmentation ")

st.sidebar.title("About LLM Agent")
st.sidebar.info(
    "This LLM Agent uses the Gemini model to reason about queries and take actions using available tools. "
    "It can segment objects in images by providing a textual query."
)
st.sidebar.markdown("### Available Tools")

try:
    planning_icon_img_base64 = get_img_as_base64(PLANNING_ICON_PATH)
    planning_markdown_string = f'- <img src="data:image/png;base64,{planning_icon_img_base64}" alt="" width="20" height="20" /> Planning'
    st.sidebar.markdown(planning_markdown_string, unsafe_allow_html=True)
except FileNotFoundError:
    st.sidebar.markdown("- 📋 Planning")

st.sidebar.markdown("- 🌐 Web Search")

try:
    segmentaion_icon_img_base64 = get_img_as_base64(SEGMENTATION_ICON_PATH)
    segmentation_markdown_string = f'- <img src="data:image/png;base64,{segmentaion_icon_img_base64}" alt="" width="20" height="20" /> General Image segmentation'
    st.sidebar.markdown(segmentation_markdown_string, unsafe_allow_html=True)
except FileNotFoundError:
    st.sidebar.markdown("- 🖼️ General Image segmentation")

st.sidebar.markdown("- 🔍 Image Segmentation Evaluation")
st.sidebar.markdown("- 🎯 One-shot Image Segmentation")

try:
    mitochondria_icon_img_base64 = get_img_as_base64(MITOCHONDRIA_ICON_PATH)
    mitochondria_markdown_string = f'- <img src="data:image/png;base64,{mitochondria_icon_img_base64}" alt="" width="20" height="20" /> Mitochondrion Segmentation'
    st.sidebar.markdown(mitochondria_markdown_string, unsafe_allow_html=True)
except FileNotFoundError:
    st.sidebar.markdown("- 🔬 Mitochondrion Segmentation")



# --- Main image (required) ---
uploaded_file = st.file_uploader("Upload your main image", type=["png", "jpg", "jpeg", "tif", "tiff"])
if uploaded_file:
    opencv_image = _decode_uploaded_image(uploaded_file)
    st.image(opencv_image, channels="BGR", caption="Uploaded Main Image", use_column_width=True)
    
    # Save the file to disk so tools can read it later via cv2.imread
    temp_path = os.path.join(IMG_EXAMPLE_DIR, uploaded_file.name)
    os.makedirs(IMG_EXAMPLE_DIR, exist_ok=True)
    cv2.imwrite(temp_path, opencv_image)
    
    st.session_state.temp_path = temp_path
    st.success(f"Uploaded file path: {temp_path}")

# --- Select Segmentation Mode ---
st.markdown("### Select Segmentation Mode")
segmentation_mode = st.radio(
    "Choose the appropriate mode for your task:",
    [
        "🪄 Auto Organelle Segmentation (ER, Golgi, Mito) - Uses General Text Guided Segmentation",
        "🔬 Cell Segmentation - Uses specialized tools (CellSAM, Micro-SAM, Cellpose)",
        "🎯 One-Shot Segmentation - Requires Example Image & Mask"
    ]
)

if "Auto" in segmentation_mode:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Auto-Mode Settings")
    st.session_state.user_max_retries = st.sidebar.slider(
        "Max Segmentation Retries",
        min_value=1,
        max_value=10,
        value=5,
        help="How many times the agent should try to refine the segmentation prompt before giving up."
    )
else:
    # Default to 1 retry for non-auto modes to prevent infinite loops
    st.session_state.user_max_retries = 1

if "One-Shot" in segmentation_mode:
    st.markdown("#### Please upload your example image and mask below:")
    
    prompt_image = st.file_uploader("Prompt Image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="prompt_image")
    if prompt_image:
        prompt_image_path = os.path.join(st.session_state.timestamp_folder, prompt_image.name)
        with open(prompt_image_path, "wb") as f:
            f.write(prompt_image.getbuffer())
        st.session_state.human_uploaded_image = prompt_image_path
        st.image(prompt_image_path, caption="Prompt Image", use_column_width=True)

    prompt_mask = st.file_uploader("Prompt Mask", type=["png", "jpg", "jpeg", "tif", "tiff"], key="prompt_mask")
    if prompt_mask:
        prompt_mask_path = os.path.join(st.session_state.timestamp_folder, prompt_mask.name)
        with open(prompt_mask_path, "wb") as f:
            f.write(prompt_mask.getbuffer())
        st.session_state.human_uploaded_mask = prompt_mask_path
        st.image(prompt_mask_path, caption="Prompt Mask", use_column_width=True)
else:
    # Clear out any previously uploaded example files if they switch modes
    st.session_state.human_uploaded_image = None
    st.session_state.human_uploaded_mask = None
    st.info(f"You selected **{segmentation_mode.split(' - ')[0]}**. No example images are required.")

# Query
user_query = st.text_area(
    "Enter a short query:",
    placeholder="e.g., Help me segment the mitochondrion in the provided image",
    height=120
)

if st.session_state.temp_path is None or not user_query:
    st.warning("Please upload a main image and enter a query before running the agent.")
else:
    if "agent" not in st.session_state:
        gemini = GenerativeModel(config.MODEL_NAME)
        trace_path = os.path.join(st.session_state.timestamp_folder, "agent_trace.txt")
        st.session_state.agent = Agent(model=gemini, trace_path=trace_path, callback=lambda x: st.session_state.events.append(x))
        st.session_state.agent.register(Name.GOOGLE, google_search_summary)
        st.session_state.agent.register(Name.IMAGESEGMENTATION, gemini_sam3_segment)
        st.session_state.agent.register(Name.CELLPOSE, cellpose_segment)
        st.session_state.agent.register(Name.CELLSAM, cellsam_segment)
        st.session_state.agent.register(Name.MICRO_SAM, micro_sam_segment)
        st.session_state.agent.register(Name.SEGMENTATIONEVALUATION, gemini_vlm_eval)
        st.session_state.agent.register(Name.ONESHOTSEGMENTATION, seggpt_inference_img)
        st.session_state.agent.register(Name.SUMMARIZER, summarizer_report)

        st.session_state.events = []
        st.session_state.trace_messages = []
        st.session_state.human_approval = False
        
        st.session_state.tool_list = [tool.name.lower() for tool in st.session_state.agent.tools.keys()]
        print(st.session_state.tool_list)
    
    # Inject the chosen mode into the prompt so the LLM knows which tools to prioritize
    enhanced_query = f"[User Selected Mode: {segmentation_mode}]\nTask: {user_query}"
    
    user_input = build_query(st.session_state.temp_path, enhanced_query,
                             st.session_state.timestamp_folder,
                             st.session_state.tool_list,
                             st.session_state.human_uploaded_image, 
                             st.session_state.human_uploaded_mask)

    if st.button("Send", key="send_button"):
        st.session_state.agent.reset()
        st.session_state.events = []
        st.session_state.trace_messages = []
        st.session_state.agent_status = "running"
        st.session_state.user_input = user_input
        st.session_state.correction_launched = False
        st.session_state.pop("trace_container", None)
        st.rerun()

    agent_status = st.session_state.get("agent_status")

    if agent_status in ["running", "paused", "finished", "error"]:
        # --- TRACE DISPLAY AREA ---
        # Always re-create the placeholder at this fixed UI position on every rerun.
        # Storing it in session_state lets trace() update it live during asyncio.run().
        trace_placeholder = st.empty()
        st.session_state.trace_container = trace_placeholder
        with trace_placeholder.container():
            render_trace_ui()

        # --- PAUSE / CONTINUE BUTTON ---
        if agent_status == "paused":
            st.markdown("---")
            st.markdown("### Step Complete — Review Above")

            pause_key = st.session_state.get('pause_iteration', 0)

            # ---- Human Correction Tool ----
            last_image, last_mask = _get_last_seg_paths()

            if st.session_state.get("correction_launched", False):
                # Waiting for the SAM correction tool to finish
                save_path = st.session_state.get("correction_save_path", "")
                done_flag = os.path.join(save_path, "done.txt")
                st.info("SAM correction tool is running in a separate window. "
                        "Make your corrections there, then click **Correction Done** below.")
                if st.button("Correction Done", key=f"corr_done_{pause_key}"):
                    if os.path.exists(done_flag):
                        result_info_path = os.path.join(save_path, "result_info.json")
                        if os.path.exists(result_info_path):
                            with open(result_info_path) as f:
                                result_info = json.load(f)
                            final_mask = os.path.join(save_path, result_info["filename"])
                            st.session_state.human_feedback = (
                                f"Human correction completed. The corrected mask is saved at: {final_mask}. "
                                f"Please use this corrected mask for the next evaluation step."
                            )
                            st.session_state.correction_launched = False
                            st.session_state.agent_status = "running"
                            st.rerun()
                        else:
                            st.error("result_info.json not found. Please complete correction first.")
                    else:
                        st.warning("done.txt not found yet. Please finish your corrections in the SAM tool.")
                st.markdown("---")

            else:
                # Show launch button only when a segmentation mask exists
                if last_image and last_mask:
                    if st.button("Launch Human Correction Tool (SAM)", key=f"launch_corr_{pause_key}"):
                        save_path = st.session_state.timestamp_folder
                        # Clean up any leftover flags from a previous correction
                        for flag in ["done.txt", "result_info.json"]:
                            p = os.path.join(save_path, flag)
                            if os.path.exists(p):
                                os.remove(p)
                        command = [
                            "streamlit", "run",
                            "/home/idies/workspace/Storage/xyu1/persistent/GenCELLAgent/src/tools/sam_correction_tool_micro_sam_time.py",
                            "--",
                            f"--image_path={last_image}",
                            f"--mask_path={last_mask}",
                            f"--save_path={save_path}",
                        ]
                        subprocess.Popen(command)
                        st.session_state.correction_launched = True
                        st.session_state.correction_save_path = save_path
                        st.rerun()

            # ---- Free-text message & tool selector ----
            human_msg = st.text_area(
                "Send a message to the agent (optional):",
                placeholder="e.g. 'Use oneshotsegmentation next', 'The result looks good, proceed to summarizer'",
                key=f"human_msg_{pause_key}",
                height=80,
            )

            tool_options = ["(Let agent decide)"] + [
                str(name).lower() for name in st.session_state.agent.tools.keys()
            ]
            selected_tool = st.selectbox(
                "Or force the agent to use a specific tool next:",
                options=tool_options,
                key=f"tool_select_{pause_key}",
            )

            if st.button("Continue", key=f"btn_{pause_key}"):
                feedback_parts = []
                if human_msg.strip():
                    feedback_parts.append(human_msg.strip())
                if selected_tool != "(Let agent decide)":
                    feedback_parts.append(f"Please use the '{selected_tool}' tool as your next action.")
                if feedback_parts:
                    st.session_state.human_feedback = " ".join(feedback_parts)
                else:
                    st.session_state.pop("human_feedback", None)
                st.session_state.agent_status = "running"
                st.rerun()

        # --- EXECUTION BLOCK ---
        elif agent_status == "running":
            stored_input = st.session_state.get("user_input", user_input)
            try:
                with st.spinner("Agent is working..."):
                    asyncio.run(st.session_state.agent.execute(stored_input))
                # If we reach here the agent finished without pausing
                st.session_state.agent_status = "finished"
                st.rerun()
            except AgentPauseSignal:
                # Agent paused itself — status already set to "paused" inside wait_for_human_approval
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.session_state.agent_status = "error"

        # --- FINAL ANSWER ---
        elif agent_status == "finished":
            st.success("✅ **Task Completed!**")
