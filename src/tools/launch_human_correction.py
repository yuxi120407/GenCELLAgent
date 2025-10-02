import subprocess
import os
from typing import Optional, Dict
import time
import json

def launch_sam_correction_tool(image_path: str, mask_path: str, save_path: Optional[str] = None) -> Dict:
    """
    Launches the SAM-based correction tool in a separate Streamlit process.

    Args:
        image_path (str): Path to the input image.
        mask_path (str): Path to the predicted mask.
        save_path (str, optional): Directory to save corrected output. Defaults to the current directory.

    Returns:
        Dict: Observation indicating the tool was launched.
    """
    if save_path is None:
        save_path = "."
        
        
    print("into tooL!!")
    done_flag = os.path.join(save_path, "done.txt")
    result_info_path = os.path.join(save_path, "result_info.json")
    
    try:
        # Clean up any previous flags
        if os.path.exists(done_flag):
            os.remove(done_flag)
        if os.path.exists(result_info_path):
            os.remove(result_info_path)
        
        command = [
            "streamlit", "run",
            "/home/idies/workspace/Storage/xyu1/persistent/Langchain/ours_test/src/tools/sam_correction_tool_micro_sam.py",
            "--", f"--image_path={image_path}", f"--mask_path={mask_path}", f"--save_path={save_path}"
        ]
        subprocess.Popen(command)
        print("🟡 Streamlit launched. Waiting for done.txt...")


            
        while not os.path.exists(done_flag):
            print("⏳ Waiting for human correction...")
            time.sleep(3)

        # After done.txt is found, read result_info
        if os.path.exists(result_info_path):
            with open(result_info_path, "r") as f:
                result_info = json.load(f)
            final_mask_path = os.path.join(save_path, result_info["filename"])
        else:
            raise FileNotFoundError("result_info.json not found after done.txt appeared.")

        return {
            "status": "human_correction_complete",
            "human_correction_mask_path": final_mask_path,
            "note": f"Correction complete and mask saved as {result_info['filename']}."
        }

    except Exception as e:
        return {
            "status": "launch_failed",
            "error": str(e)
        }