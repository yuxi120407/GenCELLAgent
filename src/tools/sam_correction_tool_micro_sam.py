import argparse
import streamlit as st
from sam_human_tool_micro_sam import sam_human_correction_ui
import json

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, required=True)
parser.add_argument("--mask_path", type=str, required=True)
parser.add_argument("--save_path", type=str, default="sam_final_mask.npy")
args, _ = parser.parse_known_args()

sam_human_correction_ui(args.image_path, args.mask_path, args.save_path)