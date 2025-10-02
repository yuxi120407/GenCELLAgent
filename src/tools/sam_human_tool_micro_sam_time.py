import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from collections import defaultdict
from segment_anything import sam_model_registry, SamPredictor
from scipy.ndimage import binary_dilation
from skimage.draw import polygon
import os
import json
import time
from datetime import datetime

@st.cache_resource
def load_sam_model():
    checkpoint = "/home/idies/workspace/Storage/xyu1/persistent/segment-anything/notebooks/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
    return SamPredictor(sam)

@st.cache_resource
def load_micro_sam_model():
    checkpoint = "/home/idies/workspace/Temporary/xyu1/scratch/micro_sam_model/EM/vit_l.pt"
    sam = sam_model_registry["vit_l"](checkpoint=checkpoint)
    return SamPredictor(sam)

def sam_human_correction_ui(image_path: str, mask_path: str, save_path: str) -> str:
    predictor = load_micro_sam_model()
    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)
    img_h, img_w = img_np.shape[:2]
    scale_factor = 6

    mask_img = Image.open(mask_path).convert("L")
    binary_mask = (np.array(mask_img) > 127).astype(bool)
    cleaned_mask = remove_small_objects(label(binary_mask), min_size=100)
    st.session_state.initial_mask = cleaned_mask.astype(np.uint8)  # ← use session state

    # Initialize timing when GUI first appears
    if "correction_start_time" not in st.session_state:
        st.session_state.correction_start_time = time.time()
        st.session_state.correction_start_datetime = datetime.now()
    
    # Display start time info
    st.sidebar.markdown("### ⏰ Timing Information")
    st.sidebar.write(f"**Start Time:** {st.session_state.correction_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show elapsed time (updates in real-time)
    elapsed_time = time.time() - st.session_state.correction_start_time
    st.sidebar.write(f"**Elapsed Time:** {int(elapsed_time//60):02d}:{int(elapsed_time%60):02d}")

    initial_mask = st.session_state.initial_mask
    labeled_mask = label(initial_mask)
    props = regionprops(labeled_mask)
    region_points_map = defaultdict(list)
    bbox_list = []

    for i, region in enumerate(props):
        min_y, min_x, max_y, max_x = region.bbox
        bbox_list.append((min_x, min_y, max_x, max_y))
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                if initial_mask[y, x] > 0:
                    region_points_map[i].append([x, y])  # x, y order for consistency
    
    st.markdown("## ✍️ Human Correction & Annotation Interface")
    st.markdown("Use the tools below to draw points and polygons on the image.")
    
    if "human_uploaded_image" not in st.session_state:
        st.session_state.human_uploaded_image = None
    if "human_uploaded_mask" not in st.session_state:
        st.session_state.human_uploaded_mask = None

    draw_mode = st.selectbox("Draw annotation as", [
        "Point: Positive 🔴",
        "Point: Negative 🔵",
        #"Box: Missed Object ⬛",
        "Polygon: Positive Region 🟡",
        "Polygon: Negative Region 🔷"
    ])

    stroke_color = {
        "Point: Positive 🔴": "#ff0000",
        "Point: Negative 🔵": "#0000ff",
        #"Box: Missed Object ⬛": "#00ff00",
        "Polygon: Positive Region 🟡": "#ffff00",
        "Polygon: Negative Region 🔷": "#0000ff"
    }[draw_mode]

    drawing_mode = {
        "Point: Positive 🔴": "point",
        "Point: Negative 🔵": "point",
        #"Box: Missed Object ⬛": "rect",
        "Polygon: Positive Region 🟡": "polygon",
        "Polygon: Negative Region 🔷": "polygon"
    }[draw_mode]

    scaled_width = int(img_w * scale_factor)
    scaled_height = int(img_h * scale_factor)
    #resized_image = image.resize((scaled_width, scaled_height))
    
    # Create a red mask overlay on top of the original image
    mask_overlay = img_np.copy()
    mask_overlay[initial_mask > 0] = [255, 0, 0]  # red for initial mask
    blended_overlay = cv2.addWeighted(img_np, 0.5, mask_overlay, 0.5, 0)
    resized_image = Image.fromarray(blended_overlay).resize((scaled_width, scaled_height))
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 0, 0.3)" if drawing_mode == "polygon" else "rgba(255, 0, 0, 0.3)",
        stroke_width=3,
        stroke_color=stroke_color,
        background_image=resized_image,
        update_streamlit=True,
        height=scaled_height,
        width=scaled_width,
        drawing_mode=drawing_mode,
        key="correction_canvas"
    )

    if "user_corrections" not in st.session_state:
        st.session_state.user_corrections = []
    if "user_boxes" not in st.session_state:
        st.session_state.user_boxes = []
    if "user_polygons" not in st.session_state:
        st.session_state.user_polygons = []

    if canvas_result.json_data:
        for obj in canvas_result.json_data["objects"]:
            if obj["type"] == "rect":
                x0 = int(obj["left"] / scale_factor)
                y0 = int(obj["top"] / scale_factor)
                w = int(obj["width"] / scale_factor)
                h = int(obj["height"] / scale_factor)
                x1, y1 = x0 + w, y0 + h
                st.session_state.user_boxes.append((x0, y0, x1, y1))

            elif obj["type"] == "circle":
                x = int(obj["left"] / scale_factor)
                y = int(obj["top"] / scale_factor)
                color = obj.get("stroke", "#ff0000").lower()
                point_label = 0 if color.startswith("#00") else 1
                st.session_state.user_corrections.append((x, y, point_label))

            elif obj["type"] == "path":
                path = obj.get("path", [])
                points = [
                    (int(p[1] / scale_factor), int(p[2] / scale_factor))
                    for p in path if p[0] in ['M', 'L']
                ]
                stroke_color = obj.get("stroke", "").lower()
                poly_label = 0 if stroke_color.startswith("#0000ff") else 1
                if len(points) >= 3:
                    st.session_state.user_polygons.append((poly_label, points))

    def get_bbox_containing_point(x, y):
        for i, (min_x, min_y, max_x, max_y) in enumerate(bbox_list):
            if min_x <= x < max_x and min_y <= y < max_y:
                return i
        return None

    #col_sam, col_human, col_reset = st.columns([1, 1, 1])
    col_human, col_reset = st.columns([1, 1])

    with col_human:
        human_clicked = st.button("Apply Human Correction & Annotation")

    with col_reset:
        if st.button("🔄 Reset Corrections"):
            st.session_state.user_corrections = []
            st.session_state.user_boxes = []
            st.session_state.user_polygons = []
            st.success("Reset all annotations.")
            st.rerun()

    if human_clicked:
        final_points = []
        final_labels = []
        region_status = {rid: None for rid in region_points_map.keys()}
        human_positive = []
        human_negative = []
        sam_points = []
        sam_labels = []

        # Ensure initial_mask is linked from session
        if "initial_mask" not in st.session_state:
            st.session_state.initial_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        initial_mask = st.session_state.initial_mask

        # Create polygon mask
        polygon_coords = set()
        polygon_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        for label_val, poly_pts in st.session_state.user_polygons:
            xs, ys = zip(*poly_pts)
            rr, cc = polygon(ys, xs, shape=img_np.shape[:2])
            polygon_mask[rr, cc] = 1
            polygon_coords.update(zip(cc, rr))  # x, y format

        # Process point clicks
        for x, y, label_val in st.session_state.user_corrections:
            if (x, y) in polygon_coords:
                continue  # Skip if this point is from a polygon (already handled)

            region_idx = get_bbox_containing_point(x, y)
            if region_idx is not None:
                region_status[region_idx] = label_val
                if label_val == 0:
                    for rx, ry in region_points_map[region_idx]:
                        initial_mask[ry, rx] = 0
            else:
                if label_val == 1:
                    # ✅ Add to SAM point list
                    sam_points.append([x, y])
                    sam_labels.append(1)
                else:
                    initial_mask[y, x] = 0

        # Region prompt propagation (fixed indentation)
        for rid, pts in region_points_map.items():
            label_val = region_status[rid]
            if label_val is None:
                continue
            for x, y in pts:
                if polygon_mask[y, x] == 0:
                    final_points.append([x, y])
                    final_labels.append(label_val)
                    (human_positive if label_val == 1 else human_negative).append([x, y])

        # Process polygons
        for label_val, poly_pts in st.session_state.user_polygons:
            xs, ys = zip(*poly_pts)
            rr, cc = polygon(ys, xs, shape=img_np.shape[:2])
            #pts = list(zip(cc, rr))

            if label_val == 1:
                initial_mask[rr, cc] = np.maximum(initial_mask[rr, cc], 1)

            else:
                initial_mask[rr, cc] = 0

        # Run SAM for positive points
        if sam_points:
            predictor.set_image(img_np)
            merged_sam_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)

            for pt in sam_points:
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([pt]),
                    point_labels=np.array([1]),
                    multimask_output=True
                )
                best_mask = masks[np.argmax(scores)]
                merged_sam_mask = np.maximum(merged_sam_mask, best_mask.astype(np.uint8))

            st.session_state.sam_mask = merged_sam_mask
        else:
            st.session_state.sam_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)

        st.session_state.human_mask = initial_mask.astype(np.uint8)

        overallmask = np.maximum(st.session_state.sam_mask, st.session_state.human_mask)
        st.session_state.overallmask = overallmask.astype(np.uint8)

    # --- Preview Both ---
    if "sam_mask" in st.session_state or "human_mask" in st.session_state:
        st.markdown("### 🖼️ Preview Masks Before Saving")
        col1, col2 = st.columns(2)

        if "overallmask" in st.session_state:
            overlay_human = img_np.copy()
            overlay_human[st.session_state.overallmask > 0] = [255, 0, 0]
            blended_human = cv2.addWeighted(img_np, 0.5, overlay_human, 0.5, 0)
            with col1:
                st.image(blended_human, caption="Human-Only Annotation", use_column_width=True)

        st.markdown("### 💾 Choose and Save Final Mask")
        available_options = []
        if "human_mask" in st.session_state:
            available_options.append("Human Only")
        if "sam_mask" in st.session_state:
            available_options.append("SAM Refined")

        choice = st.radio("Choose which mask to save:", options=available_options)
        if st.button("✅ Save Selected Final Mask"):
            # Calculate final correction time
            correction_end_time = time.time()
            total_correction_time = correction_end_time - st.session_state.correction_start_time
            correction_end_datetime = datetime.now()
            
            final = (st.session_state.human_mask if choice == "Human Only"
                     else st.session_state.sam_mask)
            
            save_filename = "final_mask_human.npy" if choice == "Human Only" else "final_mask_sam.npy"
            
            save_fullpath = os.path.join(save_path, save_filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_fullpath), exist_ok=True)
            
            # Save mask
            np.save(save_fullpath, final)
            
            # Save result metadata including timing information
            result_info = {
                "choice": choice,
                "filename": save_filename,
                "timing": {
                    "start_time": st.session_state.correction_start_datetime.isoformat(),
                    "end_time": correction_end_datetime.isoformat(),
                    "total_correction_time_seconds": total_correction_time,
                    "total_correction_time_formatted": f"{int(total_correction_time//60):02d}:{int(total_correction_time%60):02d}"
                }
            }
            with open(os.path.join(save_path, "result_info.json"), "w") as f:
                json.dump(result_info, f, indent=2)
    
            # Write the done.txt flag
            done_flag = os.path.join(save_path, "done.txt")
            with open(done_flag, "w") as f:
                f.write("done")
    
            # Display success message with timing info
            st.success(f"✅ Final mask saved to `{save_path}`")
            st.success(f"⏱️ Total correction time: {int(total_correction_time//60):02d}:{int(total_correction_time%60):02d} (minutes:seconds)")
            st.markdown("✅ You may now close this window and return to the main LLM interface.")
            
            # Also save timing info to a separate file for easy access
            timing_file = os.path.join(save_path, "timing_log.txt")
            with open(timing_file, "w") as f:
                f.write(f"Human Correction Timing Log\n")
                f.write(f"===========================\n")
                f.write(f"Start Time: {st.session_state.correction_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"End Time: {correction_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Duration: {total_correction_time:.2f} seconds\n")
                f.write(f"Total Duration: {int(total_correction_time//60):02d}:{int(total_correction_time%60):02d} (MM:SS)\n")
                f.write(f"Choice Made: {choice}\n")
            
            st.stop()  # stop execution cleanly

    return None