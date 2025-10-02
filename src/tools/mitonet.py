import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2


def mitonet_inference(image_path: str) -> str:
    save_path = "/home/idies/workspace/Storage/xyu1/persistent/Langchain/ours_test/segment_resutls/MitoNet_results.png"
    image_save_path = save_path  # Original save path for the overlay image
    mask_save_path = save_path.replace('.png', '_mask.png')  # Modify as needed
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load("/home/idies/workspace/Storage/xyu1/persistent/Langchain/MitNet/MitoNet_v1.pth", map_location=device) 
    model.eval()

    # --- Load and preprocess the image for inference ---
    # Load as grayscale
    img = Image.open(image_path).convert("L")
    # Convert to numpy array (float32) and scale to [0,1]
    img_array = np.array(img, dtype=np.float32)
    img_array_scaled = img_array / 255.0

    # Normalize using the training mean and std for MitoNet
    mean, std = 0.57571, 0.12765
    img_array_norm = (img_array_scaled - mean) / std

    # Convert to a tensor with shape (1, 1, H, W)
    img_tensor = torch.from_numpy(img_array_norm).unsqueeze(0).unsqueeze(0).to(device)

    # Store original dimensions
    original_H, original_W = img_array.shape

    # Pad the image so its height and width are multiples of 16 (if necessary)
    H, W = img_tensor.shape[2], img_tensor.shape[3]
    pad_bottom = (16 - H % 16) % 16
    pad_right  = (16 - W % 16) % 16
    if pad_bottom or pad_right:
        img_tensor = F.pad(img_tensor, (0, pad_right, 0, pad_bottom))

    # --- Run inference ---
    with torch.no_grad():
        output = model(img_tensor)

    # Process the output: For binary segmentation, apply sigmoid activation.
    sem_logits = output['sem_logits']            # expected shape: [1, 1, padded_H, padded_W]
    mito_prob = torch.sigmoid(sem_logits[0, 0])      # probability map for mitochondria
    mito_mask = (mito_prob > 0.5).cpu().numpy()      # binary mask

    # Crop the segmentation mask to the original image dimensions
    mito_mask_cropped = mito_mask[:original_H, :original_W]

    # --- Prepare the image for visualization ---
    # Convert the original grayscale image (img_array) to an RGB image
    img_rgb = np.stack([img_array.astype(np.uint8)] * 3, axis=-1)

    # --- Overlay the mask ---
    # Create an overlay color image: red ([255, 0, 0])
    overlay = np.zeros_like(img_rgb)
    overlay[..., 0] = 255  # red channel

    # Define the transparency (alpha) factor for the overlay
    alpha = 0.5

    # Create a copy of the original image to blend
    blended = img_rgb.copy()

    # Use the cropped mask for overlay
    blended[mito_mask_cropped > 0] = ((1 - alpha) * img_rgb[mito_mask_cropped > 0] + alpha * overlay[mito_mask_cropped > 0]).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_save_path, overlay_bgr)
    
    pred_mask = mito_mask_cropped > 0
    mask_to_save = pred_mask.astype(np.uint8) * 255
    cv2.imwrite(mask_save_path, mask_to_save)
    
    print(f"MitoNet completed and the segmentation results saved at {image_save_path}, the corresponding mask saved at {mask_save_path}")
    
    return f"MitoNet completed and the segmentation results saved at MitoNet_image_path: {image_save_path}, the corresponding mask saved at MitoNet_mask_path: {mask_save_path}"
    
    
    