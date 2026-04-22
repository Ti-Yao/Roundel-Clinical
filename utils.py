from dicom_utils import *
from model_utils import *

import os
import glob
import math
import hashlib
import shutil
from pathlib import Path
import io

import nibabel as nib
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageSequence, ImageDraw, ImageFont
from cv2 import resize, INTER_NEAREST
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from skimage.measure import label as cc_label, regionprops
from scipy.ndimage import (
    binary_fill_holes,
    binary_dilation,
    binary_erosion,
    binary_closing,
    gaussian_filter
) 
from skimage.morphology import disk,convex_hull_image
import pandas as pd
from skimage.measure import find_contours
import cv2
import json
import pydicom
import zipfile
import json
from stqdm import stqdm

root_path = Path(__file__).resolve().parent
data_path = str(root_path / "data")
results_path = str(root_path / "results")
models_path = str(root_path / "models")

blank_gif_path = f'{results_path}/temp/blank'
full_edited_gif_path = f'{results_path}/temp/edited'
preprocessed_gif_path = f'{results_path}/temp/preprocessed'
edv_esv_gif_path = f'{results_path}/temp/edv_esv'
edited_gif_path = f'{results_path}/temp/edited_edv_esv'
raw_curve_path = f'{results_path}/temp/raw_metrics.png'
edited_curve_path = f'{results_path}/temp/edited_metrics.png'
dicom_mask_path = f'{results_path}/masks/dicoms/'
nifti_mask_path = f'{results_path}/masks/nifti/'

cache_dir = f'{root_path}/cache/'
final_dir = f'{results_path}/results.zip'

os.makedirs(f'{data_path}', exist_ok=True)
os.makedirs(f'{results_path}/temp', exist_ok=True)
os.makedirs(f'{results_path}/gifs', exist_ok=True)
os.makedirs(f'{results_path}/edited_sax_df', exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

GIF_W = 150
DISPLAY_W = 400

labels = {
  "background": 0,
  "LV": 1,
  "RV": 2,
  "LV_myo": 3,
  "RV_myo": 4
}

background_idx = labels['background']
lv_idx = labels['LV']
rv_idx = labels['RV']
lv_myo_idx = labels['LV_myo']
rv_myo_idx = labels['RV_myo']

st.session_state['N'] = len(labels.keys())

BACKGROUND_COLOR = (10, 10, 10, 0) # THIS HAS TO BE NON-ZERO
RV_MYO_COLOR = (0, 200, 10, 50)    # Green
RV_COLOR = (255, 190, 10, 50)      # Yellow
LV_MYO_COLOR =  (0, 255, 255, 50)  # Blue
LV_COLOR = (255, 10, 10, 50)       # Red



OVERLAY_COLORS = {
    background_idx: BACKGROUND_COLOR,
    rv_idx: RV_COLOR,
    rv_myo_idx: RV_MYO_COLOR,
    lv_myo_idx: LV_MYO_COLOR,
    lv_idx: LV_COLOR,
}


BRUSH_LABELS = {
    rv_myo_idx: 'RV Myocardium 🟢',
    rv_idx: 'RV Blood Pool 🟡',
    lv_myo_idx: 'LV Myocardium 🔵',
    lv_idx: 'LV Blood Pool 🔴',
}

VENTRICLE_CHANNEL = {'lv':[lv_idx, lv_myo_idx],
                     'rv':[rv_idx, rv_myo_idx]}


BRUSH_LABELS = dict(
    sorted(
        BRUSH_LABELS.items(),
        key=lambda item: 0 if 'myocardium' in item[1].lower() else 1
    )
)


def load_font(size):
    # Try Linux font
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except:
        pass
    # Try Windows font
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except:
        pass
    # Fallback (non scalable)
    return ImageFont.load_default()


def save_cached_mask(mask, save_path):
    np.save(save_path, mask)

def load_cached_mask(save_path):
    return np.load(save_path)

def save_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(config, f, indent=2)

def load_config(path) :
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)

def save_mask(mask, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    nib_mask = nib.Nifti1Image(mask, affine=np.eye(4), dtype='uint8')
    nib.save(nib_mask, save_path)

def save_image(image, save_path):
    nib_image = nib.Nifti1Image(image, affine=np.eye(4), dtype='float32')
    nib.save(nib_image, save_path)

def normalize(image):
    image = (image - np.min(image))/(np.max(image) - np.min(image))
    return image

def merge_masks(lv_mask, rv_mask):
    combined_mask = lv_mask + rv_mask
    combined_mask = np.argmax(combined_mask, -1)
    combined_mask = np.eye(st.session_state['N'], dtype=np.uint8)[combined_mask]
    return combined_mask


def cv_zoom(images, zoom, interpolation=cv2.INTER_CUBIC):
    """
    Resize height and width of a 4D or 5D array using OpenCV. Only H and W are scaled.

    Args:
        images (numpy.ndarray): Array of shape (H, W, D, T) or (H, W, D, T, C)
        zoom_factors (list or tuple): Zoom factors for (H, W, D, T, C). Only H and W > 1
        interpolation (int): OpenCV interpolation method (default: cv2.INTER_CUBIC)

    Returns:
        numpy.ndarray: Resized array with height and width scaled, other dimensions unchanged
    """
    h_zoom, w_zoom = zoom[0], zoom[1]

    if images.ndim == 4:
        h, w, d, t = images.shape
        resized = np.zeros((int(h*h_zoom), int(w*w_zoom), d, t), dtype=images.dtype)
        for z in range(d):
            for tau in range(t):
                resized[..., z, tau] = cv2.resize(images[..., z, tau], (int(w*w_zoom), int(h*h_zoom)), interpolation=interpolation)
    elif images.ndim == 5:
        h, w, d, t, c = images.shape
        resized = np.zeros((int(h*h_zoom), int(w*w_zoom), d, t, c), dtype=images.dtype)
        for z in range(d):
            for tau in range(t):
                for ch in range(c):
                    resized[..., z, tau, ch] = cv2.resize(images[..., z, tau, ch], (int(w*w_zoom), int(h*h_zoom)), interpolation=interpolation)
    else:
        raise ValueError("Input must be 4D or 5D array.")

    return resized


def load_nii(nii_path):
    file = nib.load(nii_path)
    data = file.get_fdata(caching='unchanged')
    return data

def cv_zoom_mask(
    mask,
    zoom,
    sigma=2.0,
    interpolation=cv2.INTER_CUBIC,
):
    """
    mask: H,W,D,T,C
    returns: H,W,D,T,C one hot
    """
    zoomed = cv_zoom(mask.astype(np.float32), zoom, interpolation=interpolation)

    H, W, D, T, _ = zoomed.shape
    labels = np.zeros((H, W, D, T), dtype=np.uint8)

    ventricles = [
        (lv_idx, lv_myo_idx),
        (rv_idx, rv_myo_idx),
    ]

    for endo_idx, myo_idx in ventricles:
        endo = (zoomed[..., endo_idx] > 0.5).astype(np.float32)
        myo  = (zoomed[..., myo_idx] > 0.5).astype(np.float32)

        epi = np.zeros_like(myo, dtype=bool)
        for d in range(D):
            for t in range(T):
                epi[..., d, t] = binary_fill_holes(
                    myo[..., d, t].astype(np.uint8)
                )

        epi = gaussian_filter(
            epi.astype(np.float32), sigma=(sigma, sigma, 0, 0)
        ) > 0.5

        endo = gaussian_filter(
            endo.astype(np.float32), sigma=(sigma, sigma, 0, 0)
        ) > 0.5

        labels[epi] = myo_idx
        labels[endo] = endo_idx

    return np.eye(st.session_state['N'], dtype=np.uint8)[labels]

def format_delta(value, raw_value, suffix="", round_digits=None):
    if round_digits is not None:
        value = round(value, round_digits)
        raw_value = round(raw_value, round_digits)
    return None if value == raw_value else f"{value - raw_value:.1f}{suffix}"


def find_crop_box(mask, crop_factor):
    '''
    Calculated a bounding box that contains the masks inside.

    Parameters:
    mask: np.array
        A binary mask array, which should be the flattened 3D multislice mask, where the pixels in the z-dimension are summed
    crop_factor: float
        A scaling factor for the bounding box
    Returns:
    list
        A list containing the coordinates of the bounding box [x_min, y_min, x_max, y_max]. These co-ordinates can be used to crop each slice of the input multislice image.
    '''
    # Check shape of the input is 2D
    if len(mask.shape) != 2:
        raise ValueError("Input mask must be a 2D array")

    if np.max(mask) == 0:
        x_min, x_max = 0, mask.shape[0]
        y_min, y_max = 0, mask.shape[1]
        return [x_min, y_min, x_max, y_max]

    else:
        y = np.sum(mask, axis=1) # sum the masks across columns of array, returns a 1D array of row totals
        x = np.sum(mask, axis=0) # sum the masks across rows of array, returns a 1D array of column totals

        top = np.min(np.nonzero(y)) - 1 # Returns the indices of the elements in 1d row totals array that are non-zero, then finds the minimum value and subtracts 1 (i.e. top extent of mask)
        bottom = np.max(np.nonzero(y)) + 1 # Returns the indices of the elements in 1d row totals array that are non-zero, then finds the maximum value and adds 1 (i.e. bottom extent of mask)

        left = np.min(np.nonzero(x)) - 1 # Returns the indices of the elements in 1d column totals array that are non-zero, then finds the minimum value and subtracts 1 (i.e. left extent of mask)
        right = np.max(np.nonzero(x)) + 1 # Returns the indices of the elements in 1d column totals array that are non-zero, then finds the maximum value and adds 1 (i.e. right extent of mask)
        if abs(right - left) > abs(top - bottom):
            largest_side = abs(right - left) # Find the largest side of the bounding box
        else:
            largest_side = abs(top - bottom)

        
        x_mid = round((left + right) / 2) # Find the mid-point of the x-length of mask
        y_mid = round((top + bottom) / 2) # Find the mid-point of the y-length of mask
        half_largest_side = round(largest_side * crop_factor / 2) # Find half the largest side of the bounding box (crop factor scales the largest side to ensure whole heart and some surrounding is captured)
        x_max, x_min = round(x_mid + half_largest_side), round(x_mid - half_largest_side) # Find the maximum and minimum x-values of the bounding box
        y_max, y_min = round(y_mid + half_largest_side), round(y_mid - half_largest_side) # Find the maximum and minimum y-values of the bounding box
        if x_min < 0:
            x_max -= x_min # if x_min less than zero, expand the x_max value by the absolute value of x_min, to ensure bounding box is same size
            x_min = 0

        if y_min < 0:
            y_max -= y_min # if y_min less than zero, expand the y_max value by the absolute value of y_min, to ensure bounding box is same size
            y_min = 0

        if largest_side < 20:
            x_min, x_max = 0, mask.shape[0]
            y_min, y_max = 0, mask.shape[1]
        return [x_min, y_min, x_max, y_max]


def make_video(image, mask, save_file, ventricle = 'all', mask_frames = 'all', scale=1):
    N = st.session_state['N']
    if ventricle == 'rv':
        channels = [rv_idx, rv_myo_idx]
    elif ventricle == 'lv':
        channels = [lv_idx, lv_myo_idx]
    else:
        channels = [n for n in np.arange(N) if n != background_idx]

    if mask.shape[-1]!=N:
        mask = np.eye(N, dtype=np.uint8)[mask]

    position = image.shape[2]
    timesteps = image.shape[3]

    grid_rows = int(np.sqrt(position) + 0.5)
    grid_cols = (position + grid_rows - 1) // grid_rows

    H, W = image.shape[:2]
    GIF_H = H*GIF_W/W
    H_scaled, W_scaled = round(GIF_H * scale), round(GIF_W * scale)

    try:
        font = load_font(int(20 * scale))
    except:
        font = ImageFont.load_default()

    frames = []
    if mask_frames == 'all':
        mask_frames = np.arange(timesteps)

    for t in mask_frames:
        canvas = Image.new(
            "RGBA",
            (grid_cols * W_scaled, grid_rows * H_scaled),
            color=(0, 0, 0, 255)
        )

        draw_canvas = ImageDraw.Draw(canvas)

        for idx in range(position):
            row, col = divmod(idx, grid_cols)
        
            img_slice = image[:, :, idx, t]

            p1, p99 = np.percentile(img_slice, [0.5, 99.5]) # improve contrast
            img_slice = np.clip(img_slice, p1, p99)

            # Convert to RGB
            img_slice_norm = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-9) * 255).astype(np.uint8)
            img_rgb = np.stack([img_slice_norm] * 3, axis=-1)
            img_pil = Image.fromarray(img_rgb, mode="RGB").convert("RGBA")

            # Resize slice
            img_pil = img_pil.resize((W_scaled, H_scaled), resample=Image.NEAREST)

            overlay = np.zeros((H, W, 4), dtype=np.uint8)
            for ch in channels:
                ch_mask = mask[:,:,idx,t,ch]
                if np.any(ch_mask):
                    color = np.array(OVERLAY_COLORS[ch], dtype=np.uint8)
                    overlay[ch_mask > 0] = color
            overlay_pil = Image.fromarray(overlay, mode="RGBA").resize((W_scaled, H_scaled), resample=Image.NEAREST)
            img_pil.alpha_composite(overlay_pil)

            draw_tile = ImageDraw.Draw(img_pil)
            draw_tile.rectangle([0,0,int(28*scale), int(22*scale)], fill=(211,211,211,255))
            draw_tile.text((3*scale,2*scale), f"{idx}", fill=(0,0,0,255), font=font)

            canvas.paste(img_pil, (col * W_scaled, row * H_scaled), img_pil)

        draw_canvas.rectangle(
            [canvas.width - int(60*scale), canvas.height - int(20*scale),
             canvas.width, canvas.height],
            fill=(211,211,211,255)
        )
        draw_canvas.text(
            (canvas.width - int(55*scale), canvas.height - int(20*scale)),
            f"{t:02}/{timesteps - 1:02}",
            fill=(0,0,0,255),
            font=font
        )

        frames.append(canvas.convert("RGB"))

    if len(mask_frames) < 5:
        fps = len(mask_frames)/2
    else:
        fps = np.clip(len(mask_frames) / 2, 8, 15)

    save_file = save_file.replace('.gif','')
    imageio.mimsave(f'{save_file}.gif', frames, fps=fps, loop=0)


def calculate_sax_metrics(mask, blood_pool_idx, myo_idx, dia_idx, sys_idx):
    voxel_size = st.session_state.pixelspacing ** 2 * st.session_state.thickness / 1000
    volume = np.sum(mask[..., blood_pool_idx], axis=(0,1,2)) * voxel_size
    masses = np.sum(mask[..., myo_idx], axis=(0,1,2)) * voxel_size * 1.05
    mass = masses[dia_idx]
    edv = volume[dia_idx]
    esv = volume[sys_idx]
    sv = edv - esv
    ef = (sv) * 100/edv
    return volume, masses, edv, esv, sv, ef, mass



def thicken_close_fill_and_smooth(strokes, stroke_width):
    if strokes is None or not strokes.any():
        return strokes

    # Use power-law scaling for dilation
    dilation_factor = max(1, int(10 / (stroke_width ** 2)))

    # Detect contours to check for nested shapes
    dilated = binary_dilation(strokes, iterations=dilation_factor)
    contours = find_contours(dilated, 0.5)

    has_ring = False
    for i, c1 in enumerate(contours):
        for j, c2 in enumerate(contours):
            if i == j:
                continue
            y1, x1 = c1[:, 0], c1[:, 1]
            y2, x2 = c2[:, 0], c2[:, 1]
            if (y2.min() > y1.min() and y2.max() < y1.max() and
                x2.min() > x1.min() and x2.max() < x1.max()):
                has_ring = True
                break
        if has_ring:
            break

    if has_ring:
        # Dilation + fill + erosion
        closed = binary_dilation(strokes, iterations=dilation_factor)
        filled = binary_fill_holes(closed)
        filled = binary_erosion(filled, iterations=dilation_factor)
        return filled.astype('uint8')
    else:
        return strokes.astype('uint8')


def wrap(key, min_val, max_val):
    if st.session_state[key] > max_val:
        st.session_state[key] = min_val
    elif st.session_state[key] < min_val:
        st.session_state[key] = max_val



def frame_index_slider(
    T,
    frames,
    initial_idx,
    label,
    disabled_flag,
    key
):
    idx = st.slider(
        f"{label} | *{initial_idx}*",
        -1,
        T,
        value=initial_idx,
        key = key,
        on_change=wrap,
        args=(key, 0, T-1),
        disabled=disabled_flag
    )
    st.image(frames[idx], use_container_width=True)
    return idx

def copy_frames_channels(mask_name, dia_idx, sys_idx, blood_idx, myo_idx):
    frames = [dia_idx, sys_idx]
    channels = [blood_idx, myo_idx]

    mask = st.session_state[mask_name]
    if st.session_state.get('corrector_mask_edited'):
        source_mask = st.session_state['corrected_prior_mask']
        print("[DEBUG] Using corrected prior mask as source for final result.")
        print(f"[DEBUG] source_mask shape: {source_mask.shape}, mask shape: {mask.shape}")
    else:
        source_mask = st.session_state.preprocessed["smooth_mask"]
        print("[DEBUG] Using intial segmentation mask as source for final result.")
        print(f"[DEBUG] source_mask shape: {source_mask.shape}, mask shape: {mask.shape}")

    for f in frames:
        for c in channels:
            mask[:, :, :, f, c] = source_mask[:, :, :, f, c]

def confirm_selection(lv_dia_idx, lv_sys_idx,rv_dia_idx, rv_sys_idx):
    """Store confirmed EDV/ESV indices in session state."""
    st.session_state['edv_esv_selected'].update({
        "lv_dia_idx": lv_dia_idx,
        "lv_sys_idx": lv_sys_idx,
        "rv_dia_idx": rv_dia_idx,
        "rv_sys_idx": rv_sys_idx,
        "confirmed": True
    })

    save_config(st.session_state['edv_esv_selected'], st.session_state['cache_config_path'])

    # LV
    copy_frames_channels('edited_mask_lv', lv_dia_idx, lv_sys_idx, lv_idx, lv_myo_idx)

    # RV
    copy_frames_channels('edited_mask_rv', rv_dia_idx, rv_sys_idx, rv_idx, rv_myo_idx)

    save_cached_mask(merge_masks(st.session_state['edited_mask_lv'],st.session_state['edited_mask_rv']), save_path=st.session_state['cache_mask_path'])

    make_video(
        st.session_state.preprocessed['smooth_image'],
        st.session_state['edited_mask_lv'],
        mask_frames = [lv_dia_idx, lv_sys_idx],
        save_file=f'{edited_gif_path}_lv',

    )

    make_video(
        st.session_state.preprocessed['smooth_image'],
        st.session_state['edited_mask_rv'],
        mask_frames = [rv_dia_idx, rv_sys_idx],
        save_file=f'{edited_gif_path}_rv',
        ventricle = 'rv'
    )

    gif = Image.open(f'{edited_gif_path}_lv.gif')
    lv_frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
    st.session_state['lv_frames'] = lv_frames

    gif = Image.open(f'{edited_gif_path}_rv.gif')
    rv_frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
    st.session_state['rv_frames'] = rv_frames

def resize_to_original(edited_mask, raw_mask, crop_box, dia_idx, sys_idx, ventricle):
    """
    Place the edited mask back into the original full-size mask array.
    Assumes edited_mask has shape (H_crop, W_crop, C, 2, num_classes)
    """
    x_min, y_min, x_max, y_max = crop_box
    final_mask_2d = np.zeros_like(raw_mask)

    channels = [rv_idx, rv_myo_idx] if ventricle == 'rv' else [lv_idx, lv_myo_idx]

    for ch in channels:
        final_mask_2d[y_min:y_max, x_min:x_max, ch, dia_idx, :] = edited_mask[:, :, ch, dia_idx, :]
        final_mask_2d[y_min:y_max, x_min:x_max, ch, sys_idx, :] = edited_mask[:, :, ch, sys_idx, :]

    final_mask_2d = np.argmax(final_mask_2d, axis=-1)
    print(np.unique(final_mask_2d))
    return final_mask_2d


def make_true_vs_prior_vs_pred_mask_3d_gif(image, true_mask, prior_mask, pred_mask, gif_name = 'example_prediction', prior_image=None):
    ''' 
    Creates a side-by-side gif of the true vs prior vs predicted masks for a given image, with an overlay of the true, prior and predicted masks on the original image. 
    The function takes in the original image, the true mask, the prior mask, the predicted mask, a name for the gif, and optionally clinical data for the patient and a wandb run object to upload the gif. 
    The function checks the shape of the input image and transposes it if necessary to ensure it is in the format [z,y,x]. 
    It then creates a custom color map for the masks, extracts clinical data if provided, and creates a figure with two subplots for the true and predicted mask overlays. 
    It iterates through each slice of the image, creates the overlays, and appends them to a list of frames. 
    Finally, it creates an animation from the frames, saves it as a gif, uploads it to WandB if a run object is provided, and closes the plot to free memory.
    '''
    # print(f"image shape on entering gif code: {image.shape}")
    # Check if image_shape is in shape [z,y,x], and if not, change to this format
    if (image.shape[0] < image.shape[1]) & (image.shape[0] < image.shape[2]): # should be less slices than pixels in either x or y dimensions
            image = image
            true_mask = true_mask
            prior_mask = prior_mask
            pred_mask = pred_mask
            if prior_image is not None:
                prior_image = prior_image
    else:
            image = np.transpose(image, (2, 0, 1))
            true_mask = np.transpose(true_mask, (2, 0, 1))
            prior_mask = np.transpose(prior_mask, (2, 0, 1))
            pred_mask = np.transpose(pred_mask, (2, 0, 1))
            if prior_image is not None:
                prior_image = np.transpose(prior_image, (2, 0, 1))

    # print(f"image shape after transpose: {image.shape}")
    
    # Define the custom color map
    colour_map = {
        0: "black", # background
        1: "red", # left ventricle
        2: "blue", # right ventricle
        3: "green", # lv myocardium
        4: "yellow", # left atrium
        5: "purple", # right atrium
        6: "orange", # aorta
        7: "pink", # pulmonary artery
        8: "brown", # svc
        9: "cyan" # ivc
    }
    # Create a ListedColormap using the values from the dictionary
    custom_cmap = ListedColormap([colour_map[key] for key in sorted(colour_map.keys())])

    # Create figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 10))
    ax1, ax2, ax3 = axes  # Unpack the axes for easier referencing

    frames = []
    slice_position = image.shape[0]

    for pos in range(slice_position):
        # True mask overlay
        p1 = ax1.imshow(image[pos, :, :], cmap='gray', vmin=np.min(image), vmax=np.max(image))
        p2 = ax1.imshow(true_mask[pos, :, :], alpha=0.2, cmap=custom_cmap, 
                        vmin=0, vmax=len(colour_map) - 1, interpolation='none')
        ax1.set_title(f"{gif_name}\nTrue Mask Overlay")
        ax1.axis('off')

        # Prior mask overlay
        if prior_image is None:
            p3 = ax2.imshow(image[pos, :, :], cmap='gray', vmin=np.min(image), vmax=np.max(image))
            p4 = ax2.imshow(prior_mask[pos, :, :], alpha=0.2, cmap=custom_cmap, 
                            vmin=0, vmax=len(colour_map) - 1, interpolation='none')
            ax2.set_title(f"{gif_name}\nPrior Mask Overlay")
            ax2.axis('off')
        elif prior_image is not None:
            p3 = ax2.imshow(prior_image[pos, :, :], cmap='gray', vmin=np.min(prior_image), vmax=np.max(prior_image))
            p4 = ax2.imshow(prior_mask[pos, :, :], alpha=0.2, cmap=custom_cmap, 
                            vmin=0, vmax=len(colour_map) - 1, interpolation='none')
            ax2.set_title(f"{gif_name}\nPrior Mask Overlay")
            ax2.axis('off')

        # Predicted mask overlay
        p5 = ax3.imshow(image[pos, :, :], cmap='gray', vmin=np.min(image), vmax=np.max(image))
        p6 = ax3.imshow(pred_mask[pos, :, :], alpha=0.2, cmap=custom_cmap, 
                        vmin=0, vmax=len(colour_map) - 1, interpolation='none')
        ax3.set_title(f"{gif_name}\nPredicted Mask Overlay")
        ax3.axis('off')

    
        # Append the frames for animation
        frames.append([p1, p2, p3, p4, p5, p6])

    # Check there are the correct number of frames
    if len(frames) != slice_position:
        raise ValueError("Number of frames is not equal to the number of slices in the image")

    # Create and upload animation to neptune
    output_dir = "True_vs_Prior_vs_Predicted_Mask_gifs"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    gif_path = os.path.join(output_dir, f"{gif_name}.gif")  # Save in the directory with a unique name
    ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True)
    ani.save(gif_path, fps=3)

    plt.close()  # Close the plot to free memory

def plot_image_with_prior_masks(image, gif_name='example_prediction'):
    """Plots a 2d slice of a 3D image with its corresponding prior masks side by side, and an overlay of
    the masks on both the target image (channel 0) and the prior image (channel 1). Saves the plot to disk."""
    # image is (H, W, D, C) where channel 0 = target image, channel 1 = prior image, channels 2+ = prior mask classes
    num_prior_masks = image.shape[3] - 2  # subtract target and prior image channels
    # columns: target image, prior image, each mask channel, overlay on target, overlay on prior
    n_cols = num_prior_masks + 4
    fig, axes = plt.subplots(1, n_cols, figsize=(20 * n_cols, 20))

    mid_slice = image.shape[2] // 2
    colors = plt.cm.tab10.colors  # up to 10 distinct colours

    # Target image
    axes[0].imshow(image[:, :, mid_slice, 0], cmap='gray')
    axes[0].set_title('Target Image')
    axes[0].axis('off')

    # Prior image
    axes[1].imshow(image[:, :, mid_slice, 1], cmap='gray')
    axes[1].set_title('Prior Image')
    axes[1].axis('off')

    # Prior mask channels (one per column)
    for m in range(num_prior_masks):
        axes[2 + m].imshow(image[:, :, mid_slice, 2 + m], cmap='gray')
        axes[2 + m].set_title(f'Prior Mask {m + 1}')
        axes[2 + m].axis('off')

    def _draw_overlay(ax, bg_channel, title):
        ax.imshow(image[:, :, mid_slice, bg_channel], cmap='gray')
        for m in range(num_prior_masks):
            mask = image[:, :, mid_slice, 2 + m]
            color = colors[m % len(colors)]
            colored = np.zeros((*mask.shape, 4))  # RGBA
            colored[..., :3] = color[:3]
            colored[..., 3] = mask * 0.5  # alpha proportional to mask intensity
            ax.imshow(colored)
        ax.set_title(title)
        ax.axis('off')

    # Overlay on target image
    _draw_overlay(axes[-2], bg_channel=0, title='Target + Prior Masks')
    # Overlay on prior image
    _draw_overlay(axes[-1], bg_channel=1, title='Prior + Prior Masks')

    output_dir = "image_with_prior_masks_plots"
    # create output dir if doesnt exist
    os.makedirs(output_dir, exist_ok=True)

    plt.suptitle(gif_name)
    plt.savefig(f"{output_dir}/{gif_name}.png")
    plt.close()
