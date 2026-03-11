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
from pipeline_utils import Pipeline


root_path = f'./roundel/'
data_path = f'{root_path}/data/'
results_path = f'{root_path}/results/'

blank_gif_path = f'{results_path}/temp/blank'
full_edited_gif_path = f'{results_path}/temp/edited'
preprocessed_gif_path = f'{results_path}/temp/preprocessed'
edv_esv_gif_path = f'{results_path}/temp/edv_esv'
edited_gif_path = f'{results_path}/temp/edited_edv_esv'
raw_curve_path = f'{results_path}/temp/raw_metrics.png'
edited_curve_path = f'{results_path}/temp/edited_metrics.png'
cache_dir = f'{root_path}/cache/'
final_dir = f'{results_path}/results.zip'

os.makedirs(f'{data_path}', exist_ok=True)
os.makedirs(f'{results_path}/temp', exist_ok=True)
os.makedirs(f'{results_path}/gifs', exist_ok=True)
os.makedirs(f'{results_path}/masks', exist_ok=True)
os.makedirs(f'{results_path}/edited_sax_df', exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

GIF_W = 150
DISPLAY_W = 400

N = 5
st.session_state.N = N

background_idx = 0
rv_idx = 3
lv_myo_idx = 2
lv_idx = 1
rv_myo_idx = 4

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

def zip_folder(folder_path, zip_path):
    folder_path = Path(folder_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in folder_path.rglob("*"):
            if file.is_file():
                zf.write(file, arcname=file.relative_to(folder_path))

def save_cached_mask(mask, save_path):
    np.save(save_path, mask)

def load_cached_mask(save_path):
    return np.load(save_path)

def save_config(config: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(config, f, indent=2)

def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)

def save_mask(mask, save_path):
    nib_mask = nib.Nifti1Image(mask, affine=np.eye(4), dtype='uint8')
    nib.save(nib_mask, save_path)

def save_image(image, save_path):
    nib_image = nib.Nifti1Image(image, affine=np.eye(4), dtype='float32')
    nib.save(nib_image, save_path)
    

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


def segmentation_view():
    
    st.header("Data Upload")

    if 'disable_upload' not in st.session_state:
        st.session_state['disable_upload'] = False
    
    uploaded_files = st.file_uploader(
        "Upload DICOM directory or ZIP of DICOM directory",
        type=["dcm"],#, "zip"],
        accept_multiple_files=True,
        disabled = st.session_state['disable_upload']
    )

    if uploaded_files:
        image, sax_df = Pipeline(uploaded_files)

        first_dicom = uploaded_files[0]
        first_dicom.seek(0)
        dcm = pydicom.dcmread(first_dicom)

        st.session_state.patient_name = str(dcm.PatientName)
        st.session_state.series_date = str(dcm.SeriesDate)
        st.session_state.series_description = str(dcm.SeriesDescription)
        st.session_state.pixelspacing = sax_df.pixelspacing.unique()[0]
        st.session_state.thickness = sax_df.thickness.unique()[0]
        st.session_state['sax_series_uid'] = dcm.SeriesInstanceUID

        n_slices = sax_df['slicelocation'].nunique()
        n_phases = sax_df.loc[sax_df['slicelocation'] == sax_df['slicelocation'].values[0]]['triggertime'].nunique()

        st.markdown(f"""
        ### DICOM Metadata

        **Patient Name:** {st.session_state.patient_name}  
        **Series Date:** {st.session_state.series_date}  
        **Series Description:** {st.session_state.series_description}  

        **Pixel Size:** {st.session_state.pixelspacing} x {st.session_state.pixelspacing} mm  
        **Slice Thickness:** {st.session_state.thickness} mm

        **Number of Images:** {len(sax_df)} 
        **Number of Slices:** {n_slices} 
        **Number of Phases:** {n_phases}
        **Slice x Phases = ** {n_slices * n_phases}
        
        """
        )

        if st.button("Confirm DICOMs"):
            st.session_state['disable_upload'] = True
            segment_image(image)
            st.success('Confirmed')
            initialize_app()





def segment_image(image):
    mask = np.zeros_like(image)
    save_image(image, save_path=f'{data_path}/image___{st.session_state.sax_series_uid}.nii.gz')
    save_mask(mask, save_path=f'{data_path}/masks___{st.session_state.sax_series_uid}.nii.gz')

# --------------------------------------------------------------
# Initialization
# --------------------------------------------------------------
def initialize_app():
    raw_image = load_nii(f'{data_path}/image___{st.session_state.sax_series_uid}.nii.gz')
    raw_mask = load_nii(f'{data_path}/masks___{st.session_state.sax_series_uid}.nii.gz').astype('uint8')

    raw_mask = np.eye(st.session_state.N, dtype=np.uint8)[raw_mask]
    raw_shape = raw_image.shape

    # -----------------------------
    # Compute raw indices
    # -----------------------------
    lv_volume = np.sum(raw_mask[...,lv_idx], axis=(0,1,2))
    rv_volume = np.sum(raw_mask[...,rv_idx], axis=(0,1,2))

    if np.max(lv_volume) == 0:
        raw_lv_dia_idx = 0
        raw_lv_sys_idx = 15

        raw_rv_dia_idx = 0
        raw_rv_sys_idx = 15

    else:
        raw_lv_dia_idx = int(np.argmax(lv_volume))
        raw_lv_sys_idx = np.where(lv_volume != 0)[0][np.argmin(lv_volume[lv_volume != 0])]

        raw_rv_dia_idx = int(np.argmax(rv_volume))
        raw_rv_sys_idx = np.where(rv_volume != 0)[0][np.argmin(rv_volume[rv_volume != 0])]

    st.session_state.raw = {
        "image": raw_image,
        "mask": raw_mask,
        "shape": raw_shape,
        "raw_lv_dia_idx": raw_lv_dia_idx,
        "raw_lv_sys_idx": raw_lv_sys_idx,
        "raw_rv_dia_idx":raw_rv_dia_idx,
        "raw_rv_sys_idx":raw_rv_sys_idx
    }

    # -----------------------------
    # Initialize EDV/ESV selection
    # -----------------------------
    if "edv_esv_selected" not in st.session_state:
        st.session_state['edv_esv_selected'] = {"lv_dia_idx": None, "lv_sys_idx": None,"rv_dia_idx": None, "rv_sys_idx": None, "confirmed": False}

    # -----------------------------
    # Preprocess / crop if required
    # -----------------------------
    mask_channels = [i for i in range(st.session_state.N) if i != background_idx]

    x_min, y_min, x_max, y_max = find_crop_box(np.max(raw_mask[...,mask_channels], axis=(-1,-2,-3)), crop_factor=1.5)
    st.session_state['subpixel_resolution'] = 2

    preprocessed_image = raw_image[y_min:y_max, x_min:x_max, :, :]
    preprocessed_mask = raw_mask[y_min:y_max, x_min:x_max, :, :, :].astype('uint8')
    H, W, D, T, N = preprocessed_mask.shape

    has_masks = np.where(np.sum(preprocessed_mask[...,mask_channels], axis = (0,1,3,-1))>0)[0]
    if len(has_masks) == 0:
        has_masks = np.array([1,2,3,4,5,6])

    mid_slice = len(has_masks)//2
    

    zoom = [st.session_state['subpixel_resolution'],st.session_state['subpixel_resolution'],1,1]
    smoothed_image = cv_zoom(preprocessed_image, zoom = zoom)


    st.session_state['cache_config_path'] = f"{cache_dir}/config___{st.session_state.sax_series_uid}.json"
    st.session_state['cache_mask_path'] = f"{cache_dir}/masks___{st.session_state.sax_series_uid}.npy"

    if os.path.exists(st.session_state['cache_config_path']) and os.path.exists(st.session_state['cache_mask_path']):
        smoothed_mask = load_cached_mask(st.session_state['cache_mask_path']).astype("uint8")
        cached = True
    else:
        smoothed_mask = cv_zoom_mask(
            preprocessed_mask,
            zoom=zoom + [1],
            interpolation=cv2.INTER_NEAREST,
        )
        cached = False

    make_video(smoothed_image[:,:,has_masks[mid_slice-3:mid_slice+3],:], smoothed_mask[:,:,has_masks[mid_slice-3:mid_slice+3],:, :] * 0, save_file=edv_esv_gif_path)
    make_video(smoothed_image, smoothed_mask*0, save_file=blank_gif_path)


    gif = Image.open(f'{edv_esv_gif_path}.gif')

    st.session_state.preprocessed = {
        "image": preprocessed_image,
        "mask": preprocessed_mask,
        "smooth_image": smoothed_image,
        "smooth_mask": smoothed_mask,
        "H": H,
        "W": W,
        "D": D,
        "T": T,
        "N": N,
        "edv_esv_frames": [frame.copy() for frame in ImageSequence.Iterator(gif)],
        "crop_box": [x_min, y_min, x_max, y_max],
    }


    st.session_state[f'edited_mask_lv'] = np.zeros_like(st.session_state.preprocessed["smooth_mask"])
    st.session_state[f'edited_mask_rv'] = np.zeros_like(st.session_state.preprocessed["smooth_mask"])

    if cached:
        config = load_config(st.session_state['cache_config_path'])
        confirm_selection(lv_dia_idx=config['lv_dia_idx'], 
                          rv_dia_idx=config['rv_dia_idx'], 
                          lv_sys_idx=config['lv_sys_idx'], 
                          rv_sys_idx=config['rv_sys_idx'])

    # -----------------------------
    # Initialize edited mask
    # -----------------------------

    st.session_state[f'mask_hash_lv'] = mask_hash(st.session_state.preprocessed["smooth_mask"])
    st.session_state[f'mask_hash_rv'] = mask_hash(st.session_state.preprocessed["smooth_mask"])
    st.session_state['lv_frames'] = None
    st.session_state['rv_frames'] = None
    st.session_state["view_mode"] = 'Static'
    st.session_state["brush_mode"] = "Paint ✏️"
    st.session_state["stroke_width"] = "thin"
    st.session_state['edit_made'] = False
    st.session_state['cached'] = cached
    st.session_state["saved"] = False

    st.session_state.initialized_all = True

def merge_masks(lv_mask, rv_mask):
    combined_mask = lv_mask + rv_mask
    combined_mask = np.argmax(combined_mask, -1)
    combined_mask = np.eye(N, dtype=np.uint8)[combined_mask]
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

    return np.eye(N, dtype=np.uint8)[labels]

def format_delta(value, raw_value, suffix="", round_digits=None):
    if round_digits is not None:
        value = round(value, round_digits)
        raw_value = round(raw_value, round_digits)
    return None if value == raw_value else f"{value - raw_value:.1f}{suffix}"



def mask_hash(mask_array):
    return hashlib.md5(mask_array.tobytes()).hexdigest()


def load_nii(nii_path):
    file = nib.load(nii_path)
    data = file.get_fdata(caching='unchanged')
    return data

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
        
        # Apply minor Gaussian blur and re-threshold to smooth edges
        # blurred = gaussian_filter(filled.astype(float), sigma=0.5)
        # smoothed = blurred > 0.48  # Convert back to binary
        return filled.astype('uint8')
    else:
        # For strokes without rings, apply very mild smoothing
        # blurred = gaussian_filter(strokes.astype(float), sigma=0.5)
        # smoothed = blurred > 0.48
        return strokes.astype('uint8')


def make_video(image, mask, save_file, ventricle = 'all', mask_frames = 'all',scale=1):
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
    img_min, img_max = np.min(image), np.max(image)

    try:
        font = load_font(int(18 * scale))
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

            img_slice = ((image[:,:,idx,t] - img_min) / (img_max - img_min + 1e-9) * 255).astype(np.uint8)
            img_rgb = np.stack([img_slice]*3, axis=-1)
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


def _label_vline(ax, x, color, y_pad=0.02):
    y0, y1 = ax.get_ylim()
    y = y0 + (y1 - y0) * y_pad
    ax.text(
        x + 0.5,
        y,
        f"{x}",
        color=color,
        fontsize=10,
        ha="center",
        va="bottom",
        rotation=90,
        alpha = 0.75
    )


def plot_volume_mass_curve(
    raw_volume,
    raw_masses,
    edited_volume,
    edited_masses,
    raw_dia_idx,
    raw_sys_idx,
    dia_idx,
    sys_idx,
    save_path,
):
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.25), sharex=True)

    frames_raw = np.arange(len(raw_volume))
    frames_edit = np.arange(len(edited_volume))

    edv = edited_volume[dia_idx]
    esv = edited_volume[sys_idx]
    dia_mass = edited_masses[dia_idx]

    raw_color = "#CBCBCB"
    vol_color = "#f66161"
    mass_color = "#499bed"

    axes[0].plot(frames_raw, raw_volume, color=raw_color, linewidth=2, alpha=0.7)
    axes[0].plot(
        frames_edit,
        edited_volume,
        color=vol_color,
        linewidth=2,
        label=f"EDV: {edv:.1f} mL | ESV: {esv:.1f} mL",
    )
    axes[0].set_xticks(np.arange(len(edited_volume)))


    axes[0].axvline(raw_dia_idx, color=raw_color, linestyle="--", linewidth=1.5, alpha=0.75)
    axes[0].axvline(raw_sys_idx, color=raw_color, linestyle=":", linewidth=1.5, alpha=0.75)
    axes[0].axvline(dia_idx, color=vol_color, linestyle="--", linewidth=1.5, alpha=0.75)
    axes[0].axvline(sys_idx, color=vol_color, linestyle=":", linewidth=1.5, alpha=0.75)

    _label_vline(axes[0], raw_dia_idx, raw_color)
    _label_vline(axes[0], raw_sys_idx, raw_color)
    _label_vline(axes[0], dia_idx, vol_color)
    _label_vline(axes[0], sys_idx, vol_color)

    axes[0].set_ylabel("Volume (mL)")
    axes[0].set_xlim(0, len(edited_volume) - 1)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1), edgecolor="none")

    axes[1].plot(frames_raw, raw_masses, color=raw_color, linewidth=2, alpha=0.7)
    axes[1].plot(
        frames_edit,
        edited_masses,
        color=mass_color,
        linewidth=2,
        label=f"Mass: {dia_mass:.1f} g",
    )

    axes[1].axvline(raw_dia_idx, color=raw_color, linestyle="--", linewidth=1.5, alpha=0.75)
    axes[1].axvline(dia_idx, color=mass_color, linestyle="--", linewidth=1.5, alpha=0.75)
    axes[1].set_xticks(np.arange(len(edited_volume)))

    _label_vline(axes[1], raw_dia_idx, raw_color)
    _label_vline(axes[1], dia_idx, mass_color)

    axes[1].set_xlabel("Frames")
    axes[1].set_ylabel("Mass (g)")
    axes[1].set_xlim(0, len(edited_volume) - 1)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1), edgecolor="none")

    plt.subplots_adjust(hspace=0.05, top=1, bottom=0)
    plt.savefig(save_path, bbox_inches="tight", dpi = 400)
    plt.close(fig)

def plot_volume_curve(
    raw_volume,
    edited_volume,
    raw_dia_idx,
    raw_sys_idx,
    dia_idx,
    sys_idx,
    save_path,
):

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    frames_raw = np.arange(len(raw_volume))
    frames_edit = np.arange(len(edited_volume))

    edv = edited_volume[dia_idx]
    esv = edited_volume[sys_idx]

    raw_color = "#CBCBCB"
    vol_color = "#f66161"

    ax.plot(frames_raw, raw_volume, color=raw_color, linewidth=2, alpha=0.7)
    ax.plot(
        frames_edit,
        edited_volume,
        color=vol_color,
        linewidth=2,
        label=f"EDV: {edv:.1f} mL | ESV: {esv:.1f} mL",
    )

    ax.axvline(raw_dia_idx, color=raw_color, linestyle="--", linewidth=1.5, alpha=0.75)
    ax.axvline(raw_sys_idx, color=raw_color, linestyle=":", linewidth=1.5, alpha=0.75)
    ax.axvline(dia_idx, color=vol_color, linestyle="--", linewidth=1.5, alpha=0.75)
    ax.axvline(sys_idx, color=vol_color, linestyle=":", linewidth=1.5, alpha=0.75)

    _label_vline(ax, raw_dia_idx, raw_color)
    _label_vline(ax, raw_sys_idx, raw_color)
    _label_vline(ax, dia_idx, vol_color)
    _label_vline(ax, sys_idx, vol_color)

    ax.set_xlabel("Frames")
    ax.set_ylabel("Volume (mL)")
    ax.set_xticks(np.arange(len(edited_volume)))
    ax.set_xlim(0, len(edited_volume) - 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1), edgecolor="none")

    plt.savefig(save_path, bbox_inches="tight", dpi=400)
    plt.close(fig)


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
    smooth_mask = st.session_state.preprocessed["smooth_mask"]

    # Loop over frames and channels to ensure proper assignment
    for f in frames:
        for c in channels:
            mask[:, :, :, f, c] = smooth_mask[:, :, :, f, c]

def confirm_selection(lv_dia_idx, lv_sys_idx,rv_dia_idx, rv_sys_idx):
    """Store confirmed EDV/ESV indices in session state."""
    st.session_state['edv_esv_selected'].update({
        "lv_dia_idx": lv_dia_idx,
        "lv_sys_idx": lv_sys_idx,
        "rv_dia_idx": rv_dia_idx,
        "rv_sys_idx": rv_sys_idx,
        "confirmed": True
    })

    save_config(st.session_state.edv_esv_selected, st.session_state['cache_config_path'])

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


def edv_esv_view():
    """Full EDV/ESV Finder view layout."""
    if "edv_esv_selected" not in st.session_state:
        st.session_state['edv_esv_selected'] = {"lv_dia_idx": None, "lv_sys_idx": None, "rv_dia_idx": None, "rv_sys_idx": None,"confirmed": False}
    
    H, W, D, T, N = [st.session_state.preprocessed[k] for k in ["H","W","D","T","N"]]
    edv_esv_frames= st.session_state.preprocessed['edv_esv_frames']


    if st.session_state.edv_esv_selected['confirmed']:
        display_lv_dia_idx=st.session_state.edv_esv_selected['lv_dia_idx']
        display_rv_dia_idx=st.session_state.edv_esv_selected['rv_dia_idx']
        display_lv_sys_idx=st.session_state.edv_esv_selected['lv_sys_idx']
        display_rv_sys_idx=st.session_state.edv_esv_selected['rv_sys_idx']
    else:
        display_lv_dia_idx=st.session_state.raw['raw_lv_dia_idx']
        display_rv_dia_idx=st.session_state.raw['raw_rv_dia_idx'] 
        display_lv_sys_idx=st.session_state.raw['raw_lv_sys_idx'] 
        display_rv_sys_idx=st.session_state.raw['raw_rv_sys_idx'] 

    disabled_flag = st.session_state['edv_esv_selected']["confirmed"]

    col_lv, col_rv = st.columns(2)

    with col_lv:
        st.markdown('#### Left Ventricle')
        col_edv, col_esv = st.columns(2)

        with col_edv:
            lv_dia_idx = frame_index_slider(T, edv_esv_frames, display_lv_dia_idx, 'LV End-Diastolic Index', disabled_flag, key = 'lv_edv')

        with col_esv:
            lv_sys_idx = frame_index_slider(T, edv_esv_frames, display_lv_sys_idx, 'LV End-Systolic Index',disabled_flag, key = 'lv_esv')

    with col_rv:
        st.markdown('#### Right Ventricle')
        col_edv, col_esv = st.columns(2)
        with col_edv:
            rv_dia_idx = frame_index_slider(T, edv_esv_frames, display_rv_dia_idx, 'RV End-Diastolic Index', disabled_flag, key = 'rv_edv')

        with col_esv:
            rv_sys_idx = frame_index_slider(T, edv_esv_frames, display_rv_sys_idx, 'RV End-Systolic Index',disabled_flag, key = 'rv_esv')


    st.write('')
    if not disabled_flag:
        st.button(
            "Confirm EDV | ESV",
            on_click=lambda: confirm_selection(lv_dia_idx, lv_sys_idx, rv_dia_idx, rv_sys_idx),
            type="primary",
            use_container_width=True
        )


    else:
        st.success("EDV | ESV Confirmed!")


def slice_navigation(D):
    if "slice_idx" not in st.session_state:
        st.session_state.slice_idx = 0
    if "previous_slice_idx" not in st.session_state:
        st.session_state.previous_slice_idx = st.session_state.slice_idx

    # Store previous slice
    previous_d = st.session_state.previous_slice_idx

    # Slider (updates slice_idx immediately)
    st.slider(
        "Slice Index",
        0,
        D - 1,
        key="slice_idx",
    )

    col_prev, col_next = st.columns(2)
    with col_prev:
        st.button(
            "Previous",
            on_click=lambda: st.session_state.update(
                slice_idx=max(0, st.session_state.slice_idx - 1)
            ),
            use_container_width=True,
        )
    with col_next:
        st.button(
            "Next",
            on_click=lambda: st.session_state.update(
                slice_idx=min(D - 1, st.session_state.slice_idx + 1)
            ),
            use_container_width=True,
        )

    # Determine if canvas needs reset
    previous_objects = st.session_state.get('canvas', {}).get('previous_objects', [])
    reset_canvas = previous_d != st.session_state.slice_idx and bool(previous_objects)

    # Update previous slice for next rerun
    st.session_state.previous_slice_idx = st.session_state.slice_idx

    return st.session_state.slice_idx, reset_canvas



def get_overlay(image_slice, mask_state, H, W, N, OVERLAY_COLORS, ventricle):
    if ventricle == 'rv':
        channels = [rv_idx, rv_myo_idx]
    elif ventricle == 'lv':
        channels = [lv_idx, lv_myo_idx]
    else:
        channels = np.arange(N)

    overlay = Image.fromarray(np.stack([image_slice]*3, axis=-1)).convert("RGBA")
    for i in channels:
        ch_mask = mask_state[:, :, i]
        if np.any(ch_mask):
            mask_img = np.zeros((H*st.session_state['subpixel_resolution'], W*st.session_state['subpixel_resolution'], 4), dtype=np.uint8)
            mask_img[ch_mask > 0] = OVERLAY_COLORS[i]
            overlay = Image.alpha_composite(overlay, Image.fromarray(mask_img))
    return overlay



def select_brush(N, ventricle):
    """Brush selection UI for channel, action, and stroke width."""
    action = st.radio("Brush Stroke Selection", 
                      options=["Paint ✏️", "Erase ✂️"],  
                      index=["Paint ✏️", "Erase ✂️"].index(st.session_state.brush_mode),
                      horizontal=True)
    st.session_state['brush_mode'] = action
    
    stroke_width_map = {"thin":6,"medium":20,"thick":40}
    stroke_width_sel = st.radio("Stroke Width", 
                                options=list(stroke_width_map.keys()),  
                                index= list(stroke_width_map.keys()).index(st.session_state["stroke_width"]), 
                                horizontal=True)
    st.session_state['stroke_width'] = stroke_width_sel
    if ventricle == 'lv':
        valid_channels = [lv_myo_idx, lv_idx]
    elif ventricle == 'rv':
        valid_channels = [rv_myo_idx, rv_idx]
    else:
        valid_channels = [i for i in range(N) if i != background_idx]

    if action == "Paint ✏️":
        channel = st.radio(
            "Mask",
            options=valid_channels,
            format_func=lambda x: BRUSH_LABELS[x],
            index=0,
            horizontal=True
        )
    else:
        channel = 0
    stroke_width = stroke_width_map[stroke_width_sel]
    return channel, action, stroke_width

def normalize(image):
    image = (image - np.min(image))/(np.max(image) - np.min(image))
    return image


def mask_editor_view():
    """Full Mask Editor layout."""
    if not st.session_state['edv_esv_selected']["confirmed"]:
        st.error("Select and confirm EDV/ESV first.")
        st.stop()

    col1, col2, col3 = st.columns([1,1.5,1.5])

    H, W, D, T, N = [st.session_state.preprocessed[k] for k in ["H","W","D","T","N"]]
    image=st.session_state.preprocessed["smooth_image"]

    with col1:
        ventricle_label = st.radio("Ventricle", options=["Left Ventricle","Right Ventricle"],  index = 0, horizontal=True)
        ventricle = 'lv' if 'left' in ventricle_label.lower() else 'rv'
        channel, action, stroke_width = select_brush(N, ventricle)

        st.caption('Image Selection')
        idx_label = st.radio("Frame", options=["End-Diastole","End-Systole"],  index = 0, horizontal=True)
        d, reset_canvas = slice_navigation(D)


        edited_mask=st.session_state[f'edited_mask_{ventricle}']
        dia_idx=st.session_state.edv_esv_selected[f"{ventricle}_dia_idx"]
        sys_idx=st.session_state.edv_esv_selected[f"{ventricle}_sys_idx"]


    idx = dia_idx if idx_label=="End-Diastole" else sys_idx
    image_slice = image[:,:,d,idx]
    image_slice = (normalize(image_slice) * 255).astype(np.uint8)
    mask_slice = edited_mask[:,:,d,idx,:]

    with col2:
        edit_mode = st.radio('Segmentation Editor',['Editor','Viewer'], index=0, horizontal=True)
        stroke_color = f"rgba{OVERLAY_COLORS[background_idx][:3]+(0.7,)}" if action == "Erase ✂️" else f"rgba{OVERLAY_COLORS[channel][:3]+(0.65,)}"
        if edit_mode == 'Viewer':
            st.image(image_slice, width=DISPLAY_W)
        else:
            # Initialize canvas state
            if 'canvas' not in st.session_state:
                st.session_state['canvas'] = {
                    'canvas_key': f'editor_{d}',
                    'previous_d': d,
                    'previous_objects': []
                }

                        
            if reset_canvas:
                st.session_state['canvas']['canvas_key'] = f'editor_{d}'
                st.session_state['canvas']['previous_objects'] = []

            st.session_state['canvas']['previous_d'] = d

            canvas_result = st_canvas(
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=get_overlay(image_slice, mask_slice, H, W, N, OVERLAY_COLORS, ventricle),
                update_streamlit=True,
                height = H*DISPLAY_W/W,
                width=DISPLAY_W,
                drawing_mode='freedraw',
                key=st.session_state['canvas']['canvas_key']+ ventricle
            )


            # Track current objects
            current_objects = []
            if canvas_result and canvas_result.json_data:
                current_objects = canvas_result.json_data.get("objects", [])
            st.session_state['canvas']['previous_objects'] = current_objects

            # Save / clear buttons (trigger rerun only here)
            col_save, col_clear = st.columns([1, 0.3])
            edited_mask = st.session_state[f'edited_mask_{ventricle}']
            
            with col_save:
                save_contour = st.button('Save Contour', type='primary', use_container_width=True)
                if save_contour and canvas_result and canvas_result.image_data is not None and current_objects:
                    brush_data = np.array(canvas_result.image_data)
                    rgb = brush_data[:, :, :3].astype(np.float32)
                    alpha = brush_data[:, :, 3].astype(np.float32) / 255.0

                    overlay_colors_list = np.array([color[:3] for color in OVERLAY_COLORS.values()], dtype=np.float32)
                    overlay_channels = list(OVERLAY_COLORS.keys())

                    h, w, _ = rgb.shape
                    rgb_flat = rgb.reshape(-1, 3)
                    alpha_flat = alpha.flatten()
                    distances = np.linalg.norm(rgb_flat[:, None, :] - overlay_colors_list[None, :, :], axis=-1)
                    closest_idx = np.argmin(distances, axis=1)

                    mask_flat = np.zeros((h*w, len(overlay_channels)), dtype=np.uint8)
                    for idx_color, ch in enumerate(overlay_channels):
                        mask_flat[:, idx_color] = ((closest_idx == idx_color) & (alpha_flat > 0)).astype(np.uint8)

                    masks = []
                    for idx_color, ch in enumerate(overlay_channels):
                        mask_bool = mask_flat[:, idx_color].reshape(h, w)
                        mask_bool = thicken_close_fill_and_smooth(mask_bool, stroke_width)
                        masks.append(mask_bool)

                    combined_mask = np.stack(masks, axis=-1)
                    for idx_color, ch in enumerate(overlay_channels):
                        resized_mask = np.array(
                            Image.fromarray(combined_mask[:, :, idx_color]).resize(
                                (W*st.session_state['subpixel_resolution'], H*st.session_state['subpixel_resolution']),
                                resample=Image.NEAREST
                            )
                        )
                        edited_mask[:, :, d, idx, :][resized_mask > 0] = 0
                        edited_mask[:, :, d, idx, ch][resized_mask > 0] = 1

                    st.session_state['edit_made'] = True
                    combined_mask = merge_masks(st.session_state[f'edited_mask_lv'] , st.session_state[f'edited_mask_rv'])
                    save_cached_mask(combined_mask, save_path=st.session_state['cache_mask_path'])
                    st.rerun()

            with col_clear:
                if st.button('Clear Slice', use_container_width=True):
                    edited_mask[:, :, d, idx, :] = 0
                    combined_mask = merge_masks(st.session_state[f'edited_mask_lv'] , st.session_state[f'edited_mask_rv'])
                    save_cached_mask(combined_mask, save_path=st.session_state['cache_mask_path'])

                    st.session_state['edit_made'] = True
                    st.rerun()

            st.session_state[f'edited_mask_{ventricle}'] = edited_mask
            


    # ---------- right column preview ----------
    with col3:
        view_mode = st.radio(
            "Corrected Mask",
            ["Static", "Viewer"],
            index=0,
            horizontal=True,
        )

        if st.session_state[f'{ventricle}_frames'] is None or st.session_state['edit_made']:
            make_video(
                image,
                st.session_state[f'edited_mask_{ventricle}'],
                save_file=f'{edited_gif_path}_{ventricle}',
                mask_frames = [dia_idx, sys_idx],
                ventricle = ventricle
            )

            gif = Image.open(f'{edited_gif_path}_{ventricle}.gif')
            st.session_state[f'{ventricle}_frames'] = [frame.copy() for frame in ImageSequence.Iterator(gif)]
            st.session_state['edit_made'] = False

        if view_mode == "Static":
            view_image = st.session_state[f'{ventricle}_frames'][0 if idx_label == "End-Diastole" else 1]
            width = int(DISPLAY_W * 1.5)
        elif view_mode == "Viewer":
            view_image = image_slice
            width = int(DISPLAY_W)

        st.image(view_image, width = width)

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


def final_result_view():
    raw = st.session_state.raw
    preprocessed = st.session_state.preprocessed
    pixelspacing = st.session_state.pixelspacing
    thickness = st.session_state.thickness

    raw_image = raw["image"]
    raw_mask = raw["mask"]
    preprocessed_image = preprocessed["image"]

    H, W, D, T, N = [preprocessed[k] for k in ["H","W","D","T","N"]]

    crop_box = preprocessed['crop_box']
    
    if not st.session_state.edv_esv_selected["confirmed"]:
        st.error("Select and confirm EDV/ESV first.")
        st.stop()

    raw_lv_dia_idx = raw["raw_lv_dia_idx"]
    raw_lv_sys_idx = raw["raw_lv_sys_idx"]
    raw_rv_dia_idx = raw["raw_rv_dia_idx"]
    raw_rv_sys_idx = raw["raw_rv_sys_idx"]

    lv_dia_idx = st.session_state['edv_esv_selected']["lv_dia_idx"]
    lv_sys_idx = st.session_state['edv_esv_selected']["lv_sys_idx"]
    rv_dia_idx = st.session_state['edv_esv_selected']["rv_dia_idx"]
    rv_sys_idx = st.session_state['edv_esv_selected']["rv_sys_idx"]
    sax_series_uid = st.session_state.sax_series_uid

    final_lv_gif_path = f"{results_path}/gifs/{sax_series_uid}_lv.gif"
    final_rv_gif_path = f"{results_path}/gifs/{sax_series_uid}_rv.gif"

    lv_mask = cv_zoom(st.session_state['edited_mask_lv'], zoom = [1/st.session_state['subpixel_resolution'],1/st.session_state['subpixel_resolution'],1,1])
    rv_mask = cv_zoom(st.session_state['edited_mask_rv'], zoom = [1/st.session_state['subpixel_resolution'],1/st.session_state['subpixel_resolution'],1,1])

    combined_mask = merge_masks(lv_mask, rv_mask)


    # Calculate LV metrics
    lv_volume, lv_masses, lv_edv, lv_esv, lv_sv, lv_ef, lv_mass = calculate_sax_metrics(
        mask=combined_mask,
        blood_pool_idx=lv_idx,
        myo_idx=lv_myo_idx,
        dia_idx=lv_dia_idx,
        sys_idx=lv_sys_idx
    )
    raw_lv_volume, raw_lv_masses, raw_lv_edv, raw_lv_esv, raw_lv_sv, raw_lv_ef, raw_lv_mass = calculate_sax_metrics(
        mask=raw_mask,
        blood_pool_idx=lv_idx,
        myo_idx=lv_myo_idx,
        dia_idx=raw_lv_dia_idx,
        sys_idx=raw_lv_sys_idx
    )

    # Calculate RV metrics
    rv_volume, rv_masses, rv_edv, rv_esv, rv_sv, rv_ef, rv_mass = calculate_sax_metrics(
        mask=combined_mask,
        blood_pool_idx=rv_idx,
        myo_idx=rv_myo_idx,
        dia_idx=rv_dia_idx,
        sys_idx=rv_sys_idx
    )
    raw_rv_volume, raw_rv_masses, raw_rv_edv, raw_rv_esv, raw_rv_sv, raw_rv_ef, raw_rv_mass = calculate_sax_metrics(
        mask=raw_mask,
        blood_pool_idx=rv_idx,
        myo_idx=rv_myo_idx,
        dia_idx=raw_rv_dia_idx,
        sys_idx=raw_rv_sys_idx
    )


    x_min, y_min, x_max, y_max = crop_box
    final_mask_2d = np.zeros_like(raw_mask)
    final_mask_2d[y_min:y_max, x_min:x_max, :, :, :] = combined_mask
    final_mask_2d = np.argmax(final_mask_2d, axis=-1)


    make_video(preprocessed_image, 
               final_mask_2d[y_min:y_max,x_min:x_max,:,:], 
               save_file=final_lv_gif_path, 
               mask_frames=[lv_dia_idx,lv_sys_idx],
               ventricle='all')
    
    make_video(preprocessed_image, 
               final_mask_2d[y_min:y_max,x_min:x_max,:,:], 
               save_file=final_rv_gif_path, 
               mask_frames=[rv_dia_idx,rv_sys_idx],
               ventricle='all')
    

    col_lv, _, col_rv = st.columns([1,0.05,1])
    with col_lv:
        st.markdown('#### Left Ventricle')

        col1, col2 = st.columns([0.3,0.7])
        with col1:
            st.caption("LV Metrics")
            st.metric("EDV", f"{lv_edv:.1f}mL", delta=format_delta(lv_edv, raw_lv_edv, "mL"))
            st.metric("ESV", f"{lv_esv:.1f}mL", delta=format_delta(lv_esv, raw_lv_esv, "mL"))
            st.metric("EF", f"{lv_ef:.1f}%", delta=format_delta(lv_ef, raw_lv_ef, "%", round_digits=1))
            st.metric("Mass", f"{lv_mass:.1f}g", delta=format_delta(lv_mass, raw_lv_mass, "g"))

        with col2:
            st.caption("Final LV Mask")
            st.image(final_lv_gif_path)
        
    with col_rv:
        st.markdown('#### Right Ventricle')

        col1, col2 = st.columns([0.3,0.7])
        with col1:
            st.caption("RV Metrics")
            st.metric("EDV", f"{rv_edv:.1f}mL", delta=format_delta(rv_edv, raw_rv_edv, "mL"))
            st.metric("ESV", f"{rv_esv:.1f}mL", delta=format_delta(rv_esv, raw_rv_esv, "mL"))
            st.metric("EF", f"{rv_ef:.1f}%", delta=format_delta(rv_ef, raw_rv_ef, "%", round_digits=1))
            st.metric("Mass", f"{rv_mass:.1f}g", delta=format_delta(rv_mass, raw_rv_mass, "g"))


        with col2:
            st.caption("Final RV Mask")
            st.image(final_rv_gif_path)

    
    if st.button('Save Masks and Metrics 💾', type='primary', use_container_width=True):
        st.success('Masks and Metrics Saved!')

        # Save LV mask
        save_mask(final_mask_2d, f'{results_path}/masks/{sax_series_uid}.nii.gz')


        # Prepare LV metrics
        lv_df = pd.DataFrame({
            "sax_series_uid": [sax_series_uid],
            "chamber": ["LV"],
            "edv_frame": [lv_dia_idx],
            "esv_frame": [lv_sys_idx],
            "edv": [lv_edv],
            "esv": [lv_esv],
            "stroke_volume": [lv_sv],
            "ejection_fraction": [lv_ef],
            "mass": [lv_mass],
            "pixelspacing": [pixelspacing],
            "thickness": [thickness],
            "num_slices": [lv_mask.shape[2]],
            "num_frames": [lv_mask.shape[3]],
        })

        # Prepare RV metrics
        rv_df = pd.DataFrame({
            "sax_series_uid": [sax_series_uid],
            "chamber": ["RV"],
            "edv_frame": [rv_dia_idx],
            "esv_frame": [rv_sys_idx],
            "edv": [rv_edv],
            "esv": [rv_esv],
            "stroke_volume": [rv_sv],
            "ejection_fraction": [rv_ef],
            "mass": [rv_mass],
            "pixelspacing": [pixelspacing],
            "thickness": [thickness],
            "num_slices": [rv_mask.shape[2]],
            "num_frames": [rv_mask.shape[3]],
        })

        # Combine LV and RV metrics
        combined_df = pd.concat([lv_df, rv_df], ignore_index=True)
        combined_df.to_csv(f'{results_path}/edited_sax_df/{sax_series_uid}.csv', index=False)
        st.session_state["saved"] = True
