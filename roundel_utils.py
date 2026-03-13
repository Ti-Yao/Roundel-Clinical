from utils import *

def segmentation_view():
    st.header("Data Upload")

    if 'disable_upload' not in st.session_state:
        st.session_state['disable_upload'] = False
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_files = st.file_uploader(
            "Upload DICOM directory or ZIP of DICOM directory",
            type=["dcm"],#, "zip"],
            accept_multiple_files=True,
            disabled = st.session_state['disable_upload']
        )

    with col2:
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
            # Create dataframe
            dicom_data = {
                "Field": [
                    "Patient Name",
                    "Series Date",
                    "Series Description",
                    "Pixel Size",
                    "Slice Thickness",
                    "Number of Images",
                    "Number of Slices",
                    "Number of Phases",
                    "Slice × Phases"
                ],
                "Value": [
                    st.session_state.patient_name,
                    st.session_state.series_date,
                    st.session_state.series_description,
                    f"{st.session_state.pixelspacing} x {st.session_state.pixelspacing} mm",
                    f"{st.session_state.thickness} mm",
                    len(sax_df),
                    n_slices,
                    n_phases,
                    n_slices * n_phases
                ]
            }

            df_dicom = pd.DataFrame(dicom_data).set_index('Field')

            # Display dataframe in Streamlit
            st.dataframe(df_dicom, use_container_width=True)

    if uploaded_files:
        # if st.button("Segment!", use_container_width=True, type = 'primary'):
        st.session_state['disable_upload'] = True
        if "initialized_all" not in st.session_state:
            with st.spinner("Segmenting..."):
                segment_image(image)
            
            with st.spinner("Initialising..."):
                initialize_app()
    
    if "initialized_all" in st.session_state:
        st.success('Segmentation Confirmed! ⭕️')


def segment_image(image):
    # Crop and pad the image to correct shape

    st.session_state['model'] = tf.keras.models.load_model('UNet3plus.h5', 
                                    compile = False,
                                    custom_objects={"InstanceNormalization":InstanceNormalization,
                                                    "ResizeAndConcatenate":ResizeAndConcatenate})

    target_shape = (256,256)
    mask = []
    for t in stqdm(range(image.shape[-1])):  
        image_cropped, meta =  crop_pad_image_only(image[..., t], target_shape = target_shape)
        X = z_normalise_image(image_cropped)[np.newaxis,...,np.newaxis]
        pred_mask = sliding_window_inference_3d(
                        st.session_state.model,
                        X,                   # np.ndarray [batch_size,Y,X,Z,1] already preprocessed & normalised
                        patch_size=[256,256,10],
                        overlap=0.5,
                        apply_softmax=False,       # set False as the model already uses softmax activation
                        out_channels=5,    # if known, can be set to avoid dry run
                        tta=False,                 # Whether to use test-time augmentation (flips)
                        plot_tta=False,           # Whether to plot the prediction after each TTA variant (for debugging)
                        scan_id = None,           # used for naming the TTA variant plots
                        time_step_counter=0, # used for naming the TTA variant plots
                        gaussian_sigma_scale=1/8, # controls how peaked the Gaussian is
                        deep_supervision=True,
                        run = None               # neptune run instance for logging
                    )
        pred_mask = reverse_crop_pad(pred_mask, meta)

        mask.append(pred_mask)
    mask = np.stack(mask, -1)
    save_image(image, save_path=f'{data_path}/image___{st.session_state.sax_series_uid}.nii.gz')
    save_mask(mask, save_path=f'{data_path}/masks___{st.session_state.sax_series_uid}.nii.gz')


# --------------------------------------------------------------
# Initialization
# --------------------------------------------------------------
def initialize_app():
    stages = 5
    with stqdm(total=stages) as pbar:
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

        pbar.update(1)

        has_masks = np.where(np.sum(preprocessed_mask[...,mask_channels], axis = (0,1,3,-1))>0)[0]
        if len(has_masks) == 0:
            has_masks = np.array([1,2,3,4,5,6])

        mid_slice = len(has_masks)//2
        

        zoom = [st.session_state['subpixel_resolution'],st.session_state['subpixel_resolution'],1,1]
        smoothed_image = cv_zoom(preprocessed_image, zoom = zoom)


        st.session_state['cache_config_path'] = f"{cache_dir}/config___{st.session_state.sax_series_uid}.json"
        st.session_state['cache_mask_path'] = f"{cache_dir}/masks___{st.session_state.sax_series_uid}.npy"

        pbar.update(1)

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
        pbar.update(1)
        make_video(smoothed_image, smoothed_mask*0, save_file=blank_gif_path)
        pbar.update(1)


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
        st.session_state['lv_frames'] = None
        st.session_state['rv_frames'] = None
        st.session_state["view_mode"] = 'Static'
        st.session_state["brush_mode"] = "Paint ✏️"
        st.session_state["stroke_width"] = "thin"
        st.session_state['edit_made'] = False
        st.session_state['cached'] = cached
        st.session_state["saved"] = False
        st.session_state.initialized_all = True



def edv_esv_view():
    """Full EDV/ESV Finder view layout."""
    if not st.session_state['initialized_all']:
        st.error("Select and confirm EDV/ESV first.")
        st.stop()

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
        st.success("EDV | ESV Confirmed! 🔍")



def slice_navigation(D):
    if "slice_idx" not in st.session_state:
        st.session_state.slice_idx = 0
    if "previous_slice_idx" not in st.session_state:
        st.session_state.previous_slice_idx = st.session_state.slice_idx

    # Store previous slice
    previous_d = st.session_state.previous_slice_idx

    # Slider (updates slice_idx immediately)
    st.slider("Slice Index", 0, D - 1,key="slice_idx")

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

    save_button = st.button('Save Masks and Metrics 💾', type='primary', use_container_width=True)


    if save_button:
        if st.session_state.get("saved", False):
            st.success('Masks and Metrics Overwritten! ✅')
        else:
            st.success('Masks and Metrics Saved! ✅')

        # Save LV mask
        save_mask(final_mask_2d, f'{results_path}/masks/{sax_series_uid}.nii.gz')

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

        combined_df = pd.concat([lv_df, rv_df], ignore_index=True)
        combined_df.to_csv(f'{results_path}/edited_sax_df/{sax_series_uid}.csv', index=False)

        st.session_state["saved"] = True
    
    elif st.session_state.get("saved", False):
        st.info('Masks and Metrics Previously Saved! ✅')

