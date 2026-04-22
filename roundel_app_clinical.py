# --------------------------------------------------------------
# Configure Streamlit page
# --------------------------------------------------------------
from roundel_utils import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
st.set_page_config(page_title="Roundel", page_icon="⭕️", layout='wide')

# --------------------------------------------------------------
# App
# --------------------------------------------------------------
st.write('# Roundel')

view = st.segmented_control(
    "Tab",
    options=["Segmentation ⭕", "Preview Segmentation 👁️", "Corrector Model 🔧", "EDV/ESV Finder 🔍", "EDV/ESV Mask Editor 📝", "Final Result ✅"],
    default = "Segmentation ⭕",
    label_visibility='hidden'
)
st.divider()

# --------------------------------------------------------------
# Initialize App
# --------------------------------------------------------------
if view == "Segmentation ⭕":
    segmentation_view()


# --------------------------------------------------------------
# Preview Segmentation
# --------------------------------------------------------------
if view == "Preview Segmentation 👁️":
    preview_segmentation_view()


# --------------------------------------------------------------
# Corrector Model
# --------------------------------------------------------------
if view == "Corrector Model 🔧":
    corrector_model_view()


# --------------------------------------------------------------
# EDV/ESV Finder
# --------------------------------------------------------------
if view == "EDV/ESV Finder 🔍":
    edv_esv_view()


# --------------------------------------------------------------
# Mask Editor 
# --------------------------------------------------------------

if view == "EDV/ESV Mask Editor 📝":
    mask_editor_view()


# --------------------------------------------------------------
# Final Result
# --------------------------------------------------------------
if view == "Final Result ✅":
    final_result_view()
    