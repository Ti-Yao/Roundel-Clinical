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
    options=["Segmentation ⭕","EDV/ESV Finder 🔍", "Mask Editor 📝", "Final Result ✅"],
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
# EDV/ESV Finder 
# --------------------------------------------------------------
if view == "EDV/ESV Finder 🔍":
    edv_esv_view()


# --------------------------------------------------------------
# Mask Editor 
# --------------------------------------------------------------

if view == "Mask Editor 📝":
    mask_editor_view()


# --------------------------------------------------------------
# Final Result
# --------------------------------------------------------------
if view == "Final Result ✅":
    final_result_view()
    