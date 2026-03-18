FROM tensorflow/tensorflow:2.9.1-gpu

RUN apt-get update && apt-get install -y libopenexr-dev
RUN pip install pydicom h5py \
scikit-learn scikit-image matplotlib \
tensorflow-addons nibabel \
typing_extensions --upgrade

RUN apt-get update && apt-get install -y libopenexr-dev

RUN pip install --no-cache-dir \
    pydicom==2.2.2 \
    pylibjpeg==1.4.0 \
    pylibjpeg-libjpeg==1.3.4 \
    pylibjpeg-openjpeg==1.3.2 \
    streamlit==1.40.1 \
    stqdm \
    opencv-python-headless \
    streamlit_drawable_canvas

RUN pip uninstall protobuf -y
RUN pip install protobuf==3.20.3