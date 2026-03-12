# FROM tensorflow/tensorflow:2.9.1-gpu

FROM tensorflow/tensorflow:2.9.1-gpu

RUN apt-get update && apt-get install -y libopenexr-dev
RUN pip install tensorflow-mri tqdm pydicom h5py \
tensorflow-addons scikit-learn scikit-image \
neptune neptune_tensorflow_keras jupyter nibabel pydicom kaleido \
albumentations typing_extensions --upgrade

RUN apt-get update && apt-get install -y libopenexr-dev

RUN pip install --no-cache-dir \
    pydicom==2.2.2 \
    pylibjpeg==1.4.0 \
    pylibjpeg-libjpeg==1.3.4 \
    pylibjpeg-openjpeg==1.3.2 \
    streamlit==1.40.1 \
    stqdm \
    streamlit_drawable_canvas 

RUN pip uninstall protobuf -y
RUN pip install protobuf==3.20.3