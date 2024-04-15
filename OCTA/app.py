import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
model = load_model("OCTA_with_early_stopping.h5")
labels = ["DR", "Normal"]
st.title("DR Detection")
st.write("Upload an image to check for Diabetic Retinopathy")
file = st.file_uploader("Upload an OCTA scan image")
input_shape = (256, 256, 3)  
if file is not None:
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_shape[1], input_shape[0]))
    img_disp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
    img = np.expand_dims(img, axis=0) / 255.0
    pred = model.predict(img)[0][0]
    prob = round(pred * 100, 2)
    label = labels[int(round(pred))]
    st.subheader("Prediction:")
    st.write(f"The image is classified as: **{label}**")
    st.subheader("Original Image:")
    st.image(img_disp, channels="BGR", caption="Uploaded Image", use_column_width=True)
