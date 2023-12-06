import streamlit as st
import os
from PIL import Image
import tensorflow
from keras import models
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

model= load_model('model.h5')


st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://files.123freevectors.com/wp-content/original/202451-pastel-blue-plain-background-vector-eps.jpg")
    }
   .sidebar .sidebar-content {
        background: url("https://files.123freevectors.com/wp-content/original/202451-pastel-blue-plain-background-vector-eps.jpg")
    }
    
    </style>
    """,
    unsafe_allow_html=True
)





















st.title('Brain Tumor Detection')
st.subheader('Please upload your MRI')
uploaded_image = st.file_uploader('Choose an image')

def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

if uploaded_image is not None:
    # save the image in a directory
    if save_uploaded_image(uploaded_image):
        # load the image
        display_image = Image.open(uploaded_image)
        img=image.load_img(os.path.join('uploads',uploaded_image.name),target_size=(224,224))
        st.image(img)
        x=image.img_to_array(img)
        x=x/255
        x=np.expand_dims(x,axis=0)
        pred=model.predict(x)
        if pred[0][0] < 0.6:
            st.subheader('No Tumor Detected')
            st.subheader('Accuracy')
            st.subheader(pred[0][0])
        else:
            st.subheader('Tumor Detected')
            st.subheader('Accuracy')
            st.subheader(pred[0][0])
            

