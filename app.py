import streamlit as st
import pickle
from PIL import Image
import numpy as np
import io
import cv2 


#aading innomatics logo
st.image(r"E:\innomatics\logo.png", width=200)

# Load the logistic regression model
model_path = "E:\innomatics\ml\Dog Breed Image  classification\dt_model.plk"

model = pickle.load(open(model_path,"rb"))

def predict(image):
    
    arr = np.array(image,dtype = np.uint8)
    resized = cv2.resize(arr,(200,200))
    data = resized.flatten()
    
    predictio = model.predict([data])
    return predictio
  

image_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
  
if image_file is not None:
    # Display the uploaded image
    image = Image.open(image_file)
    image = image.convert('RGB')
    st.image(image, caption='Uploaded Image', width=500)
    
    # Convert PIL image to OpenCV format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Make a prediction
    prediction = predict(image)[0]

if st.button('Submit'):
    st.write(f'The Dog Breed is : {prediction}')










