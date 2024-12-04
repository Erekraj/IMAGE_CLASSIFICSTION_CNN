import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image


# Load CIFAR_10 Model
        
model = tf.keras.models.load_model("cifar10_classifier.h5")
# Funvtion for MobileNetV2 ImageNet model

def mobilenetv2_imagenet():
    st.title("Image Classification withNetV2")
    
    uploaded_file = st.file_uploader("Choose an image...",type=["jpg","jpeg","png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image,caption='Uploaded Image',use_column_width=True)
    
        st.write("Classifying...")
    
       # Load MobileV2Net Model
        model = tf.keras.application.MobileNetV2(weights='imagenet')
    
    
       # preprocessing the image
    
        img = image.resize((224,224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array,axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
        # Make Prediction
    
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
        for i, (imagenet_id,label,score) in enumerate(decoded_predictions):
            st.write(f"{label}: {score *100:.2f}%")
        
# Function for CIFAR-10 MODEL
        
def cifar10_classification():
    st.title("CIFAR-10 IMAGE Classification")
    
    uploaded_file = st.file_uploader("Choose an img...",type=["jpg","jpeg","png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Classifying...")
        
        
        
        # cifar-10 classses
        
        cifar10_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        
        # Preprocessing image
        
        img = image.resize((32,32)) 
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make Prediction
        
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
         
        
        st.write(f"Predicted Class: {cifar10_classes[predicted_class]}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
        
def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Choose Model", ("CIFAR-10","Mobilenetv2_imagenet"))
    
    
    
    if choice == "Mobilenetv2 (ImageNet)":
        mobilenetv2_imagenet()
    elif choice == "CIFAR-10":
        cifar10_classification()
        
if __name__ == "__main__":
    main()
                   
                          
                          
                           