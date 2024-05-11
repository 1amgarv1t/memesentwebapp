import streamlit as st
import cv2
from PIL import Image
import easyocr
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import transformers
from transformers import TFRobertaModel, RobertaTokenizer
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from collections import defaultdict
import regex as re
from keras.preprocessing import image as keras_image

#tensorflow 2.13.0, transformers 4.35.0, nltk 3.8.1, easyocr, 1.7.1, streamlit 1.29.0, pillow 9.4.0


# Set the page configuration
st.set_page_config(
    page_title="Image OCR and Prediction",
    layout="wide",  # Use a wide layout for more space
    initial_sidebar_state="expanded",  # Open sidebar by default
)


# Header with some styling
st.title("📷 Image OCR and Prediction")

# Introduction
st.markdown(
    """
    This application extracts text from images using Optical Character Recognition (OCR) and makes predictions based on text and image models.
    """
)

# File uploader for images
st.header("Step 1: Upload an Image")
uploaded_file = st.file_uploader("Upload your image here", type=["png", "jpg", "jpeg"])

# Check if an image has been uploaded
if uploaded_file:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    display_width = 600
    st.image(image, caption="Uploaded Image", use_column_width='False',width=display_width)    
    
    # OCR Section
    st.header("Step 2: Extracting Text with EasyOCR")
    reader = easyocr.Reader(['en'])  # Initialize the OCR reader
    result = reader.readtext(np.array(image))  # Perform OCR
    
    # Extract the recognized text
    extracted_text = " ".join([text[1] for text in result])
    
    # Display the extracted text
    st.write("### Extracted Text")
    st.write(extracted_text)
    
    # Text Preprocessing Section
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('stopwords')

    def preprocess_txt(text):
        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        word_Lemmatized = WordNetLemmatizer()
        text = text.lower()
        text = re.sub(r"\n"," ",text)
        text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)
        text = re.sub(r'http\S+', '', text)
        stop = stopwords.words('english')
        pat = r'\b(?:{})\b'.format('|'.join(stop))
        text = text.replace(pat, '')
        text = text.replace(r'\s+', ' ')
        text = re.sub(r'[^a-zA-Z0-9 -]', '', text)
        text = re.sub('@[^\s]+','',text)
        text = word_tokenize(text)
        Final_words = []
        for word, tag in pos_tag(text):
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
        text = " ".join(Final_words)

        return text
    
    prepro_text = preprocess_txt(extracted_text)
    
    st.write("### Preprocessed Text")
    st.write(prepro_text)
    
    # Image Model Prediction Section
    st.header("Step 3: Image and Text Model Predictions")
    img_mod = load_model(r"C:\\STUDY COLLEGE\\SEM 6\\minor\\WEBAPP\\xcept.h5")  # Load image model
    resized_image = tf.image.resize(np.array(image), (256, 256))  # Resize image
    yhat_img = img_mod.predict(np.expand_dims(resized_image / 255, 0))  # Predict with image model
    
    st.write("### Image Model Prediction")
    st.write(yhat_img)
    
    # Text Model Prediction Section
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    def encode_new(text):
        dict_in = tokenizer.encode_plus(text, max_length=42, padding='max_length', truncation=True)
        input_id = np.array([dict_in['input_ids']])
        attention_mask = np.array([dict_in['attention_mask']])
        return input_id, attention_mask
    
    with tf.keras.utils.custom_object_scope({'TFRobertaModel': TFRobertaModel}):
        text_mod = load_model(r"C:\\STUDY COLLEGE\\SEM 6\\minor\\WEBAPP\\rbert.h5")
    
    input_id, attention_mask = encode_new(prepro_text)
    yhat_text = text_mod.predict([input_id, attention_mask])
    yhat_text = yhat_text[:, ::-1]
    
    st.write("### Text Model Prediction")
    st.write(yhat_text)
    
    # Combined Prediction
    yhat_combined = (yhat_img + yhat_text) / 2  # Combined prediction
    
    st.write("### Combined Models Prediction")
    st.write(yhat_combined)
    
    # Determine the final sentiment based on the combined prediction
    final_sentiment = np.argmax(yhat_combined)
    
    st.write("### FINAL SENTIMENT OF THE IMAGE")
    if final_sentiment == 0:
        st.write("🔴 Negative")
    elif final_sentiment == 1:
        st.write("⚪ Neutral")
    else:
        st.write("🟢 Positive")
