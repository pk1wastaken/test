import streamlit as st
import xgboost
import pickle
import os
import nltk
from pathlib import Path
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from transformers import BertTokenizer  
import gdown
#current_directory = Path(__file__).parent #Get current directory
#file = open(os.path.join(current_directory, 'bestBERT.pkl'), 'rb')
import requests
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
nltk.download('all')
nltk.download('stopwords')
# Replace 'YOUR_DRIVE_LINK_HERE' with the actual link to your .pkl file
google_drive_link = 'https://drive.google.com/file/d/153DIFyVs6Zc8PakmrwnZRMuG-wekYT7M/view?usp=drive_link'


# Define a function to download the .pkl file and load it as a model
def load_model_from_drive(google_drive_link):
    try:
        # Define the URL for the Google Drive file
        url = f'https://drive.google.com/uc?id={google_drive_link}'
        
        # Download the file using gdown
        gdown.download(url, 'model.pkl', quiet=False)
        
        # Load the model from the downloaded .pkl file
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        return model
    except Exception as e:
        print("An error occurred:", str(e))
        return None

# Call the function to load the model from the Google Drive link
loaded_model = load_model_from_drive(google_drive_link)

# Check if the model was successfully loaded
if loaded_model is not None:
    print("Model loaded successfully.")
    # You can now use the 'loaded_model' in your application.
else:
    print("Failed to load the model.")
# Load the trained model from the .pkl file
#with open('bestBERT.pkl', 'rb') as file:
 #   model = pickle.load(file)
def anonymisation (text , first_party , second_party):
    
    first_words = first_party.split()
    second_words = second_party.split()
    
    nltk_results = ne_chunk(pos_tag(word_tokenize(text)))
    
    for nltk_result in nltk_results:
        if type(nltk_result) == Tree:
            name = ''
            for nltk_result_leaf in nltk_result.leaves():
                name += nltk_result_leaf[0] + ' '
        
                #print ('Type: ', nltk_result.label(), 'Name: ', name)
                if nltk_result.label() != 'PERSON':
                    text = text.replace(name,'anonymized')
            
    text = text.replace(first_party ,'first_party') 
    text = text.replace(second_party , 'second_party')
    for f_w in first_words:
        text = text.replace(f_w , 'first_party')
    for s_w in second_words :
        text = text.replace(s_w , 'second_party')
                            
    return text
    
def convert_text (text):

  tokenn = tokenizer.encode_plus(text ,
      max_length=256,
      truncation = True,
      padding='max_length',
      add_special_tokens=True,
      return_tensors='tf')
  return tokenn.input_ids

w=0
v=0

def main():
    st.markdown("<h1 style='text-align: center; color: White;background-color:#e84343'>Legal Case Predictor</h1>",unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: Black;'>Please provide me with the first party name, second party name, and all the available facts using said party names.</h3>", unsafe_allow_html=True)
    
    p1 = st.text_input("Enter first party name:")
    p2 = st.text_input("Enter second party name:")

    st.write("Please enter all the facts pertaining to the case file.(use the given party names)")
    text= st.text_input("facts:")
    prompt = anonymisation(text,p1,p2)
    xx= convert_text(prompt)
    
    if st.button('Predict'):
        y=model.predict(xx)
        w=y[0][0] *100
        v=y[0][1] *100
        if y[0][0]> y[0][1] and y[0][0]>0.6 :
            st.write(f"{p1} is predicted to win the case with a winning probability of {w}")
        elif y[0][1]> y[0][0] and y[0][1]>0.6: 
            st.write(f"{p2} is predicted to win the case with a winning probability of {v}")
        else:
            st.write(f"Couldnt predict the winner as the probability to win of {p1} is {w} and {p2} is {v}. This is way too close and could go in either way.")


if __name__ == '__main__':
    main()
