import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

vector_form = pickle.load(open('vect.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

# re.sub(pattern, replacement, string): Searches for the pattern in the string and replaces it with the specified replacement string
def remove_special_character(obj):
    # Define a regular expression pattern to match special characters
    pattern = r'[^a-zA-Z0-9\s]'  # This pattern matches anything that's not a letter, digit, or whitespace
       # Use the sub() function to replace matched patterns with an empty string
    clean_text = re.sub(pattern, '', obj)
    clean_text=clean_text.lower()
    
    return clean_text

def Suicide_or_not(Tweets):
    Tweets=remove_special_character(Tweets)
    input_data=[Tweets]
    vector_form1=vector_form.transform(input_data)
    prediction=load_model.predict(vector_form1)
    return prediction



if __name__ == '__main__':
    st.title('Suicidal Tweet Detection Classifier app')
    st.subheader("Input the News content below")
    sentence = st.text_area("Enter your news content here", "",height=200)
    predict_btt = st.button("predict")
    if predict_btt:
        prediction_class=Suicide_or_not(sentence)
        print(prediction_class)
        if prediction_class == [0]:
            st.success('Not Suicidal Post')
        if prediction_class == [1]:
            st.warning('Potenial Suicidal Post')