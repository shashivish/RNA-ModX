import streamlit as st

from encode_sequence import RNAPredictor,RNAClassifier, RNATransformerModel
from format_results import format_to_json, save_json_to_excel

import os

current_directory = os.getcwd()

# Get the path of the current script
script_path = os.path.abspath(__file__)

data_file_path = os.path.join(os.path.dirname(script_path), "3-mer-dictionary.pkl")
st.text(data_file_path)
if os.path.exists(data_file_path):
    st.text("File exists")
    st.text(data_file_path)
else:
    st.text("File does not exist")
    st.text(data_file_path)

# Construct relative path to a file (assuming it's in the same directory as the script)
relative_path = os.path.join(os.path.dirname(script_path), "model", "XGB_OverallBinary.pkl")
st.text(relative_path)
if os.path.exists(relative_path):
    st.text("File exists")
    st.text(relative_path)
else:
    st.text("File does not exist")
    st.text(relative_path)
st.title("Let's predict RNA Sequences!")

# Adding a text input widget
rna_sequence = st.text_input("Enter RNA Sequence", "Type here...")

# Make prediction when user submits input
if st.button("Predict"):
    print([rna_sequence][0])
    encoding_file = './3-mer-dictionary.pkl'

    rna_predictor = RNAPredictor(encoder_file_path=os.path.join(os.path.dirname(script_path), "3-mer-dictionary.pkl"),
                                 model_directory_path=os.path.join(os.path.dirname(script_path), "model"))
    response = rna_predictor.get_predictions([rna_sequence][0])

    print('response')
    print(response)

    formatted_result = format_to_json(response)

    st.write("Sequence:", [rna_sequence][0])

    # Display the formatted result as JSON
    st.dataframe(formatted_result)

    # Save the JSON data to an Excel file
    save_json_to_excel(formatted_result, "rna-sequence_prediction_results.xlsx")