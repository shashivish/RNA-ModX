import streamlit as st

from encode_sequence import RNAPredictor,RNAClassifier, RNATransformerModel
from format_results import format_to_json, save_json_to_excel

st.title("Let's predict RNA Sequences!")

# Adding a text input widget
rna_sequence = st.text_input("Enter RNA Sequence", "Type here...")

# Make prediction when user submits input
if st.button("Predict"):
    print([rna_sequence][0])
    encoding_file = './3-mer-dictionary.pkl'

    rna_predictor = RNAPredictor(encoder_file_path=encoding_file, model_directory_path='../model')
    response = rna_predictor.get_predictions([rna_sequence][0])

    print('response')
    print(response)

    formatted_result = format_to_json(response)

    st.write("Sequence:", [rna_sequence][0])

    # Display the formatted result as JSON
    st.dataframe(formatted_result)

    # Save the JSON data to an Excel file
    save_json_to_excel(formatted_result, "rna-sequence_prediction_results.xlsx")