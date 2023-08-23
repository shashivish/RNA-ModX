import streamlit as st

from RNAModXApp.scripts.encode_sequence import RNAPredictor
from format_results import format_to_json, save_json_to_excel

st.title("Let's predict RNA Sequences!")

# Adding a text input widget
rna_sequence = st.text_input("Enter RNA Sequence", "Type here...")

# Make prediction when user submits input
if st.button("Predict"):
    # Process the input using the ML model
    encoding_file = './3-mer-dictionary.pkl'
    format_results = 'GGGGCCGTGGATACCTGCCTTTTAATTCTTTTTTATTCGCCCATCGGGGCCGCGGATACCTGCTTTTTATTTTTTTTTCCTTAGCCCATCGGGGTATCGGATACCTGCTGATTCCCTTCCCCTCTGAACCCCCAACACTCTGGCCCATCGGGGTGACGGATATCTGCTTTTTAAAAATTTTCTTTTTTTGGCCCATCGGGGCTTCGGATA'

    rna_predictor = RNAPredictor(encoder_file_path=encoding_file, model_directory_path='../model')
    response = rna_predictor.get_predictions(format_results)
    print(response)

    formatted_result = format_to_json(response)

    st.write("Sequence:", response)
    # Display the formatted result as JSON
    st.table(formatted_result)

    # Save the JSON data to an Excel file
    save_json_to_excel(formatted_result, "sentiment_results.xlsx")