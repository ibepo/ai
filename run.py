import chromadb
import ollama
import pandas as pd
import streamlit as st


def initialize():
    if "already_executed" not in st.session_state:
        st.session_state.already_executed = False

    if not st.session_state.already_executed:
        setup_database()


def setup_database():
    client=chromadb.Client()  
    file_path = "data/data.csv"
    documents=pd.read_excel(file_path,header=None)

    collection=client.get_or_create_collection(name="demodocs")

    for index,content in documents.iterrows():
        response=ollama.embeddings(model="mxbai-embeddings-large",prompt=content[0])
        collection.add(ids={str(index)},embeddings=[response["embeddings"]],documents=[content[0]])

    st.session_state.already_executed=True
    st.session_state.collection=collection