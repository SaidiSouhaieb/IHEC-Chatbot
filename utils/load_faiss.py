import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_faiss_db(DB_PATH, EMBEDING_MODEL):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDING_MODEL, model_kwargs={"device": "cpu"}
    )
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return db
