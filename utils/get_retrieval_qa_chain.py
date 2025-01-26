from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from .create_chunks_from_db import create_chunks_from_db

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from .create_chunks_from_db import create_chunks_from_db
from langchain import PromptTemplate

def get_retrieval_qa_chain(llm, prompt, db):
    # Create retrievers
    vector_retriever = db.as_retriever(
        search_kwargs={"k": 5},
        search_type="similarity"
    )
    prompt = PromptTemplate(template=prompt, input_variables=["chat_history",'context', 'question'])
    print(prompt)
    
    # Create BM25 retriever
    chunks = create_chunks_from_db(db)  # Ensure this matches your chunking logic
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 5  # Increase for more keyword-based results
    
    # Create ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[0.5, 0.5]
    )
    
    # Create QA chain with answer generation
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=ensemble_retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return qa_chain
