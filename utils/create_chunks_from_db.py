from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def create_chunks_from_db(db):
    documents = [
        Document(page_content=doc.page_content, metadata=doc.metadata)
        for doc in db.docstore._dict.values()
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=20)
    chunks = text_splitter.split_documents(documents)
    return chunks
