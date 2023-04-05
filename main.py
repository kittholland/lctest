import os
import pinecone
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number

filepath = 'SCRulings22/21-887_k53m.pdf'
pinecone_index_name = "scrdemo"

#load pdf to text and split in chunks
loader = PyPDFLoader(filepath)
document = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
split_text = text_splitter.split_documents(document)
embeddings = OpenAIEmbeddings()

#initialize pinecone database and upsert vectors
pinecone.init()
add_docs = Pinecone.add_documents(split_text)
print(split_text)