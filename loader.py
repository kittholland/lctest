import os
import pinecone
import yaml
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

directory_path = config['docs_path']
pinecone_index_name = config['pinecone_index_name']
pinecone_environment = config['pinecone_environment']

pinecone.init(environment=pinecone_environment)

for filename in os.listdir(directory_path):
    #load pdf documents and split in chunks
    filepath = f"{directory_path}{filename}"
    print(f"Processing {filepath}...")
    loader = PyPDFLoader(filepath)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    docs = text_splitter.split_documents(document)
    embeddings = OpenAIEmbeddings()

    #initialize pinecone database and upsert vectors
    docsearch = Pinecone.from_documents(
        docs,
        embeddings,
        index_name=pinecone_index_name
    )