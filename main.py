import os, yaml, pinecone, tiktoken
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

pinecone_index_name = config['pinecone_index_name']
pinecone_environment = config['pinecone_environment']
pinecone.init(environment=pinecone_environment)
docsearch = Pinecone.from_existing_index(pinecone_index_name, OpenAIEmbeddings())

qa_chain = load_qa_chain(OpenAI(temperature=0.8), chain_type="stuff", verbose=True)
qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=docsearch.as_retriever(), verbose=True)
query = "What is Bartenwerfer v. Buckley?"
qa.run(query)