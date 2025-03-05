import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS


def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="gemma:2b")
        st.session_state.loader = PyPDFLoader(
            "/home/srinivasan/PycharmProjects/Langchain/Rag/NIPS-2017-attention-is-all-you-need-Paper.pdf")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.loader = PyPDFLoader(
            "/home/srinivasan/PycharmProjects/Langchain/Rag/2304.10557v5.pdf")
        st.session_state.docs += st.session_state.loader.load()
        print(st.session_state.docs)

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


st.title("ChatGroq llama3 demo")
llm = ChatGroq(groq_api_key = groq_api_key,model_name = "llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only, please provide the most accurate response based on the question
<context>
{context}
<context>
Question:{input}
    """
)

prompt1 = st.text_input("Enter your question here")
if st.button("Document Embeddings"):
    vector_embeddings()
    st.write("vector db is ready")

import time
if prompt1:
    start = time.process_time()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": prompt1})
    print("Response time : ", time.process_time() - start)
    st.write(response["answer"])

    with st.expander("Show Context"):
        # st.write(response["context"])
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("_________________________")
            st.write(doc.metadata)
            st.write("_________________________")
            st.write("_________________________")