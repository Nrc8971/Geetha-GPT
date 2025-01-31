import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm


def main():
    st.title("Bhagwat Geeta 🦚")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Ask question Krishna will answer your questions!!!")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'You', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """Final Prompt: Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know; strictly don't add anything from your side.

    Context: {context}
    Question: {question}

    Only return the helpful answer. Give a direct answer with reference to the context. Answer it like Krishna, in a melodious and little bit brief way. 
    Answer:
    """
        
        HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN=os.environ.get("HF_TOKEN")

        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt})

            result=response["result"]
            source_documents=response["source_documents"]
            #result_to_show=result+"\nSource Docs:\n"+str(source_documents)
            #response="Hi, welocme to my devine world!"
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role':'Krishna', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()