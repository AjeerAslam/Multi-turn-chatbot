import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
import os
import time
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def main():
    st.header("Chat with PDF ..")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
 
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        
        # Initialize chats
        if "currentChat" not in st.session_state:
            st.session_state.currentChat = []
            st.session_state.memory = ConversationBufferMemory(
                    memory_key="history",
                    input_key="question"
                )
            
        with st.sidebar:
            st.write('Made with ❤️ by Ajeer')

        # Display chat messages from history on app rerun
        for message in st.session_state.currentChat:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("What is up?"):
            # Add user message to chat history
            st.session_state.currentChat.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):

                message_placeholder = st.empty()
                full_response = ""
                llm = OpenAI()

                template = """You are a chatbot having a conversation with a human.
                {context}
                {history}
                Human: {question}
                Chatbot:"""

                Prompt_Template = PromptTemplate(
                    input_variables=["history", "context", "question"], template=template
                )

                
                # memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
                chain = RetrievalQA.from_chain_type(llm,
                                                            chain_type='stuff',
                                                            retriever=VectorStore.as_retriever(),
                                                            chain_type_kwargs={
                                                                "prompt": Prompt_Template,
                                                                "memory": st.session_state.memory
                                                            })
                
                assistant_response=chain.run(prompt)
                # Simulate stream of response with milliseconds delay
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            # Add assistant response to chat history
            st.session_state.currentChat.append({"role": "assistant", "content": full_response})
 
if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] =st.secrets['OPENAI_API_KEY']
    main()