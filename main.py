import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
import magic
from io import BytesIO
import os
from PyPDF2 import PdfReader
from docx import Document
import zipfile
import nltk

from langchain.chains.summarize import load_summarize_chain


# Prompt templates for dynamic values
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate, # I included this one so you know you'll have it but we won't be using it
    HumanMessagePromptTemplate
)

# To create our chat messages
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
API_KEY = os.getenv('PROJECT_API_KEY')

openai_api_key = API_KEY
nltk.download('averaged_perceptron_tagger')



def convert_pdf_to_txt(file_bytes):
    # Create a PDF file reader object from the bytes
    pdf = PdfReader(BytesIO(file_bytes))

    # Extract text from each page and join it into one string
    text = ' '.join(page.extract_text() for page in pdf.pages)

    return text

def convert_docx_to_txt(file_bytes):
    # Create a Document object from the bytes
    doc = Document(BytesIO(file_bytes))

    # Extract text from each paragraph and join it into one string
    text = ' '.join(paragraph.text for paragraph in doc.paragraphs)

    return text


def convert_txt_to_txt(path): ##HAVING PROBLEMS WITH TXT CONVERSION FOR SOME GODDAMN REASON!
    with open(path, 'r') as file:
        text = file.read()

    return text

def convert_uploaded_files_to_text(uploaded_files):
    text_files = {}

    for uploaded_file in uploaded_files:
        # Get the file extension
        _, extension = os.path.splitext(uploaded_file.name)

        # Read the file bytes
        file_bytes = uploaded_file.getvalue()

        if extension == '.pdf':
            text_files[uploaded_file.name] = convert_pdf_to_txt(file_bytes)
        elif extension == '.docx':
            text_files[uploaded_file.name] = convert_docx_to_txt(file_bytes)
        elif extension == '.txt':
            text_files[uploaded_file.name] = convert_txt_to_txt(file_bytes)
        else:
            print(f"Unsupported file format: {uploaded_file.name}")

    return text_files

# Set the title of the Streamlit application
st.title("RAG-based Question Answering App")

    # Create a slider to let the user select how many blocks they want to create
    # The slider ranges from 1 to 30, with a default value of 1
num_blocks = st.slider("Number of blocks", 1, 30, 1)

with st.form("file_upload_form"):
    # Initialize an empty list to store the blocks
    blocks = []

    # For each block that the user wants to create
    for i in range(num_blocks):
        # Display a subheader that shows which block the user is creating
        st.subheader(f'Block {i+1}')

        # Create a dropdown menu where the user can select the type of the block
        block_type = st.selectbox(f'Choose the type of block for Block {i+1}:', ('Title', 'Question', 'Text'), key=f'block_type{i}')

        # Create a single text input field for each block
        content = st.text_input('Enter the content', key=f'content{i}')

        # Add the block to the list of blocks
        blocks.append({'type': block_type, 'content': content})

    uploaded_files = st.file_uploader("Choose your source files", accept_multiple_files=True)

    # Create a submit button
    if st.form_submit_button('Submit'):
        with st.spinner('Processing...'):
            if uploaded_files:
                text_files = convert_uploaded_files_to_text(uploaded_files)
                transcript = ' '.join(text_files.values())
                # Load up your text splitter
                text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=2000, chunk_overlap=0)
                # I'm only doing the first 23250 characters. This to save on costs. When you're doing your exercise you can remove this to let all the data through
                #transcript_subsection_characters = 23250
                docs = text_splitter.create_documents([transcript])
                print(f"You have {len(docs)} docs.")
                texts = text_splitter.create_documents([transcript])
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                # Get your docsearch ready
                docsearch = FAISS.from_documents(texts, embeddings)
                # Load up your LLM
                llm = OpenAI(openai_api_key=openai_api_key)
                qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
                # Initialize an empty list to store the answers
                answers = []
                # Initialize a dictionary to store the blocks by type
                blocks_by_type = {'Title': [], 'Question': [], 'Text': []}
                # Process each block individually
                for block in blocks:
                    # Run the question through the qa model and get the answer
                    # if the block type is 'Question'
                    if block['type'] == 'Question':
                        query = str(block['content'])
                        answer = qa.run(query)
                        # Store the answer in the block
                        block['content'] = answer
                    # Append the content of the block to the corresponding list in the dictionary
                    blocks_by_type[block['type']].append(block['content'])

                # Display the blocks by type
                for block_type, block_contents in blocks_by_type.items():
                    # Join all the contents of the same type together into a single string
                    joined_content = ' '.join(block_contents)
                    if block_type == 'Title':
                        st.header(joined_content)
                    elif block_type == 'Question':
                        st.write(joined_content)  # Now this will display all the answers together
                    else:  # block_type == 'Text'
                        st.write(joined_content)     
