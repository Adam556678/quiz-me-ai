import streamlit as st
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings

from doc_processor import DocProcessor
from question_generator import QuestionGenerator


def generate_answer(file_path_or_url, source, model_name):
    # Process the document
    processor = DocProcessor()
    if source == "PDF":
        chunks = processor.process_pdf(file_path_or_url)
    elif source == "Text File":
        chunks = processor.process_txt(file_path_or_url)
    else: # Web page
        chunks = processor.process_web_page(file_path_or_url)
        
    # Create Vector Database
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = FAISS.from_documents(chunks, embedding=embeddings)

    # Initialize the generator
    generator = QuestionGenerator(model_name, vector_db)

    # TODO: Implement the Evaluator Class

        

st.set_page_config(page_title="QuizMe AI", page_icon="🤖")

st.title("QuizMe AI")
st.caption("Upload your study materials and test yourself with AI-generated quizzes!")

# --- Initialize session state variables ---
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'question' not in st.session_state:
    st.session_state.question = None
if 'feedback' not in st.session_state:
    st.session_state.feedback = None
if 'generating' not in st.session_state:
    st.session_state.generating = False


# --- Sidebar for options ---
st.sidebar.header("📂 Input Source")
source = st.sidebar.radio(
    "Choose your document source:",
    ("PDF", "Text File", "Web Page"),
    key="source_type"
)

model = st.selectbox(
    "Choose the Language Model",
    (
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b"
    ),
    key="model_name"
)

# --- File upload or input depending on source ---
uploaded_file = None
url_input = None

if source == "PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
elif source == "Text File":
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
else:
    url_input = st.text_input("Enter webpage URL")

# Process input
if uploaded_file is not None:
    st.success(f"{uploaded_file.name} uploaded successfully!")
    # st.write("✅ Document ready for processing.")

elif url_input:
    st.success(f"URL received: {url_input}")
    # st.write("✅ Web page ready for processing.")

st.divider()


if (uploaded_file is not None or url_input) and st.session_state.processed:
    if st.button("Start Quiz Engine"):
        with st.spinner("Processing document.... This may take a moment"):
            if uploaded_file:
                
                # Save the file 
                with open(os.path.join("tempDir", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                flepath = os.path.join("tempDir", uploaded_file.name)
                # TODO: Call the generate answer function 


# --- Question generation section ---
if st.session_state.processed:
    st.info("Quiz engine is ready. Click below to generate your first question!")

    # Button to generate a nwe question
    if st.button("🧠 Generate New Question", disabled=st.session_state.generating):
        st.session_state.generating = True
        st.session_state.question = None # Clear previous question
        st.session_state.feedback = None # Clear previous feedback
        with st.spinner("The AI of thinking of a question...."):
            