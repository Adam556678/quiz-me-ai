import streamlit as st
import os
import tempfile

from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from doc_processor import DocProcessor
from evaluator import Evaluator
from question_generator import QuestionGenerator

def reset_quiz_state():
    """Resets the session state variables related to the quiz."""
    st.session_state.processed = False
    st.session_state.question = None
    st.session_state.feedback = None
    if 'generator' in st.session_state:
        del st.session_state.generator
    if 'evaluator' in st.session_state:
        del st.session_state.evaluator


def process_document(file_path_or_url, source, model_name):
    
    # Process the document
    processor = DocProcessor()
    if source == "PDF":
        chunks = processor.process_pdf(file_path_or_url)
    elif source == "Text File":
        chunks = processor.process_txt(file_path_or_url)
    else: # Web page
        chunks = processor.process_web_page(file_path_or_url)
        
    # Create the Vector Database
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embedding=embeddings)

    # Create the Groq LLM
    llm = ChatGroq(
        groq_proxy=os.environ["GROQ_API_KEY"],
        model=model_name
    )

    # Initialize the generator and the evaluator
    st.session_state.generator = QuestionGenerator(llm, vector_db)
    st.session_state.evaluator = Evaluator(llm, vector_db)
    
    # file processing done
    st.session_state.processed = True
        

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
    key="source_type",
    on_change=reset_quiz_state
)

model = st.sidebar.selectbox(
    "Choose the Language Model",
    (
        "llama-3.3-70b-versatile", # LLama-3 70b
        "llama-3.1-8b-instant", # LLama-3 8b
        "openai/gpt-oss-120b" # GPT
    ),
    key="model_name"
)

# --- File upload or input depending on source ---
uploaded_file = None
url_input = None

if source == "PDF":
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        on_change=reset_quiz_state
    )
elif source == "Text File":
    uploaded_file = st.file_uploader(
        "Upload a text file", 
        type=["txt"],
        on_change=reset_quiz_state
    )
else:
    url_input = st.text_input(
        "Enter webpage URL",
        on_change=reset_quiz_state
    )


# Process input
if uploaded_file is not None:
    st.success(f"{uploaded_file.name} uploaded successfully!")
    # st.write("✅ Document ready for processing.")

elif url_input:
    st.success(f"URL received: {url_input}")
    # st.write("✅ Web page ready for processing.")

st.divider()


if (uploaded_file is not None or url_input) and not st.session_state.processed:
    if st.button("Start Quiz Engine"):
        with st.spinner("Processing the document.... This may take a while"):
            if uploaded_file:
                os.makedirs("tempDir", exist_ok=True)
                
                # Save the file 
                with open(os.path.join("tempDir", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                filepath = os.path.join("tempDir", uploaded_file.name)
                
                # process the user input
                process_document(filepath, source, model) 

            if url_input:
                # process the user input
                process_document(url_input, source, model) 
            
            # Start a new run
            st.rerun()

    # st.divider()

# --- Question generation section ---
if st.session_state.processed:
    st.info("Quiz engine is ready. Click below to generate your first question!")

    # Button to generate a new question
    if st.button("🧠 Generate a Question", disabled=st.session_state.generating):
        st.session_state.generating = True
        st.session_state.question = None # Clear previous question
        st.session_state.feedback = None # Clear previous feedback
        with st.spinner("The AI is thinking of a question...."):
            st.session_state.question = st.session_state.generator.generate_question()
        st.session_state.generating = False
        # st.rerun()

    # --- Question and answer section ---
    # Column Layouts
    col1, col2 = st.columns(2)

    if st.session_state.question:

        # AI question area 
        with col1:
            with st.container(border=True):
                st.write("**Question:**")
                st.write(st.session_state.question)

            st.write("") # Add a space

        # User answer area
        with col2:
            with st.form(key='answer_form'):
                user_answer = st.text_area(
                    "Your Answer",
                    placeholder="Type your answer here..."
                )    
                submit_button = st.form_submit_button(label='Submit Answer')

                if submit_button and user_answer:
                    with st.spinner("AI is cehcking your answer..."):
                        st.session_state.feedback = st.session_state.evaluator.validate_answer(
                            st.session_state.question,
                            user_answer
                        )
                
    # Display AI feedback
    if st.session_state.feedback:
        if st.session_state.feedback.strip().upper().startswith("CORRECT"):
            st.success(st.session_state.feedback)
        else:
            st.error(st.session_state.feedback)
    