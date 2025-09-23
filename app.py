import streamlit as st
import os
import tempfile

from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from doc_processor import DocProcessor
from evaluator import Evaluator
from question_generator import QuestionGenerator


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
    key="source_type"
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
        st.rerun()

    st.divider()
    
    # Display question and answer if a question has been generated
    st.markdown(
        f"""
            <div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #e6e6e6;">
                <p style="font-size: 1.1em; color: #333;"><strong>Question:</strong></p>
                
                <!-- ADDED a style to this paragraph -->
                <p style="color: #333;">{st.session_state.question}</p> 
                
            </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("") # Add a space

    # User answer area
    with st.form(key='answer_form'):
        user_answer = st.text_area(
            "Your Answer",
            placeholder="Type you answer here..."
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
    