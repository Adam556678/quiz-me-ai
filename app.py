import streamlit as st

st.set_page_config(page_title="QuizMe AI", page_icon="🤖")

st.title("QuizMe AI")
st.caption("Upload your study materials and test yourself with AI-generated quizzes!")

# Sidebar for options
st.sidebar.header("📂 Input Source")
source = st.sidebar.radio(
    "Choose your document source:",
    ("PDF", "Text File", "Web Page")
)

# File upload or input depending on source
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

# Question generation section

if uploaded_file is not None or url_input:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Question Generated:")

    with col2:
        st.text("Your Answer")
        answer = st.text_input("Give an answer to the question")