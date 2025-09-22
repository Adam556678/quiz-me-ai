import os
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
load_dotenv()

class QuestionGenerator:
    def __init__(self, model_name, vectorDB : FAISS):

        # Create the Groq LLM
        self.model = ChatGroq(
            groq_proxy=os.environ["GROQ_API_KEY"],
            model=model_name
        )
        
        # create the chat prompt
        self.prompt = self.create_prompt()

        # create the retriveal chain
        self.chain = self.create_chain(vectorDB)
        
    def create_prompt(self):
        """Creates the chat prompt"""
        system = SystemMessagePromptTemplate.from_template(
            """
            You are a helpful tutor. Based on the following context from a document, please generate one insightful, 
            open-ended question that would test a user's understanding of the material.
            """
        )

        human = HumanMessagePromptTemplate.from_template(
            "<context>\n{context}\n</context>\nQuestion:"
        )
        
        prompt = ChatPromptTemplate.from_messages([system, human])
        return prompt

    def create_chain(self, vectorDB):
        """Create the retrieval chain"""
        
        # create the document chain
        document_chain = create_stuff_documents_chain(
            llm=self.model,
            prompt=self.prompt
        )
        
        # create the retrieval chain
        retriever = vectorDB.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain
        

    def generate_question(self):
        """Generate questions from a given context"""
        response = self.chain.invoke({"input" : self.prompt})
        return response["answer"]        
        