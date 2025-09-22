import os
import random 
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains.llm import LLMChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

class QuestionGenerator:
    def __init__(self, model_name : str, vectorDB : FAISS):

        self.vector_db = vectorDB
        
        # Create the Groq LLM
        self.model = ChatGroq(
            groq_proxy=os.environ["GROQ_API_KEY"],
            model=model_name
        )
        
        # create the chat prompt
        self.prompt = self.create_prompt()

        # create the LLM chain
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt) 
        
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

    def generate_question(self):
        """
        Pulls a random chunk from the vector store and generates a question about it.
        """
        retriever = self.vector_db.as_retriever(search_kwargs={"k" : 5})

        # get few random chunks from the document 
        docs = retriever.get_relevant_documents("a")

        # pick one random chunk
        random_doc = random.choice(docs)

        # generate a question
        response = self.chain.invoke({"context" : random_doc.page_content})
        
        return response["text"].strip()
        
        
        