import os
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from dotenv import load_dotenv
load_dotenv()

class QuestionGenerator:
    def __init__(self, model_name):

        # Create the Groq LLM
        self.model = ChatGroq(
            groq_proxy=os.environ["GROQ_API_KEY"],
            model=model_name
        )
        
        self.prompt = self.create_prompt()
        
    def create_prompt(self):
        """Creates the chat prompt"""
        system = SystemMessagePromptTemplate.from_template(
            """
            You are a helpful tutor. Based on the following context from a document,please generate one insightful, 
            open-ended question that would test a user's understanding of the material.
            """
        )

        human = HumanMessagePromptTemplate.from_template(
            "<context>\n{context}\n</context>\nQuestion:"
        )
        
        prompt = ChatPromptTemplate.from_messages([system, human])
        return prompt

    def generate_question(self):
        """Generate andswer from a given context"""
        pass