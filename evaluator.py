from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

class Evaluator:
    def __init__(self, llm : ChatGroq, vectorDB : FAISS):
        self.llm = llm
        self.vector_db = vectorDB
        self.retrieval_chain = self._create_eval_chain() 
        
        
    def _create_eval_prompt(self):
        """Creates the asnwer evaluation prompt"""
        prompt = ChatPromptTemplate.from_template("""
            You are an expert examiner. Your task is to determine if the user's answer is correct
            based ONLY on the provided document context.
            Do not use any outside knowledge.
            
            **Document Context:**
            <context>
            {context}
            </context>
            
            **Question Asked to the User:**
            {input}
            
            **The User's Answer:**
            {answer}
            
            **Your Task:**
            1.  Carefully analyze the user's answer against the document context.
            2.  Determine if the answer is correct, incorrect, or partially correct.
            3.  Provide a brief, one or two-sentence explanation for your decision.
            4.  Start your response with the single word 'CORRECT:', 'INCORRECT:', or 'PARTIALLY CORRECT:'.
            
            **Response:**    
            """)
        return prompt

    def _create_eval_chain(self):
        """Creates chain for answer evaluation"""

        validation_prompt = self._create_eval_prompt()
        
        # Create a document chain that knows how to process the retrieved context
        document_chain = create_stuff_documents_chain(
            llm=self.llm, 
            prompt=validation_prompt
        )
        
        # create the retriever from the Vector Database
        retriever = self.vector_db.as_retriever(search_kwargs={"k" : 4})
        
        # Crate the full retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        return retrieval_chain
    
    def validate_answer(self, question_asked : str, user_answer : str):
        response = self.retrieval_chain.invoke({
            "input" : question_asked,
            "answer" : user_answer
        })
        
        return response["answer"].strip()