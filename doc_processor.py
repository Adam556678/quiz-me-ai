from langchain_community.document_loaders import(
    WebBaseLoader,
    PyPDFLoader,
    UnstructuredPDFLoader    
)  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import logging

class DocProcessor:
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """Initialize processor with configurable chunking."""
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def chunk_document(self, doc):
        """Splits the document into chunks."""
        return self.splitter.split_documents(doc)
        
    
    def process_pdf(self, pdf : str):
        """Process PDF given by user"""
        
        try:
            # try a loader for unstructured PDFs
            loader = UnstructuredPDFLoader(pdf, mode="single")
            document = loader.load()
            return self.chunk_document(document)            
        except Exception as e:
            logging.error(f"Failed to process PDF with the UnstructuredLoader : {e}")
            
            # Switch to a simple PDF Loader if
            # the first one failed
            try:
                loader = PyPDFLoader(pdf)
                document = loader.load()
                return self.chunk_document(document)
            except Exception as e:
                logging.error("Failed to process the PDF")
                return []
        
    def process_web_page(self, link : str):
        """Process web page given by user"""
        loader = WebBaseLoader(link)
        document = loader.load()
        return self.chunk_document(document)
    
    def process_txt(self, txt_file : str):
        """Process plain text file given by user."""
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Wrap the text in a LangChain Document
        document = [Document(page_content=text, metadata={"source": txt_file})]
        return self.chunk_document(document)
 
    
    def process_ppt():
        """Process PowerPoint file given by user"""
        #TODO: Add pptx loader
        pass
