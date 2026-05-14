## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
The objective is to create a chatbot that can intelligently respond to queries based on information extracted from a PDF document. By using LangChain, the chatbot will be able to process the content of the PDF and use a language model to provide relevant answers to user queries. The effectiveness of the chatbot will be evaluated by testing it with various questions related to the document.

### DESIGN STEPS:

#### STEP 1:
Use a library like PyPDFLoader to load the PDF document. Split the content into smaller text chunks using a TextSplitter (e.g., RecursiveCharacterTextSplitter) for efficient information retrieval.

#### STEP 2:
Convert the text chunks into numerical embeddings using OpenAIEmbeddings and store them in a vector database (e.g., Chroma) to enable similarity-based search.

#### STEP 3:
Use the vector database's retriever to fetch relevant text chunks based on user queries. Set the retriever parameters (e.g., search type) for accurate results.

### STEP 4:
Use an OpenAI language model (e.g., GPT-4 or GPT-3.5) and a retrieval chain like RetrievalQA or ConversationalRetrievalChain to combine document context with conversational responses.

### STEP 5:
Create a conversational interface (e.g., a command-line or GUI-based system) where users can input queries, and the chatbot responds with context-aware answers based on the PDF.

### PROGRAM:
```py
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # Load OpenAI API key
openai_api_key = os.environ['OPENAI_API_KEY']

def load_pdf_to_db(file_path):
    # Load the PDF file
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Split the documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    
    # Embed the documents
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    
    # Set retriever to fetch relevant document chunks
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever
def create_conversational_chain(retriever):
    # Initialize memory for conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Define the conversational retrieval chain
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),  # Using OpenAI Chat model
        retriever=retriever,
        memory=memory
    )
    return conversational_chain
if __name__ == "__main__":
    # Load the PDF file
    pdf_file_path = "DOC1.pdf"  # Replace with your file path
    retriever = load_pdf_to_db(pdf_file_path)
    
    # Create chatbot chain
    chatbot = create_conversational_chain(retriever)
    
    # Start a conversation
    print("Welcome to the PDF Question-Answering Chatbot!")
    print("Type 'exit' to quit.")
    
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Chatbot: Thanks for chatting! Goodbye!")
            break
        
        result = chatbot({"question": user_query})
        print("Chatbot:", result["answer"])
```


### OUTPUT:

<img width="840" height="492" alt="image" src="https://github.com/user-attachments/assets/0a8acf23-00c0-4f7a-8089-619f63327814" />



### RESULT:
Prompt: A structured prompt template was designed to pass the document content and user query to the language model.

Model: OpenAI's GPT model was used to process the input data and provide an answer based on the document's content.

Output Parsing: The model's output is returned as the answer to the query, ensuring that it provides relevant responses based on the content extracted from the PDF.
