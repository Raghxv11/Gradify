from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import tempfile

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Load environment variables and configure Gemini
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

# Reuse functions from docs.py with modifications for API context
def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Grade the following student work based on the provided context and question.
    Context: {context}
    Question: {question}
    Please provide a detailed evaluation.
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

#hello world
@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello, World!'})

@app.route('/api/grade/pdf', methods=['POST', 'OPTIONS'])
def grade_pdf():
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No PDF file uploaded'}), 400
        
        pdf_file = request.files['pdf']
        question = request.form.get('question')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
            pdf_file.save(temp_pdf.name)
            temp_pdf.close()
            
            # Process the PDF
            raw_text = get_pdf_text(temp_pdf.name)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            
            # Load the vector store and process the question
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            relevant_docs = vector_store.similarity_search(question)
            
            # Get the response using the conversational chain
            chain = get_conversational_chain()
            response = chain(
                {"input_documents": relevant_docs, "question": question},
                return_only_outputs=True
            )
            
            # Clean up temporary file
            os.unlink(temp_pdf.name)
            
            return jsonify({
                'status': 'success',
                'response': response['output_text']
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)