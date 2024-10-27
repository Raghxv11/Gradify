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

def get_conversational_chain(rubric_text=None):
    if rubric_text:
        rubric_prompt = f" according to the provided rubric:\n{rubric_text}. Strictly based on the grading criteria, total points, and the points for each criteria given in the provided rubric do the grading\n"
    else:
        rubric_prompt = " based on the general grading criteria.\n"
    
    prompt_template = f"""
    You are a trained expert on writing and literary analysis. Your job is to accurately and effectively grade a student's essay{rubric_prompt}
    Respond back with graded points and a level for each criteria. Don't rewrite the rubric in order to save processing power. In the end, write short feedback about what steps they might take to improve on their assignment. Write a total percentage grade and letter grade. In your overall response, try to be lenient and keep in mind that the student is still learning. While grading the essay remember the writing level the student is at while considering their course level, grade level, and the overall expectations of writing should be producing.
    Your grade should only be below 70 percent if the essay does not succeed at all in any of the criteria. Your grade should only be below 80 percent if the essay is not sufficient in most of the criteria. Your grade should only be below 90% if there are a few criteria where the essay doesn't excell. Your grade should only be above 90 percent if the essay succeeds in most of the criteria.
    Understand that the essay was written by a human and think about their writing expectations for their grade level/course level, be lenient and give the student the benefit of the doubt.

        Context:\n {{context}}?\n
        Question: \n{{question}}\n

    Answer: Get the answer in beautiful format
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
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
        
        pdf_files = request.files.getlist('pdf')
        rubric_file = request.files.get('rubric')
        question = request.form.get('question')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Process multiple PDF files
        all_text = ""
        for pdf_file in pdf_files:
            with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
                pdf_file.save(temp_pdf.name)
                temp_pdf.close()
                all_text += get_pdf_text(temp_pdf.name)
                os.unlink(temp_pdf.name)

        # Process rubric file if provided
        rubric_text = ""
        if rubric_file:
            with tempfile.NamedTemporaryFile(delete=False) as temp_rubric:
                rubric_file.save(temp_rubric.name)
                temp_rubric.close()
                rubric_text = get_pdf_text(temp_rubric.name)
                os.unlink(temp_rubric.name)

        # Process the text
        text_chunks = get_text_chunks(all_text)
        get_vector_store(text_chunks)
        
        # Load the vector store and process the question
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        relevant_docs = vector_store.similarity_search(question)
        
        # Get the response using the conversational chain
        chain = get_conversational_chain(rubric_text)
        response = chain(
            {"input_documents": relevant_docs, "rubric": rubric_text, "question": question},
            return_only_outputs=True
        )
        
        return jsonify({
            'status': 'success',
            'response': response['output_text']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)
