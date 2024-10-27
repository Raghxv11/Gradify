import json
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
from langchain.docstore.document import Document
import tempfile

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Load environment variables and configure Gemini
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

# Initialize variables
visualization_data = []
percentage_grade = None
letter_grade = None

def convert_text_to_documents(text_chunks):
    # Convert each chunk of text to a Document object
    return [Document(page_content=chunk) for chunk in text_chunks]


def get_pdf_text(pdf_docs):
    text = ""
    tasks = {}

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        tasks[pdf] = text
        
    return tasks


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_rubric_chain():
    prompt_template = f"""
    Extract the given total points, criteria, and points/pts from the given rubric:\n {{context}}?\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def get_conversational_chain(rubric=None):
    if rubric:
        rubric_text = f" according to the provided rubric:\n{{rubric}}. Strictly based on the grading criteria, total points, and the points for each criteria given in the provided rubric do the grading\n"
    else:
        rubric_text = " based on the general grading criteria.\n"
    
    prompt_template = f"""
    You are a trained expert on writing and literary analysis. Your job is to accurately and effectively grade a student's essay{rubric_text}
    Respond back with graded points and a level for each criteria. Don't rewrite the rubric in order to save processing power. In the end, write short feedback about what steps they might take to improve on their assignment. Write a total percentage grade and letter grade. In your overall response, try to be lenient and keep in mind that the student is still learning. While grading the essay remember the writing level the student is at while considering their course level, grade level, and the overall expectations of writing should be producing.
    Your grade should only be below 70 percent if the essay does not succeed at all in any of the criteria. Your grade should only be below 80 percent if the essay is not sufficient in most of the criteria. Your grade should only be below 90% if there are a few criteria where the essay doesn't excell. Your grade should only be above 90 percent if the essay succeeds in most of the criteria.
    Understand that the essay was written by a human and think about their writing expectations for their grade level/course level, be lenient and give the student the benefit of the doubt.

        Context:\n {{context}}?\n
        Question: \n{{question}}\n

    Answer: Get the answer in beautiful format
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["rubric", "context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def extract_criteria_and_values(output_text):
    lines = output_text.split('\n')

    current_criteria = None
    for line in lines:
        line = line.strip()
        if line.startswith("**") and "Total Percentage Grade" not in line and "Letter Grade" not in line and "Feedback" not in line:
            current_criteria = line.split("**")[1].split(" ")[0]
            print(line)
            scored = line.split("/")[0].split(" ")[-1]
            total = line.split("/")[1].split(" ")[0]
            print(current_criteria, scored, total)
            visualization_data.append({
                "criteria": current_criteria,
                "scored": scored,
                "total": total
                })

def create_visualizations(output_text):
    # Split the text into lines
    lines = output_text.split('\n')

    # Iterate through each line to find the grades
    for line in lines:
        if "Total Percentage Grade" in line:
            percentage_grade = float(line.split(':')[1].strip().replace('%', '').replace('*', '').strip())
        elif "Letter Grade" in line:
            letter_grade = line.split(':')[1].strip().replace('*', '').strip()


#hello world
@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello, World!'})

@app.route('/api/grade/pdf', methods=['POST', 'OPTIONS'])
def grade_pdf():
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No PDF file uploaded'}), 400
        
        pdf_file = request.files.getlist('pdf')
        rubric_file = request.files.get('rubric')
        question = request.form.get('question')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        temp_pdfs = []
        pdf_names = []

        for pdf in pdf_file:
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
                pdf.save(temp_pdf.name)
                temp_pdfs.append(temp_pdf.name)
                pdf_names.append(pdf.filename)
                temp_pdf.close()
        
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_rubric:
            rubric_file.save(temp_rubric.name)
            temp_rubric.close()
            
        # Process the PDF
        raw_text = get_pdf_text(temp_pdfs)
        responses = ""

        for key, value in raw_text.items():
            text_chunks = get_text_chunks(value)
            get_vector_store(text_chunks)
            rubric_text = get_pdf_text([temp_rubric.name]) if temp_rubric.name else None

            if rubric_text:
                for rubric_key in rubric_text:
                    rubric_str = rubric_text[rubric_key]

                rubric_chain = get_rubric_chain()

                response = rubric_chain({"input_documents": convert_text_to_documents([rubric_str])}, return_only_outputs=True)
                rubric_text = response["output_text"]
            
            chain = get_conversational_chain(rubric=rubric_text)
            
            # Convert text chunks to Document objects
            documents = convert_text_to_documents(text_chunks)
            
            response = chain({"input_documents": documents, "rubric": rubric_text, "question": question}, return_only_outputs=True)
            responses += f"\nResponse for {pdf_names[temp_pdfs.index(key)]}: \n\n" + response['output_text']
            
            create_visualizations(response["output_text"])
            extract_criteria_and_values(response["output_text"])
        
        # Clean up temporary file
        for temp in temp_pdfs:
            os.unlink(temp)
        os.unlink(temp_rubric.name)
        
        return jsonify({
            'status': 'success',
            'response': responses
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualization', methods=['POST', 'OPTIONS'])
def visualization_pdf():
    try:       
        return json.dumps({"criteria": visualization_data, "percentage_grade": percentage_grade, "letter_grade": letter_grade})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

if __name__ == '__main__':
    app.run(debug=True, port=8080)
