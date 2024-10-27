
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.docstore.document import Document

copyleaks_api_key = os.getenv("COPYLEAKS_API_KEY")

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Implementation of the Copyleaks API
import http.client
import json

import uuid
import time


def create_scan_id(student_id):
    # Generate a submission_id with a shortened UUID
    uuid_part = str(uuid.uuid4()).replace('-', '')[:8]
    submission_id = f"submission{uuid_part}"
    
    # Combine the components
    scan_id = f"{student_id}-{submission_id}"
    
    # Ensure the scan_id meets the length requirement (3-36 characters)
    if len(scan_id) > 36:
        scan_id = scan_id[:36]
    
    return scan_id

def check_content_origin(copyleaks_api_key, text):
    conn = http.client.HTTPSConnection("api.copyleaks.com")

    login_token = os.getenv("COPYLEAKS_LOGIN_TOKEN")
    
    headers = {
        'Authorization': f"Bearer {login_token}",
        'Content-Type': "application/json",
        'Accept': "application/json"
    }
    
    payload = json.dumps({
        "text": text,
        "language": "en",
        "sandbox": False
    })

    try:
        scan_id = create_scan_id("studentid123")
        conn.request("POST", f"/v2/writer-detector/{scan_id}/check", body=payload, headers=headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        
        try:
            result = json.loads(data)
        except json.JSONDecodeError:
            return f"Error: Invalid JSON response from API. Raw response: {data}"
        
        summary = result.get('summary', {})
        ai_score = summary.get('ai', 0)
        human_score = summary.get('human', 0)
        probability = summary.get('probability', 0.0)
        
        total_words = result.get('scannedDocument', {}).get('totalWords', 0)
        
        if ai_score > human_score:
            classification = "AI-generated content"
        elif human_score > ai_score:
            classification = "Human-generated content"
        else:
            classification = "Undetermined"

        
        return {
            "classification": classification,
            "ai_score": ai_score,
            "human_score": human_score,
            "total_words": total_words,
            "model_version": result.get('modelVersion', 'Unknown'),
        }
    
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        conn.close()


def convert_text_to_documents(text_chunks):
    #   Convert each chunk of text to a Document object
    return [Document(page_content=chunk) for chunk in text_chunks]


def get_pdf_text(pdf_docs):
    text = ""
    tasks = {}

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        tasks[pdf.name] = text
        
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


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    # print(response)
    st.write("Reply: ", response["output_text"])


def extract_criteria_and_values(output_text):
    criteria_values = []
    lines = output_text.split('\n')

    current_criteria = None
    for line in lines:
        line = line.strip()
        if line.startswith("**") and "Total Percentage Grade" not in line and "Letter Grade" not in line and "Feedback" not in line:
            current_criteria = line.split("**")[1].split(" ")[0]
            scored = line.split("/")[0].split(" ")[-1]
            total = line.split("/")[1].split(" ")[0]
            criteria_values.append((current_criteria, scored, total))

    return criteria_values

def create_visualizations(output_text):
    # Initialize variables
    percentage_grade = None
    letter_grade = None

    # Split the text into lines
    lines = output_text.split('\n')

    # Iterate through each line to find the grades
    for line in lines:
        if "Total Percentage Grade" in line:
            percentage_grade = float(line.split(':')[1].strip().replace('%', '').replace('*', '').strip())
        elif "Letter Grade" in line:
            letter_grade = line.split(':')[1].strip().replace('*', '').strip()
    
    print("percentage_grade: ", percentage_grade)
    print("letter_grade: ", letter_grade)

def main():
    st.header("Automate grading using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your Essay PDF Files",
            accept_multiple_files=True,
        )
        rubric_doc = st.file_uploader(
            "Optionally upload a Rubric PDF File",
            type=['pdf'],
            accept_multiple_files=False
        )
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                st.success("Done")
    
    if user_question:
        raw_text = get_pdf_text(pdf_docs)

        for key, value in raw_text.items():

            content_check_result = check_content_origin(copyleaks_api_key, value)
            st.write("Content Origin Check:")
        
            if isinstance(content_check_result, dict):
                st.write(f"Classification: {content_check_result['classification']}")
                st.write(f"AI Score: {content_check_result['ai_score']:.2f}")
                st.write(f"Human Score: {content_check_result['human_score']:.2f}")
                st.write(f"Total Words: {content_check_result['total_words']}")
                st.write(f"Model Version: {content_check_result['model_version']}")
            else:
                st.write(content_check_result)

            text_chunks = get_text_chunks(value)
            get_vector_store(text_chunks)
            rubric_text = get_pdf_text([rubric_doc]) if rubric_doc else None

            if rubric_text:
                for rubric_key in rubric_text:
                    rubric_str = rubric_text[rubric_key]

                rubric_chain = get_rubric_chain()

                response = rubric_chain({"input_documents": convert_text_to_documents([rubric_str])}, return_only_outputs=True)
                rubric_text = response["output_text"]
            chain = get_conversational_chain(rubric=rubric_text)
            
            # Convert text chunks to Document objects
            documents = convert_text_to_documents(text_chunks)
            
            if user_question:
                response = chain({"input_documents": documents, "rubric": rubric_text, "question": user_question}, return_only_outputs=True)
                st.write(f"Reply for {key}:")
                st.write(response["output_text"])
                print(response["output_text"])
                create_visualizations(response["output_text"])
                print(extract_criteria_and_values(response["output_text"]))


if __name__ == "__main__":
    main()