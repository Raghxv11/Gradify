
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

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


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

# def extract_criteria_and_values(rubric_text):
#         criteria_values = []
#         lines = rubric_text.split('\n')
#         print(lines)
#         i = 0

#         current_criteria = None
#         while i < len(lines):
#             line = lines[i].strip()

#             if line.startswith("**") and line.endswith("**"):
#                 print(line)
#                 while(line.startswith("*") and len(line) == 0):
#                     if(len(line) > 0):
#                         criteria_values.append(line)
#                     i += 1
#                 break

#         return criteria_values

def extract_criteria_and_values(output_text):
    criteria_values = []
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
            text_chunks = get_text_chunks(value)
            get_vector_store(text_chunks)
            rubric_text = get_pdf_text([rubric_doc]) if rubric_doc else None

            if rubric_text:
                for rubric_key in rubric_text:
                    rubric_str = rubric_text[rubric_key]

                rubric_chain = get_rubric_chain()

                response = rubric_chain({"input_documents": convert_text_to_documents([rubric_str])}, return_only_outputs=True)
                rubric_text = response["output_text"]
            #print(rubric_text)
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
