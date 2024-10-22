
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

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
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


def get_conversational_chain(rubric=None):
    if rubric:
        rubric_text = f" according to the provided rubric:\n{rubric}\n"
    else:
        rubric_text = " based on the general grading criteria.\n"
    
    prompt_template = f"""
    You are a trained expert on writing and literary analysis. Your job is to accurately and effectively grade a student's essay{rubric_text}
    Respond back with graded points and a level for each criteria. Don't rewrite the rubric in order to save processing power. In the end, write short feedback about what steps they might take to improve on their assignment. Write a total percentage grade and letter grade. In your overall response, try to be lenient and keep in mind that the student is still learning. While grading the essay remember the writing level the student is at while considering their course level, grade level, and the overall expectations of writing should be producing.
    Your grade should only be below 70 percent if the essay does not succeed at all in any of the criteria. Your grade should only be below 80 percent if the essay is not sufficient in most of the criteria. Your grade should only be below 90% if there are a few criteria where the essay doesn't excell. Your grade should only be above 90 percent if the essay succeeds in most of the criteria.
    Understand that the essay was written by a human and think about their writing expectations for their grade level/course level, be lenient and give the student the benefit of the doubt.

        Context:\n {{context}}?\n
        Question: \n{{question}}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
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

    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.header("Automate grading using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

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
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                rubric_text = get_pdf_text([rubric_doc]) if rubric_doc else None
                chain = get_conversational_chain(rubric=rubric_text)
                if user_question:
                    response = chain({"input_documents": text_chunks, "question": user_question}, return_only_outputs=True)
                    st.write("Reply: ", response["output_text"])
                st.success("Done")


if __name__ == "__main__":
    main()
