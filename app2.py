import streamlit as st
import os
import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")  # Store API key securely
genai.configure(api_key=API_KEY)


model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])


def get_gemini_response(question, student_solution):
    prompt = f"""
    The student provided solution: {student_solution}.  
    Your task is to determine if the student's solution is correct or not.
    To solve the problem do the following:
    - First, work out your own solution to the problem. 
    - Then compare your solution to the student's solution and evaluate if the student's solution is correct or not. 
    Don't decide if the student's solution is correct until you have done the problem yourself. 
    Use the following format: 
    Question: Question here
    \n
    Student's solution: Student's solution here
    \n
    Actual solution: steps to work out the solution and your solution here
    \n
    Is the student's solution the same as actual solution just calculated: Yes or No
    \n
    Student grade: correct or incorrect
"""
    question = question + "\n" + prompt
    response = chat.send_message(question, stream=True)
    return response



def main():
    st.header("Automate Grading with Gemini")

    input_question = st.text_input("Ask a Question: ")
    student_solution = st.text_input("Student Solution: ")
    submit = st.button("Submit")

    if submit and input_question:
        response = get_gemini_response(input_question, student_solution)
        st.subheader("The response is")
        for chunk in response:
            text = chunk.text
            st.write(text)

if __name__ == "__main__":
    main()
