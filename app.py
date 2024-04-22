from dotenv import load_dotenv

load_dotenv()  ## load all the environment variables

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_gemini_response(input_text, image, prompt):
    model = genai.GenerativeModel("gemini-pro-vision")
    response = model.generate_content([prompt, image[0], input_text])
    return response.text


def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data,
            }
        ]
        return image_parts
    else:
        return None


## Initialize our streamlit app

st.set_page_config(page_title="Automate Grading with Gemini")

st.header("Automate Grading with Gemini")

# Question input
input_question_text = st.text_input("Question: ", key="input_question_text")
uploaded_question = st.file_uploader(
    "Upload file with the question (optional, text or image)...",
    type=["jpg", "jpeg", "png", "pdf", "txt", "docx"],
    key="question_file"
)

# Process question upload
question_image = None
if uploaded_question is not None:
    if uploaded_question.type in ["jpg", "jpeg", "png"]:
        question_image = Image.open(uploaded_question)
        st.image(question_image, caption="Uploaded Question Image.", use_column_width=True)
    elif uploaded_question.type in ["txt", "pdf", "docx"]:
        input_question_text = str(uploaded_question.getvalue(), "utf-8")

# Solution input
input_solution_text = st.text_area("Solution Text: ", key="input_solution_text")
uploaded_solution = st.file_uploader(
    "Upload file with the student solution (optional, image or text)...",
    type=["jpg", "jpeg", "png", "pdf", "docx", "txt"],
    key="solution_file"
)

# Process solution upload
solution_image = None
if uploaded_solution is not None:
    if uploaded_solution.type in ["jpg", "jpeg", "png"]:
        solution_image = Image.open(uploaded_solution)
        st.image(solution_image, caption="Uploaded Solution Image.", use_column_width=True)
    elif uploaded_solution.type in ["txt", "pdf", "docx"]:
        input_solution_text = str(uploaded_solution.getvalue(), "utf-8")

submit = st.button("Grade")

input_prompt = """
Your task is to determine if the student's solution \
is correct or not.
To solve the problem do the following:
- First, work out your own solution to the problem. 
- Then compare your solution to the student's solution \ 
and evaluate if the student's solution is correct or not. 
Don't decide if the student's solution is correct until 
you have done the problem yourself.
Use the following format:
Question:
```
question here
```
\n
Student's solution:
```
student's solution here
```
\n
Actual solution:
```
steps to work out the solution and your solution here
```
\n
Is the student's solution the same as actual solution \
just calculated:
```
yes or no
```
\n
Student grade:
```
correct or incorrect

"""

## If submit button is clicked

if submit:
    if (input_question_text.strip() == "" and uploaded_question is None) or (input_solution_text.strip() == "" and uploaded_solution is None):
        if input_question_text.strip() == "" and uploaded_question is None:
            st.error("Please provide the question by typing or uploading a file.")
        if input_solution_text.strip() == "" and uploaded_solution is None:
            st.error("Please provide the solution by typing or uploading a file.")
    else:
        solution_data = input_image_setup(uploaded_solution) if uploaded_solution else None
        full_prompt = input_prompt.format(
            question_here=input_question_text or "No question provided",
            student_solution_here=input_solution_text or "No solution provided",
            actual_solution_here="The actual solution will be calculated here."
        )
        response = get_gemini_response(full_prompt, solution_data, input_question_text)
        st.subheader("The Response is")
        st.write(response)
