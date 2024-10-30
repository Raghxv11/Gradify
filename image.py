from dotenv import load_dotenv

load_dotenv()  ## load all the environment variables

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_gemini_response(image, prompt):
    # model = genai.GenerativeModel("gemini-pro-vision")
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([prompt, image[0]])
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

def main():
    st.header("Automate Grading with Gemini")

    # Question input
    uploaded_question = st.file_uploader(
        "Upload an image of the question...",
        type=["jpg", "jpeg", "png"],
        key="question_file"
    )

    # Process question upload
    question_image = None
    if uploaded_question is not None:
        question_image = Image.open(uploaded_question)
        st.image(question_image, caption="Uploaded Question Image.", use_column_width=True)

    # Solution input
    uploaded_solution = st.file_uploader(
        "Upload an image of the student solution...",
        type=["jpg", "jpeg", "png"],
        key="solution_file"
    )

    # Process solution upload
    solution_image = None
    if uploaded_solution is not None:
        solution_image = Image.open(uploaded_solution)
        st.image(solution_image, caption="Uploaded Solution Image.", use_column_width=True)

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

    question here

    \n
    Student's solution:

    student's solution here

    \n
    Actual solution:

    steps to work out the solution and your solution here

    \n
    Is the student's solution the same as actual solution \
    just calculated:

    yes or no

    \n
    Student grade:
    ```
    correct or incorrect
    """
    ## If submit button is clicked

    if submit:
        if uploaded_question is None or uploaded_solution is None:
            if uploaded_question is None:
                st.error("Please upload an image of the question.")
            if uploaded_solution is None:
                st.error("Please upload an image of the student solution.")
        else:
            solution_data = input_image_setup(uploaded_solution) if uploaded_solution else None
            full_prompt = input_prompt.format(
                question_here="The question will be read from uploaded image.",
                student_solution_here="The solution will be read from uploaded image.",
                actual_solution_here="The actual solution will be calculated here."
            )
            response = get_gemini_response(solution_data, full_prompt)
            st.subheader("The Response is")
            st.write(response)

if __name__ == "__main__":
    main()