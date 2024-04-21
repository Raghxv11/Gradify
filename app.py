### STUDENT AUTOGRADER
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
#import streamlit as st
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

load_dotenv()  ## load all the environment variables

#import streamlit as st
import os
import google.generativeai as genai
from PIL import Image

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_repsonse(input, image, prompt):
    model = genai.GenerativeModel("gemini-pro-vision")
    response = model.generate_content([input, image[0], prompt])
    return response.text

def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.mimetype,  # Get the mime type of the uploaded file
                "data": bytes_data,
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
    

#initialize our streamlit app

# st.set_page_config(page_title="Automate Grading with Gemini")

# st.header("Automate Grading with Gemini")
# input = st.text_input("Question: ", key="input")
# uploaded_file = st.file_uploader("Upload Student Solution...", type=["jpg", "jpeg", "png", "pdf"])
# image = ""
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image.", use_column_width=True)


# submit = st.button("Grade")    

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
Student's solution:
```
student's solution here
```
Actual solution:
```
steps to work out the solution and your solution here
```
Is the student's solution the same as actual solution \
just calculated:
```
yes or no
```
Student grade:
```
correct or incorrect

"""

#If submit button is clicked

# if submit:
#     image_data = input_image_setup(uploaded_file)
#     response = get_gemini_repsonse(input_prompt, image_data, input)
#     st.subheader("The Response is")
#     st.write(response)

from flask import Flask, request, render_template, jsonify
from PIL import Image

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            image = Image.open(file)
            image_data = input_image_setup(file)
            response = get_gemini_repsonse(input_prompt, image_data, request.form['question'])
            return render_template('index.html', response=response)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)