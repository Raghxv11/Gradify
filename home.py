import streamlit as st

# Define the main function to display the home page
def main():
    st.set_page_config(page_title="Automate Grading with Gemini")

    st.title('Assignment Grading App')

    # Option for user to select the input method
    option = st.sidebar.selectbox("Choose the method to submit your question and solution:",
                                  ['Upload Image', 'Upload PDF', 'Type Manually'])

    # Load the corresponding app based on user selection
    if option == 'Upload PDF':
        # Import and run the app that handles image uploads
        import app
        app.main()
    elif option == 'Type Manually':
        # Import and run the app that handles PDF uploads
        import app2
        app2.main()
    elif option == 'Upload Image':
        # Import and run the app that handles typing input
        import app4
        app4.main()

# Run the main function
if __name__ == "__main__":
    main()
