import streamlit as st
import google.generativeai as genai
import pandas as pd
import docx
import PyPDF2
from PIL import Image
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load environment variables from .env file
load_dotenv()

# Configure Gemini
api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')
vision_model = genai.GenerativeModel('gemini-1.5-flash')

# Helper for chat-style layout
def display_message(message, is_user):
    """
    Helper to render user and bot messages with custom styles in Streamlit.
    """
    if is_user:
        st.markdown(
            f"<div style='text-align: right; color: white; background: #1e90ff; padding: 10px; "
            f"border-radius: 10px; margin: 5px;'>{message}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='text-align: left; color: black; background: #f0f0f0; padding: 10px; "
            f"border-radius: 10px; margin: 5px;'>{message}</div>",
            unsafe_allow_html=True
        )

# File processing functions
def extract_text_based_file(uploaded_file):
    """
    Extract content from text-based files: txt, pdf, docx, csv, xlsx.
    Returns a string if textual data,
    or a Pandas DataFrame if CSV/XLSX data,
    or None if something went wrong.
    """
    content = None
    file_type = uploaded_file.type

    try:
        if file_type == "text/plain":  # TXT file
            content = uploaded_file.getvalue().decode("utf-8")

        elif file_type == "application/pdf":  # PDF file
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            pdf_text = []
            for page in pdf_reader.pages:
                pdf_text.append(page.extract_text())
            content = "\n".join(pdf_text)

        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":  # DOCX
            doc = docx.Document(uploaded_file)
            paragraphs = [p.text for p in doc.paragraphs]
            content = "\n".join(paragraphs)

        elif file_type in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            # CSV or XLSX
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            content = df  # we'll store a DataFrame in this variable

        else:
            content = "Unsupported file type."

    except Exception as e:
        content = f"Error processing file: {str(e)}"

    return content

def create_visualization(df, x_axis=None, y_axis=None, chart_type=None):
    """
    Create a Matplotlib/Seaborn plot directly in Streamlit.
    """
    st.subheader("Data Visualization")
    if not x_axis or not chart_type:
        st.error("Insufficient information to create a plot. Provide x_axis and chart_type at least.")
        return

    plt.figure(figsize=(8, 5))
    try:
        if chart_type.lower() == "line":
            sns.lineplot(data=df, x=x_axis, y=y_axis)
        elif chart_type.lower() == "bar":
            sns.barplot(data=df, x=x_axis, y=y_axis)
        elif chart_type.lower() == "scatter":
            sns.scatterplot(data=df, x=x_axis, y=y_axis)
        elif chart_type.lower() == "pie":
            # For a pie chart, typically we only have x_axis as the label column
            df_grouped = df.groupby(x_axis).size().reset_index(name='Count')
            plt.pie(df_grouped['Count'], labels=df_grouped[x_axis], autopct='%1.1f%%')
        elif chart_type.lower() == "histogram":
            sns.histplot(df[x_axis], kde=True)
        else:
            st.warning(f"Chart type '{chart_type}' not recognized. Supported: line, bar, scatter, pie, histogram.")
            return

        # Render the plot
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")

# Main App
def main():
    st.set_page_config(page_title="Chat with Any File", layout="wide")
    st.title("📁 Chat with Any File")
    
    # Sidebar for file upload
    with st.sidebar:
        st.subheader("Upload your file")
        uploaded_file = st.file_uploader(
            label="Upload File",
            type=['txt', 'docx', 'pdf', 'csv', 'xlsx', 'png', 'jpg', 'jpeg']
        )

    # Session state initialization
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # We'll keep the content of the uploaded file in session state so we only process once
    if "file_content" not in st.session_state:
        st.session_state["file_content"] = None
    if "df" not in st.session_state:
        st.session_state["df"] = None  # store DataFrame if CSV or XLSX

    # If a new file is uploaded, read it and store content
    if uploaded_file is not None:
        file_type = uploaded_file.type

        # If it's an image file, store it directly
        if file_type in ["image/png", "image/jpeg"]:
            # Store the image in session
            st.session_state["file_content"] = Image.open(uploaded_file)
            st.session_state["df"] = None
        else:
            # It's text-based
            extracted = extract_text_based_file(uploaded_file)
            if isinstance(extracted, pd.DataFrame):
                st.session_state["df"] = extracted
                st.session_state["file_content"] = None
            else:
                st.session_state["file_content"] = extracted
                st.session_state["df"] = None

    # Container to show the conversation
    chat_container = st.container()

    # Display any past messages
    with chat_container:
        for msg, is_user in st.session_state["messages"]:
            display_message(msg, is_user)

    # Chat input
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Show user message
        st.session_state["messages"].append((user_input, True))
        with chat_container:
            display_message(user_input, True)

        # If no file uploaded, do a standard text model generation
        if uploaded_file is None:
            prompt = f"User Question: {user_input}"
            response = model.generate_content(prompt)
            bot_response = response.text.strip()

            st.session_state["messages"].append((bot_response, False))
            with chat_container:
                display_message(bot_response, False)

        else:
            # We do have an uploaded file
            try:
                # If image file
                if uploaded_file.type in ["image/png", "image/jpeg"]:
                    image = st.session_state["file_content"]
                    prompt = f"{user_input}"
                    response = vision_model.generate_content([prompt, image])
                    bot_response = response.text.strip()

                    st.session_state["messages"].append((bot_response, False))
                    with chat_container:
                        display_message(bot_response, False)

                else:
                    # Text-based or DataFrame-based
                    if st.session_state["df"] is not None:
                        # We have a DataFrame
                        df = st.session_state["df"]

                        prompt = (f"We have a DataFrame with the following columns: {', '.join(df.columns)}.\n"
                                  f"User request: {user_input}\n"
                                  "If the user wants to create a plot, reply in JSON format only: "
                                  '{"chart_type": "<Bar/Scatter/Line/etc>", "x_axis": "<col>", "y_axis": "<col or None>"}.\n'
                                  "If the user does not want a plot, answer normally in plain text.")
                        
                        response = model.generate_content(prompt)
                        bot_response = response.text.strip()
                        
                        try:
                            plot_params = json.loads(bot_response)
                            chart_type = plot_params.get("chart_type")
                            x_axis = plot_params.get("x_axis")
                            y_axis = plot_params.get("y_axis", None)

                            st.session_state["messages"].append(("Creating your plot...", False))
                            with chat_container:
                                display_message("Creating your plot...", False)

                            create_visualization(df, x_axis=x_axis, y_axis=y_axis, chart_type=chart_type)

                        except json.JSONDecodeError:
                            st.session_state["messages"].append((bot_response, False))
                            with chat_container:
                                display_message(bot_response, False)

                    else:
                        file_text = st.session_state["file_content"]
                        snippet = file_text[:1000] if len(file_text) > 1000 else file_text

                        prompt = f"Context: {snippet}\nUser Question: {user_input}"
                        response = model.generate_content(prompt)
                        bot_response = response.text.strip()

                        st.session_state["messages"].append((bot_response, False))
                        with chat_container:
                            display_message(bot_response, False)

            except Exception as e:
                error_msg = f"Error processing file or generating response: {str(e)}"
                st.session_state["messages"].append((error_msg, False))
                with chat_container:
                    display_message(error_msg, False)

if __name__ == "__main__":
    main()
