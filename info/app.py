# --------------------------------------------------------
# 🤖 YouTube & Website Summarizer App
# --------------------------------------------------------

# Streamlit for UI
import streamlit as st
# Regular expressions for extracting YouTube video ID
import re
# Requests for fetching website HTML content
import requests
# BeautifulSoup for parsing HTML
from bs4 import BeautifulSoup
# YouTube transcript API to fetch video transcripts
from youtube_transcript_api import YouTubeTranscriptApi
# LangChain components for LLM summarization
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

# -----------------------------
# App Title and Description
# -----------------------------
st.title("🤖 YouTube & Website Summarizer")
st.caption("Powered by Groq (Llama-3.1-8B) + LangChain")

# -----------------------------
# User Input: YouTube or Website URL
# -----------------------------
url = st.text_input("Enter YouTube or Website URL:")

# -----------------------------
# Helper Function: Extract YouTube Video ID
# -----------------------------
def extract_video_id(url):
    """
    Extracts the 11-character YouTube video ID from a URL.
    Works with both standard (v=VIDEO_ID) and shortened (youtu.be/VIDEO_ID) URLs.
    Returns the video ID string if found, else None.
    """
    pattern = r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

# -----------------------------
# Main Summarization Logic Triggered by Button
# -----------------------------
if st.button("Summarize"):
    # Check if user has entered a URL
    if not url:
        st.error("Please enter a URL")
    else:
        try:
            # -----------------------------
            # Case 1: YouTube Video
            # -----------------------------
            video_id = extract_video_id(url)
            if video_id:
                # Fetch transcript using the latest YouTubeTranscriptApi
                transcript_list = YouTubeTranscriptApi().fetch(video_id)
                # Join all transcript snippets into a single string
                text = " ".join([t.text for t in transcript_list])
            else:
                # -----------------------------
                # Case 2: Regular Website
                # -----------------------------
                # Fetch HTML content of the page
                response = requests.get(url)
                soup = BeautifulSoup(response.content, "html.parser")
                
                # Try to extract main content if <main> tag exists
                main_content = soup.find('main')
                if main_content:
                    # Get all visible text inside <main>
                    text = main_content.get_text(separator=' ', strip=True)
                else:
                    # Fallback: get all visible text from the page
                    text = " ".join(soup.stripped_strings)

            # -----------------------------
            # Initialize Groq LLM
            # -----------------------------
            llm = ChatGroq(model="llama-3.1-8b-instant")

            # -----------------------------
            # Define Summarization Prompt
            # -----------------------------
            prompt = PromptTemplate(
                input_variables=["text"],
                template="Summarize the following content in simple terms:\n\n{text}"
            )

            # Create a LangChain LLMChain for summarization
            chain = LLMChain(llm=llm, prompt=prompt)

            # -----------------------------
            # Generate Summary with Spinner
            # -----------------------------
            with st.spinner("Summarizing... 🤖"):
                summary = chain.run(text)

            # Display the summary in Streamlit
            st.subheader("Summary:")
            st.write(summary)

        except Exception as e:
            # Handle any errors and show them in the app
            st.error(f"Failed to summarize content: {e}")
