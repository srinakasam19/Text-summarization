# --------------------------------------------------------
# 🤖 YouTube & Website Summarizer App (Fixed Map-Reduce + .env)
# --------------------------------------------------------

import os
from dotenv import load_dotenv   # To load API key from .env file
import streamlit as st           # For building the web app
import re                        # Regex for extracting YouTube video ID
import requests                  # To fetch website content
from bs4 import BeautifulSoup    # For parsing HTML content
from youtube_transcript_api import YouTubeTranscriptApi  # Fetch YouTube transcripts
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain

# -----------------------------
# Load .env file for GROQ API Key
# -----------------------------
load_dotenv()  # Make sure your .env file contains GROQ_API_KEY=<your_key>
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found! Please set it in your .env file.")
    st.stop()

# -----------------------------
# App Title
# -----------------------------
st.title("🤖 YouTube & Website Summarizer")
st.caption("Powered by Groq (Llama-3.1-8B) + LangChain")

# -----------------------------
# Input URL
# -----------------------------
url = st.text_input("Enter YouTube or Website URL:")

# -----------------------------
# Helper Function: Extract YouTube Video ID
# -----------------------------
def extract_video_id(url):
    """
    Extracts the 11-character video ID from a YouTube URL.
    Supports both standard (v=VIDEO_ID) and shortened (youtu.be/VIDEO_ID) formats.
    """
    pattern = r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

# -----------------------------
# Summarize Button Logic
# -----------------------------
if st.button("Summarize"):
    if not url:
        st.error("Please enter a URL")
    else:
        try:
            # -----------------------------
            # Initialize Groq LLM
            # -----------------------------
            llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)

            # -----------------------------
            # Case 1: YouTube Video
            # -----------------------------
            video_id = extract_video_id(url)
            if video_id:
                # Fetch transcript from YouTube
                transcript_list = YouTubeTranscriptApi().fetch(video_id)
                transcript_text = " ".join([t.text for t in transcript_list])

                # Convert transcript into a LangChain Document
                doc = Document(page_content=transcript_text)

                # -----------------------------
                # Map-Reduce Summarization Technique
                # -----------------------------
                # 1. Map step: Split transcript into chunks and summarize each chunk
                map_prompt = PromptTemplate(
                    input_variables=["text"],
                    template="Summarize this part of the transcript in simple terms:\n\n{text}"
                )

                # 2. Reduce step: Combine all chunk summaries into one final summary
                combine_prompt = PromptTemplate(
                    input_variables=["text"],
                    template="Combine the following partial summaries into a final concise summary:\n\n{text}"
                )

                # 3. Use load_summarize_chain to create a Map-Reduce summarization chain
                # Text Summarization Technique: Abstractive Summarization
                # The LLM reads the text and generates a concise summary in natural language,
                # rather than copying sentences verbatim (extractive summarization).
                chain = load_summarize_chain(
                    llm,
                    chain_type="map_reduce",
                    map_prompt=map_prompt,
                    combine_prompt=combine_prompt
                )

                with st.spinner("Summarizing long transcript using Map-Reduce... 🤖"):
                    summary = chain.run([doc])

                st.subheader("YouTube Summary:")
                st.write(summary)

            # -----------------------------
            # Case 2: Regular Website
            # -----------------------------
            else:
                response = requests.get(url)
                soup = BeautifulSoup(response.content, "html.parser")

                # Attempt to extract main content
                main_content = soup.find('main')
                if main_content:
                    text = main_content.get_text(separator=' ', strip=True)
                else:
                    # Fallback: get all visible text
                    text = " ".join(soup.stripped_strings)

                # Single-step summarization for website text
                prompt = PromptTemplate(
                    input_variables=["text"],
                    template="Summarize the following website content in simple terms:\n\n{text}"
                )
                chain = LLMChain(llm=llm, prompt=prompt)

                with st.spinner("Summarizing website content... 🤖"):
                    summary = chain.run(text)

                st.subheader("Website Summary:")
                st.write(summary)

        except Exception as e:
            st.error(f"Failed to summarize content: {e}")
