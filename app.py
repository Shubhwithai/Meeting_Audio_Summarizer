import streamlit as st
from openai import OpenAI
from pathlib import Path
import tempfile

# Initialize OpenAI client
client = OpenAI()

def save_uploaded_file(uploaded_file):
    """Save uploaded file temporarily and return the path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name

def transcribe_audio(audio_path):
    """Transcribe audio file using OpenAI Whisper"""
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    return transcript

def summarize_transcript(transcript):
    """Summarize transcript using GPT-4"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"Please summarize the following meeting transcript into 5 key bullet points:\n\n{transcript}"
            }
        ]
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("Meeting Summarizer")
st.write("Upload an audio recording to get a summary")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'm4a'])

if uploaded_file:
    # Save uploaded file
    temp_path = save_uploaded_file(uploaded_file)
    
    try:
        # Get transcript
        st.subheader("Transcript")
        transcript = transcribe_audio(temp_path)
        st.text_area("Full Transcript", transcript, height=200)
        
        # Get summary
        st.subheader("Summary")
        summary = summarize_transcript(transcript)
        st.write(summary)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
    finally:
        # Clean up temporary file
        Path(temp_path).unlink()
