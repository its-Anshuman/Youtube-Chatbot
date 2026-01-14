import os
from dotenv import load_dotenv
 
load_dotenv()
 
import streamlit as st
 
from backend import (
    extract_video_id,
    get_transcript,
    build_vector_store_from_transcript,
    build_qa_chain,
)
 
 
def main():
    st.set_page_config(page_title="YouTube Video Chatbot", page_icon="🎥", layout="wide")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "qa_chain" not in st.session_state:
        st.session_state["qa_chain"] = None
    if "video_id" not in st.session_state:
        st.session_state["video_id"] = None
 
    st.title("🎥 YouTube Video Chatbot")
    st.write("Process a YouTube video in the sidebar, then chat with it here.")
 
    #Sidebar for video input
    with st.sidebar:
        st.header("Video")
        youtube_input = st.text_input(
            "YouTube URL or Video ID",
            value="",
            placeholder="Paste full URL or just ID",
        )
 
        if st.button("Process Video"):
            if not youtube_input.strip():
                st.error("Please enter a YouTube URL or video ID.")
            else:
                video_id = extract_video_id(youtube_input)
                st.session_state["video_id"] = video_id
 
                with st.spinner(f"Fetching transcript and building index for video: {video_id}"):
                    try:
                        transcript = get_transcript(video_id)
                    except RuntimeError as e:
                        st.error(str(e))
                    else:
                        # Build vector store & chain
                        vector_store = build_vector_store_from_transcript(transcript)
                        qa_chain, retriever = build_qa_chain(vector_store)
                        st.session_state["qa_chain"] = qa_chain
 
                        st.success("✅ Video processed successfully. You can now ask questions.")
                        
                        # Add system message
                        st.session_state["messages"].append(
                            {
                                "role": "assistant",
                                "content": "Video processed successfully. Ask me anything about it!",
                            }
                        )
 
        if st.session_state.get("video_id"):
            st.caption(f"Current video ID: `{st.session_state['video_id']}`")
 
    #CSS for message alignment
    st.markdown(
        """
<style>
.user-msg {
    padding: 10px 14px;
    border-radius: 12px;
    margin: 8px;
    max-width: 70%;
    float: right;
    clear: both;
    text-align: left;
    box-shadow: 0px 1px 4px rgba(0,0,0,0.1);
}
.assistant-msg {
    padding: 10px 14px;
    border-radius: 12px;
    margin: 8px;
    max-width: 70%;
    float: left;
    clear: both;
    text-align: left;
    box-shadow: 0px 1px 4px rgba(0,0,0,0.1);
}
.chat-container {
    width: 100%;
    overflow: auto;
}
</style>
""",
        unsafe_allow_html=True,
    )
 
    #Display message history
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state["messages"]:
        role = msg["role"]
        content = msg["content"]
 
        if role == "user":
            st.markdown(f"<div class='user-msg'>{content}</div>", unsafe_allow_html=True)
        else:  # assistant (or system-like)
            st.markdown(f"<div class='assistant-msg'>{content}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
 
    # ---------- Chat input at bottom ----------
    user_input = st.chat_input("Ask a question about the video...")
    if user_input:
        # Add user message to history
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.markdown(f"<div class='user-msg'>{user_input}</div>", unsafe_allow_html=True)
 
        # Check if video is processed
        if st.session_state["qa_chain"] is None:
            assistant_reply = "Please process a video first from the sidebar, then ask your question."
        else:
            qa_chain = st.session_state["qa_chain"]
            with st.spinner("Thinking..."):
                try:
                    answer = qa_chain.invoke(user_input)
                    assistant_reply = answer
                except Exception as e:
                    assistant_reply = f"Error while generating answer: {e}"
 
        # Show assistant reply and save to history
        st.markdown(f"<div class='assistant-msg'>{assistant_reply}</div>", unsafe_allow_html=True)
        st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})
 
 
if __name__ == "__main__":
    main()