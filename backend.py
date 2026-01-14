from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, CouldNotRetrieveTranscript
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
 
 
# extract video id
def extract_video_id(url_or_id: str) -> str:
    """Accepts a YouTube URL or a plain video ID and returns the video ID."""
    text = url_or_id.strip()
 
    # If it's already a simple ID (no 'http'), just return it
    if "http" not in text and "www" not in text and "youtu" not in text:
        return text
 
    # Handle full URLs
    parsed = urlparse(text)
 
    # Normal youtube URL: https://www.youtube.com/watch?v=VIDEO_ID
    if "youtube.com" in parsed.netloc:
        query = parse_qs(parsed.query)
        if "v" in query:
            return query["v"][0]
 
    # Short youtu.be URL: https://youtu.be/VIDEO_ID
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")
 
    # Fallback: return original
    return text
 
 
#Step 1: Get transcript
def get_transcript(video_id: str) -> str:
    """
    Fetch the transcript for a given video ID.
    Handles different possible versions / return types from youtube-transcript-api.
    """
    try:
        # 1) Preferred: class method
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id, languages=["en"]
            )
        else:
            # 2) Fallback: instance method or older API
            ytt_api = YouTubeTranscriptApi()
 
            if hasattr(ytt_api, "get_transcript"):
                transcript_list = ytt_api.get_transcript(
                    video_id, languages=["en"]
                )
            elif hasattr(ytt_api, "fetch"):
                transcript_list = ytt_api.fetch(video_id, languages=["en"])
            else:
                raise RuntimeError(
                    "Installed youtube-transcript-api has an unexpected interface. "
                    "Try upgrading it with: pip install -U youtube-transcript-api"
                )
 
        # Handle dict / object return types
        def extract_text(snippet):
            if hasattr(snippet, "text"):
                return snippet.text
            if isinstance(snippet, dict) and "text" in snippet:
                return snippet["text"]
            return str(snippet)
 
        transcript = " ".join(extract_text(snippet) for snippet in transcript_list)
        return transcript
 
    except CouldNotRetrieveTranscript:
        raise RuntimeError(
            "Transcript not available for this video (disabled, no subtitles, or language mismatch)."
        )
 
 
# Step 1b & 1c: Chunk + Build Vector Store
def build_vector_store_from_transcript(transcript: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
 
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding)
    return vector_store
 
 
# Step 2 & 3 & 4: Build Retrieval + QA chain
def build_qa_chain(vector_store: FAISS):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
 
    # Groq Llama 70B model
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # Groq's 70B Llama model
        temperature=0.7,
    )
 
    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant.\n"
            "Answer ONLY using the provided transcript context.\n"
            "If the context is insufficient, just say you don't know.\n\n"
            "Transcript context:\n{context}\n\n"
            "Question: {question}\n"
        ),
        input_variables=["context", "question"],
    )
 
    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)
 
    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )
 
    parser = StrOutputParser()
 
    main_chain = parallel_chain | prompt | llm | parser
    return main_chain, retriever