# === Imports & Config ===
import streamlit as st
import os
import re
import json
import pdfplumber
from scripts.text_extract import chunk_text_data
from scripts.combined import process_full_exam_with_metadata
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import tempfile
import threading
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# === Page Configuration ===
st.set_page_config(
    page_title="QPaper Chatbot", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Hide Streamlit Default Elements ===
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display:none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


st.title("📄 QPaper Chatbot")

# === Initialize Session State ===
if "files_processed" not in st.session_state:
    st.session_state.files_processed = False
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []
if "combined_data" not in st.session_state:
    st.session_state.combined_data = []
if "cleaned_data" not in st.session_state:
    st.session_state.cleaned_data = None
if "dashboard_processing" not in st.session_state:
    st.session_state.dashboard_processing = False
if "dashboard_ready" not in st.session_state:
    st.session_state.dashboard_ready = False

# === Background Processing Function ===
def process_dashboard_data_if_needed():
    """Process dashboard data if needed and not already processed"""
    if (st.session_state.files_processed and 
        not st.session_state.dashboard_processing and 
        not st.session_state.dashboard_ready and 
        st.session_state.combined_data):
        
        print("🔄 Starting dashboard data processing...")  # Debug print
        st.session_state.dashboard_processing = True
        
        try:
            # Show processing message
            with st.spinner("📊 Processing dashboard data in background..."):
                print("📊 Processing dashboard data...")  # Debug print
                cleaned_data = process_full_exam_with_metadata(st.session_state.combined_data)
                print("✅ Dashboard data processed successfully!")  # Debug print
                
                # Store in session state
                st.session_state.cleaned_data = cleaned_data
                st.session_state.dashboard_ready = True
                st.session_state.dashboard_processing = False
                
        except Exception as e:
            print(f"❌ Error processing dashboard data: {str(e)}")  # Debug print
            st.session_state.dashboard_processing = False
            st.error(f"Error processing dashboard data: {str(e)}")
            return False
        
        return True
    return False

# === Function to Process Files ===
def process_uploaded_files(uploaded_files):
    """Process uploaded files and store data in session state"""
    combined_text = ""
    combined_data = []
    
    os.makedirs("data", exist_ok=True)
    global_page_counter = 1

    for uploaded_file in uploaded_files:
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(f"🔍 Extracting from {uploaded_file.name}..."):
            with pdfplumber.open(file_path) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text:
                        combined_text += text
                        cleaned_text = re.sub(r'\(cid:\d+\)', ' ', text)
                        combined_data.append({
                            "text": cleaned_text,
                            "page": global_page_counter,
                            "source": uploaded_file.name
                        })
                        global_page_counter += 1

    # Save to files
    with open("file.txt", "w", encoding="utf-8") as file:
        for item in combined_data:
            file.write(f"--- Page {item['page']} from {item['source']} ---\n")
            file.write(item["text"] + "\n\n")

    with st.spinner("🔗 Splitting text into 5–6 line chunks..."):
        chunks = chunk_text_data(combined_data, lines_per_chunk=6)
        with open("chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    # Store in session state
    st.session_state.combined_data = combined_data
    st.session_state.files_processed = True
    st.session_state.uploaded_file_names = [f.name for f in uploaded_files]
    
    # Reset dashboard state when new files are uploaded
    st.session_state.cleaned_data = None
    st.session_state.dashboard_ready = False
    st.session_state.dashboard_processing = False
    
    # Clear chat history when new files are uploaded
    st.session_state.messages = []
    
    print("📁 Files processed, dashboard processing will start automatically...")  # Debug print
    
    return combined_data

# === Sidebar: PDF Upload ===
st.sidebar.header("📂 Upload Question Papers")

# Show currently loaded files if any
if st.session_state.files_processed and st.session_state.uploaded_file_names:
    st.sidebar.info(f"📁 Currently loaded: {', '.join(st.session_state.uploaded_file_names)}")
    
    # Add clear button
    if st.sidebar.button("🗑️ Clear All Files"):
        st.session_state.files_processed = False
        st.session_state.uploaded_file_names = []
        st.session_state.combined_data = []
        st.session_state.cleaned_data = None
        st.session_state.dashboard_ready = False
        st.session_state.dashboard_processing = False
        # Clear chat history when files are cleared
        st.session_state.messages = []
        # Clean up files
        if os.path.exists("file.txt"):
            os.remove("file.txt")
        if os.path.exists("chunks.json"):
            os.remove("chunks.json")
        st.rerun()

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# Process files only if new files are uploaded or no files are processed yet
if uploaded_files and (not st.session_state.files_processed or 
                      set([f.name for f in uploaded_files]) != set(st.session_state.uploaded_file_names)):
    combined_data = process_uploaded_files(uploaded_files)
    st.sidebar.success("✅ All files uploaded, text extracted, and chunks saved successfully.")
elif st.session_state.files_processed:
    # Use existing data from session state
    combined_data = st.session_state.combined_data

# Dashboard Button - only show if files are processed
if st.session_state.files_processed:
    # Try to process dashboard data if needed
    if process_dashboard_data_if_needed():
        st.rerun()  # Refresh to show updated status
    
    # Show dashboard processing status
    if st.session_state.dashboard_processing:
        st.sidebar.info("⏳ Processing dashboard data...")
        st.sidebar.button("📊 View Dashboard", disabled=True)
    elif st.session_state.dashboard_ready:
        st.sidebar.success("✅ Dashboard data ready!")
        if st.sidebar.button("📊 View Dashboard"):
            st.switch_page("pages/dashboard.py")
    else:
        st.sidebar.warning("⚠️ Dashboard data will be ready soon...")
        if st.sidebar.button("📊 View Dashboard"):
            # This will trigger processing on next run
            st.rerun()

# === Load Chunks & Prepare Documents ===
if os.path.exists("chunks.json") and st.session_state.files_processed:
    with open("chunks.json", "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    documents = [
        Document(page_content=chunk["content"], metadata=chunk["metadata"])
        for chunk in raw_chunks
    ]

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Always create a fresh temporary database for new uploads
    persist_directory = tempfile.mkdtemp()
    db = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    
    retriever = db.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
You are an intelligent and concise assistant. Answer the following question using **only** the information provided in the <context> section.

<context>
{context}
</context>

if you are returning the questions then number them as 1,2 not acc to context
If the answer is not explicitly present in the context, respond with:
"I couldn't find this information in the uploaded question papers."

Now, here is the question:
{input}
""")

    selection_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. The user has a question and you are given 5 chunks of context.

Your job is to return **only those chunks** which are relevant for answering the question.
If none of them are relevant, return an empty response.
important:- if more than one chunk are relevant return all of them

return relevant chunks like these:-
Chunk (Page 2, Chunk 18, Source: UCS414 (1).PDF)
Chunk (Page 4, Chunk 35, Source: UCS414.PDF)

<context>
{context}
</context>

Question: {input}
""")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    selector_chain = create_stuff_documents_chain(llm, selection_prompt)

    # === Chat Interface ===
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input at bottom
    user_query = st.chat_input("Ask something about the papers...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("🤖 Thinking..."):
            # Step 1: Get top 5 most similar chunks
            top_5_docs = db.similarity_search(user_query, k=5)

            # Step 2: Format for selection prompt
            formatted_docs = []
            for doc in top_5_docs:
                chunk_number = doc.metadata["chunk_number"]
                page_number = doc.metadata["page_number"]
                source = doc.metadata["source"]

                # Generate chunk ID used by both Gemini and your matcher
                chunk_id = f"Chunk (Page {page_number}, Chunk {chunk_number}, Source: {source})"
                doc.metadata["chunk_id"] = chunk_id  # store for later

                # Format chunk as Gemini sees it
                formatted_docs.append(
                    Document(
                        page_content=f"{chunk_id}:\n{doc.page_content}",
                        metadata=doc.metadata
                    )
                )

            # Step 3: Ask Gemini which chunks are relevant
            selection_result = selector_chain.invoke({
                "context": formatted_docs,
                "input": user_query
            })

            selected_chunks_text = selection_result.strip()

            # Step 4: Match selected chunks from original top_5
            relevant_chunks = []
            for doc in top_5_docs:
                if doc.metadata["chunk_id"] in selected_chunks_text:
                    relevant_chunks.append(doc)

            # Step 5: Expand with neighbor logic if relevant chunks found
            final_context_chunks = []

            for selected_doc in relevant_chunks:
                selected_chunk_num = selected_doc.metadata["chunk_number"]
                selected_source = selected_doc.metadata["source"]

                for doc in documents:
                    meta = doc.metadata
                    if meta["source"] == selected_source and meta["chunk_number"] in [selected_chunk_num - 1, selected_chunk_num, selected_chunk_num + 1]:
                        final_context_chunks.append(doc)

            # Step 6: Remove duplicates by chunk_number + source
            unique_keys = set()
            unique_context = []
            for doc in final_context_chunks:
                key = (doc.metadata["source"], doc.metadata["chunk_number"])
                if key not in unique_keys:
                    unique_keys.add(key)
                    unique_context.append(doc)

            # Step 7: Final answer
            if unique_context:
                final_answer = doc_chain.invoke({
                    "context": unique_context,
                    "input": user_query
                })
            else:
                final_answer = "I couldn't find this information in the uploaded question papers."

        with st.chat_message("ai"):
            st.markdown(final_answer)

        st.session_state.messages.append({"role": "ai", "content": final_answer})

elif not st.session_state.files_processed:
    st.info("👆 Please upload PDF files using the sidebar to start chatting!")
else:
    st.info("Processing files... Please wait.")