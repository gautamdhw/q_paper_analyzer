import streamlit as st
import json
import re
import plotly.express as px
import streamlit as st
import os
import sys

# Add the parent directory to the path so we can import from main.py
sys.path.append('..')

# Your existing CSS for hiding basic elements
hide_basic_elements = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display:none;}
</style>
"""
st.markdown(hide_basic_elements, unsafe_allow_html=True)

# === Recreate the same sidebar from main app ===
st.sidebar.header("📂 Upload Question Papers")

# Show currently loaded files if any
if st.session_state.get("files_processed", False) and st.session_state.get("uploaded_file_names", []):
    st.sidebar.info(f"📁 Currently loaded: {', '.join(st.session_state.uploaded_file_names)}")
    
    # Add clear button
    if st.sidebar.button("🗑️ Clear All Files"):
        st.session_state.files_processed = False
        st.session_state.uploaded_file_names = []
        st.session_state.combined_data = []
        st.session_state.cleaned_data = None
        st.session_state.dashboard_ready = False
        st.session_state.dashboard_processing = False
        st.session_state.messages = []
        # Clean up files
        if os.path.exists("../file.txt"):  # Note the .. for parent directory
            os.remove("../file.txt")
        if os.path.exists("../chunks.json"):
            os.remove("../chunks.json")
        st.rerun()

# File uploader
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# Process files - import from parent directory
if uploaded_files and (not st.session_state.get("files_processed", False) or 
                      set([f.name for f in uploaded_files]) != set(st.session_state.get("uploaded_file_names", []))):
    
    try:
        # Import from main.py in parent directory
        import chatai
        combined_data = chatai.process_uploaded_files(uploaded_files)
        st.sidebar.success("✅ All files uploaded, text extracted, and chunks saved successfully.")
    except ImportError as e:
        st.sidebar.error(f"Error importing main.py: {e}")

# Back to Chat button - navigate to main page
if st.sidebar.button("💬 Back to Chat"):
    st.switch_page("chatai.py")  # This should work for root directory files


def extract_clean_json(text: str) -> dict:
    """
    Extract and clean JSON content from a messy string.
    """
    # Remove the markdown code block if present
    cleaned = re.sub(r"^.*?```json", "", text, flags=re.DOTALL)  # Remove text before ```json
    cleaned = re.sub(r"```.*$", "", cleaned, flags=re.DOTALL)    # Remove text after ending ```
    
    # Strip any extra whitespace
    cleaned = cleaned.strip()   
    # Parse JSON
    try:
        data = cleaned
        return data
    except json.JSONDecodeError as e:
        print("JSON decoding failed:", e)
        return None
    

def show_dashboard():
    st.title("📈 Dashboard")

    if "cleaned_data" in st.session_state and st.session_state.cleaned_data:
        cleaned = extract_clean_json(st.session_state.cleaned_data)
        cleaned = json.loads(cleaned)
        topic_data = cleaned.get("topics_frequency", [])

        st.subheader("📚 Frequent Topics")

        if topic_data:
            # Sort and limit to top 30
            topic_data = sorted(topic_data, key=lambda x: x["frequency"], reverse=True)[:15]

            # Create 3 columns
            cols = st.columns(3)

            # Loop through topics and assign to columns in round robin
            for i, topic in enumerate(topic_data):
                with cols[i % 3]:
                    st.markdown(
                        f"""
                        <div style="background-color: #f0f0f0;
                                    padding: 12px;
                                    margin: 8px 0;
                                    border-radius: 10px;
                                    font-weight: bold;
                                    color: #000000;
                                    border: 1px solid #ccc;">
                            {topic['topic']}<br>
                            <span style="font-weight: normal; color: #333;">Freq: {topic['frequency']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.info("No topics found.")

        # === DIFFICULTY DISTRIBUTION CHART ===
        st.subheader("📊 Year-wise Difficulty Distribution")
        difficulty = cleaned.get("yearwise_difficulty", {})

        if difficulty:
            # Flatten to dataframe-friendly structure
            rows = []
            for year, levels in difficulty.items():
                for level, count in levels.items():
                    rows.append({"Year": year, "Level": level, "Count": count})

            try:
                import pandas as pd
                import plotly.express as px
                df = pd.DataFrame(rows)

                fig2 = px.bar(
                    df,
                    x="Year",
                    y="Count",
                    color="Level",
                    barmode="group",
                    title="Difficulty Distribution by Year"
                )
                st.plotly_chart(fig2, use_container_width=True)
            except:
                st.error("Pandas or Plotly is not installed.")
        else:
            st.info("No difficulty data found.")
    else:
        st.warning("No data found. Please upload and process files first.")


show_dashboard()
