# 📄 QPaper Chatbot

An intelligent chatbot application that processes PDF question papers and provides interactive Q&A capabilities with dashboard analytics. Built with Streamlit, LangChain, and Google Gemini AI.

## ✨ Features

- **PDF Processing**: Upload multiple PDF question papers for analysis
- **Smart Text Extraction**: Extracts and chunks text data from PDFs
- **AI-Powered Chat**: Interactive chatbot using Google Gemini for answering questions
- **Intelligent Retrieval**: Context-aware document retrieval with neighbor chunk expansion
- **Dashboard Analytics**: Visual insights into topics frequency and difficulty distribution
- **Multi-file Support**: Process multiple PDFs simultaneously
- **Persistent Sessions**: Maintains chat history and file state

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API Key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd qpaper-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run chatai.py
   ```

## 📦 Dependencies

```txt
streamlit
pdfplumber
langchain
langchain-community
langchain-google-genai
langchain-core
chromadb
google-generativeai
python-dotenv
plotly
pandas
```

## 🏗️ Project Structure

```
qpaper-chatbot/
├── chatai.py                 # Main application file
├── pages/
│   └── dashboard.py          # Dashboard analytics page
├── scripts/
│   ├── text_extract.py       # Text chunking utilities
│   └── combined.py           # Data processing utilities
├── data/                     # Temporary PDF storage
├── .env                      # Environment variables
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 💡 How It Works

### 1. File Processing
- Upload PDF files through the sidebar interface
- Text is extracted using `pdfplumber`
- Content is cleaned and chunked into 5-6 line segments
- Metadata (page numbers, source files) is preserved

### 2. Vector Search & Retrieval
- Text chunks are embedded using Google's `text-embedding-004` model
- Vector database created using ChromaDB
- Smart retrieval with neighbor chunk expansion for better context

### 3. AI Chat Interface
- Two-stage retrieval process:
  1. Get top 5 similar chunks
  2. Use Gemini to select most relevant chunks
- Context expansion includes neighboring chunks
- Answers are generated using only the relevant context

### 4. Dashboard Analytics
- Background processing of uploaded files
- Topic frequency analysis
- Year-wise difficulty distribution charts
- Visual insights using Plotly

## 🎯 Usage

### Uploading Files
1. Use the sidebar "Upload Question Papers" section
2. Select one or more PDF files
3. Files are automatically processed and chunked
4. Chat interface becomes available

### Chatting
1. Type questions in the chat input at the bottom
2. The AI will search through your uploaded papers
3. Responses are based only on the uploaded content
4. Chat history is maintained during the session

### Viewing Dashboard
1. After files are processed, dashboard data is prepared in the background
2. Click "📊 View Dashboard" in the sidebar when ready
3. View topic frequency and difficulty distribution charts

## ⚙️ Configuration

### Embedding Model
The application uses Google's `text-embedding-004` model for creating embeddings.

### Chat Model
Responses are generated using `gemini-2.5-flash` with temperature set to 0.2 for consistent answers.

### Chunking Strategy
- Text is split into 6-line chunks by default
- Neighbor chunks (±1) are included for better context
- Metadata preservation ensures accurate source attribution

## 🔧 Key Components

### Session State Management
- `files_processed`: Track if files have been uploaded
- `combined_data`: Store extracted text data
- `cleaned_data`: Processed data for dashboard
- `messages`: Chat conversation history

### Smart Context Selection
1. **Similarity Search**: Find top 5 most relevant chunks
2. **AI Selection**: Use Gemini to identify truly relevant chunks
3. **Context Expansion**: Include neighboring chunks for completeness
4. **Deduplication**: Remove duplicate chunks from final context

## 🛠️ Customization

### Adjusting Chunk Size
Modify the `lines_per_chunk` parameter in `process_uploaded_files()`:
```python
chunks = chunk_text_data(combined_data, lines_per_chunk=6)  # Change this value
```

### Changing Retrieval Count
Adjust the number of chunks retrieved:
```python
retriever = db.as_retriever(search_kwargs={"k": 3})  # Change k value
```

### Model Configuration
Update the AI models in the initialization:
```python
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
```

## 🚨 Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure your Google Gemini API key is correctly set in the `.env` file
   - Verify the API key has proper permissions

2. **PDF Processing Fails**
   - Check if PDF files are not password-protected
   - Ensure files are valid PDF format

3. **Dashboard Not Loading**
   - Dashboard data processing happens in background
   - Wait for processing to complete before accessing dashboard

4. **Memory Issues**
   - Large PDF files may cause memory issues
   - Consider processing fewer files at once

### Debug Mode
The application includes debug prints. Check the console for detailed processing information.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Google Gemini AI](https://ai.google.dev/)
- Uses [LangChain](https://langchain.com/) for document processing
- PDF processing with [pdfplumber](https://github.com/jsvine/pdfplumber)
- Vector storage with [ChromaDB](https://www.trychroma.com/)

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Create an issue on GitHub
3. Review the code documentation

---

**Note**: Make sure to keep your API keys secure and never commit them to version control.
