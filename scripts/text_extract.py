import pdfplumber
import re
def chunk_text_data(extracted_data, lines_per_chunk=6):
    chunks = []
    global_chunk=1;
    for entry in extracted_data:
        text = entry["text"]
        page = entry["page"]
        source = entry["source"]

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for i in range(0, len(lines), lines_per_chunk):
            chunk_lines = lines[i:i + lines_per_chunk]
            chunk_text = " ".join(chunk_lines)

            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "page_number": page,
                    "chunk_number": global_chunk,
                    "source": source
                }
            })
            global_chunk+=1

    return chunks


import nltk
from nltk.corpus import stopwords

custom_stopwords = stopwords.words("english")
custom_stopwords += ["give", "question", "questions", "list", "show", "tell"]

def clean_query(query: str) -> str:
    # Lowercase
    query = query.lower()

    # Remove non-ASCII characters (like emojis)
    query = query.encode("ascii", "ignore").decode()

    # Remove punctuation
    query = re.sub(r'[^\w\s]', '', query)

    # Tokenize and remove stopwords
    words = query.split()
    filtered_words = [word for word in words if word not in custom_stopwords]

    # Join back into string
    cleaned = ' '.join(filtered_words)

    return cleaned.strip()