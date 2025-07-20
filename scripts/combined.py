import os
import re
import json
import time
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_google_genai import GoogleGenerativeAI

# Load environment variables
load_dotenv()

# Initialize the LangChain Google Generative AI model
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,
    top_p=0.95,
    top_k=40
)

def analyze_paper(paper):
    paper_text = paper["text"]
    source = paper.get("source", "Unknown")

    print(f"🔄 Starting analysis for: {source}")
    start_time = time.time()

    prompt = f"""
Act as a professional exam paper analyzer. Your job is to analyze the following full exam question paper and return:

1. A list of topics with their frequency and conceptual category.
2. Year-wise difficulty distribution with count of Easy, Medium, and Hard questions.

Return only valid JSON in this format:
{{
  "topics_frequency": [
    {{
      "topic": "Topic Name",
      "frequency": Integer,
      "category": "Conceptual Category"
    }},
    ...
  ],
  "yearwise_difficulty": {{
    "2023": {{ "Easy": Integer, "Medium": Integer, "Hard": Integer }},
    ...
  }}
}}

Do not return any explanation or markdown. Just JSON.

Question Paper:
{paper_text}
"""

    try:
        response = llm.invoke(prompt)
        output = response.strip()
        output = re.sub(r"^```json\s*|\s*```$", "", output)
        parsed = json.loads(output)

        end_time = time.time()
        print(f"✅ Done: {source} in {end_time - start_time:.2f} seconds")
        return parsed
    except Exception as e:
        print(f"❌ Error for {source}: {e}")
        print("🔎 Raw LLM output:", response)
        return None


def process_full_exam_with_metadata(grouped_pdfs):
    files_dict = {}
    for page_data in grouped_pdfs:
        source = page_data.get("source", "Unknown")
        if source not in files_dict:
            files_dict[source] = []
        files_dict[source].append(page_data["text"])
    
    # Combine pages per file
    file_papers = []
    for source, page_texts in files_dict.items():
        combined_text = "\n".join(page_texts)
        file_papers.append({
            "text": combined_text,
            "source": source
        })
    
    print(f"📊 Processing {len(file_papers)} files:")
    for i, paper in enumerate(file_papers):
        print(f"  {i+1}. {paper.get('source', 'Unknown')}")
    
    # Now process the combined files
    start_time = time.time()
    topics_counter = {}
    yearwise_difficulty = {}

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(analyze_paper, paper) for paper in file_papers]

        for future in as_completed(futures):
            parsed = future.result()
            if not parsed:
                continue

            # Merge topics
            for t in parsed.get("topics_frequency", []):
                key = (t["topic"], t["category"])
                topics_counter[key] = topics_counter.get(key, 0) + t["frequency"]

            # Merge difficulty
            for year, counts in parsed.get("yearwise_difficulty", {}).items():
                if year not in yearwise_difficulty:
                    yearwise_difficulty[year] = counts
                else:
                    for level in ["Easy", "Medium", "Hard"]:
                        yearwise_difficulty[year][level] += counts.get(level, 0)

    end_time = time.time()
    print(f"🕒 Total processing time: {end_time - start_time:.2f} seconds")

    final_topics = [
        {"topic": topic, "category": category, "frequency": freq}
        for (topic, category), freq in topics_counter.items()
    ]

    report = {
        "topics_frequency": final_topics,
        "yearwise_difficulty": yearwise_difficulty
    }

    return json.dumps(report, indent=2)
