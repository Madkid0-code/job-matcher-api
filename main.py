from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from duckduckgo_search import DDGS  # Free search library

app = FastAPI()

class MatchRequest(BaseModel):
    company: str
    resume_text: str

def calculate_score(resume_text, job_description):
    documents = [resume_text, job_description]
    vectorizer = TfidfVectorizer(stop_words='english')
    sparse_matrix = vectorizer.fit_transform(documents)
    score = cosine_similarity(sparse_matrix[0:1], sparse_matrix[1:2])
    return round(float(score[0][0]), 3)

@app.get("/")
def home():
    return {"status": "Search Brain is Live"}

@app.post("/match")
async def match(data: MatchRequest):
    company = data.company
    resume_text = data.resume_text
    
    # NEW: Actually search the web for job openings
    search_query = f"{company} careers data analyst python jobs"
    job_description_snippet = ""
    
    try:
        with DDGS() as ddgs:
            # Get the first 3 search results
            results = [r for r in ddgs.text(search_query, max_results=3)]
            # Combine the snippets from the search results to "read" the JDs
            job_description_snippet = " ".join([r['body'] for r in results])
    except Exception as e:
        job_description_snippet = f"Generic hiring at {company}"

    score = calculate_score(resume_text, job_description_snippet)
    
    return {
        "company": company,
        "match_score": score,
        "verdict": "High Interest" if score > 0.15 else "Potential Match",
        "link": f"https://www.google.com/search?q={company}+careers+jobs"
    }
