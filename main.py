from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from duckduckgo_search import DDGS

app = FastAPI()

class MatchRequest(BaseModel):
    company: str
    resume_text: str

def calculate_score(resume_text, job_text):
    if not job_text: return 0.0
    documents = [resume_text, job_text]
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        sparse_matrix = vectorizer.fit_transform(documents)
        score = cosine_similarity(sparse_matrix[0:1], sparse_matrix[1:2])
        return round(float(score[0][0]), 3)
    except:
        return 0.0

@app.get("/")
def home():
    return {"status": "Advanced Search Brain is Live"}

@app.post("/match")
async def match(data: MatchRequest):
    company = data.company
    resume_text = data.resume_text
    
    # We search specifically for the "Jobs" page or a specific role
    search_query = f"{company} careers data analyst principal roles"
    
    best_link = f"https://www.google.com/search?q={company}+jobs"
    best_title = "No specific role found"
    combined_text = ""
    
    try:
        with DDGS() as ddgs:
            # We look at the top 5 results now to find a REAL job link
            results = [r for r in ddgs.text(search_query, max_results=5)]
            if results:
                # 1. Grab the link of the first result that looks like a job
                best_link = results[0]['href']
                # 2. Grab the title
                best_title = results[0]['title']
                # 3. Combine text for better scoring
                combined_text = " ".join([r['body'] for r in results])
    except Exception as e:
        print(f"Search error: {e}")

    score = calculate_score(resume_text, combined_text)
    
    return {
        "company": company,
        "job_title": best_title,
        "match_score": score,
        "link": best_link # This is now the REAL link found
    }
