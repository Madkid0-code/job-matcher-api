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
    if not job_text.strip():
        return 0.0
    documents = [resume_text.lower(), job_text.lower()]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    try:
        sparse_matrix = vectorizer.fit_transform(documents)
        score = cosine_similarity(sparse_matrix[0:1], sparse_matrix[1:2])[0][0]
        return round(float(score), 3)
    except:
        return 0.0

@app.get("/")
def home():
    return {"status": "Smart Job Brain v3 is Live"}

@app.post("/match")
async def match(data: MatchRequest):
    company = data.company
    resume_text = data.resume_text
    
    # More targeted query to hit actual job postings
    search_query = f'"{company}" ("data analyst" OR "data scientist" OR "analytics") (job OR opening OR position OR role) (python OR sql OR tableau)'
    
    best_link = f"https://www.google.com/search?q={company}+data+analyst+jobs"
    best_title = "No relevant job found"
    best_body = ""
    combined_job_text = ""
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=10))
            
            if results:
                # Filter for results that look like actual job postings
                job_results = []
                for r in results:
                    title_lower = r['title'].lower()
                    body_lower = r['body'].lower()
                    href_lower = r['href'].lower()
                    
                    # Score how "job-like" this result is
                    job_keywords = ['data analyst', 'job', 'opening', 'position', 'apply', 'requirements', 'qualifications']
                    relevance_score = sum(1 for kw in job_keywords if kw in title_lower or kw in body_lower)
                    
                    # Bonus for known job board or careers URLs
                    if any(site in href_lower for site in ['linkedin.com/jobs', 'indeed.com', 'greenhouse.io', 'lever.co', 'careers', '/jobs']):
                        relevance_score += 3
                    
                    if relevance_score > 0:  # Only keep promising results
                        job_results.append((relevance_score, r))
                
                if job_results:
                    # Pick the most job-like result
                    job_results.sort(reverse=True)  # Highest relevance first
                    best_result = job_results[0][1]
                    
                    best_link = best_result['href']
                    best_title = best_result['title']
                    best_body = best_result['body']
                    
                    # Combine text from top 3 job-like results for better scoring
                    top_3 = [r[1] for r in job_results[:3]]
                    combined_job_text = " ".join([r['title'] + " " + r['body'] for r in top_3])
                else:
                    # Fallback: use first result if nothing clearly job-like
                    best_link = results[0]['href']
                    best_title = results[0]['title']
                    combined_job_text = results[0]['body']
                    
    except Exception as e:
        print(f"Search error: {e}")
        combined_job_text = f"Generic data analyst role at {company} requiring Python and SQL."

    score = calculate_score(resume_text, combined_job_text or best_body)

    return {
        "company": company,
        "job_title": best_title,
        "match_score": score,
        "link": best_link,
        "verdict": "Strong Match" if score > 0.2 else "Possible Match" if score > 0.1 else "Low Match"
    }
