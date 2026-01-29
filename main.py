from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# This defines EXACTLY what the Brain expects from n8n
class MatchRequest(BaseModel):
    company: str
    resume_text: str

def calculate_score(resume_text, job_description):
    documents = [resume_text, job_description]
    count_vectorizer = TfidfVectorizer(stop_words='english')
    sparse_matrix = count_vectorizer.fit_transform(documents)
    score = cosine_similarity(sparse_matrix[0:1], sparse_matrix[1:2])
    return round(float(score[0][0]), 3)

@app.get("/")
def home():
    return {"status": "Brain is fixed and online"}

@app.post("/match")
async def match(data: MatchRequest):
    # We pull the data out of the 'data' object n8n sends
    company = data.company
    resume_text = data.resume_text
    
    job_snippet = f"We are looking for a professional to work at {company}. Skills: Python, SQL, Data Analysis."
    score = calculate_score(resume_text, job_snippet)
    
    return {
        "company": company,
        "match_score": score,
        "link": f"https://www.google.com/search?q={company}+careers"
    }
