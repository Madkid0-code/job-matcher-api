from fastapi import FastAPI
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os

app = FastAPI()

# A very light "Brain" that compares words and meaning without using 500MB of RAM
def calculate_score(resume_text, job_description):
    documents = [resume_text, job_description]
    count_vectorizer = TfidfVectorizer(stop_words='english')
    sparse_matrix = count_vectorizer.fit_transform(documents)
    doc_term_matrix = sparse_matrix.todense()
    # This calculates how similar the two texts are (0.0 to 1.0)
    score = cosine_similarity(sparse_matrix[0:1], sparse_matrix[1:2])
    return round(float(score[0][0]), 3)

@app.get("/")
def home():
    return {"status": "Lightweight Brain is online"}

@app.post("/match")
async def match(company: str, resume_text: str):
    # For now, we simulate finding a job description. 
    # In a later step, we can add a real search API here.
    job_snippet = f"We are looking for a Data Analyst at {company} proficient in Python and SQL."
    
    score = calculate_score(resume_text, job_snippet)
    
    return {
        "company": company,
        "match_score": score,
        "verdict": "Strong Match" if score > 0.1 else "Weak Match",
        "link": f"https://www.google.com/search?q={company}+careers+jobs"
    }
