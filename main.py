from fastapi import FastAPI, UploadFile, File
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/")
def home():
    return {"status": "Brain is online"}

@app.post("/match")
async def match(company: str, resume_text: str):
    # This is a simplified search using a free search engine (DuckDuckGo style)
    # To keep it free, we use a simple search query
    query = f"{company} careers data analyst jobs"
    # Note: In a real search, you'd use SerpAPI here. 
    # For now, let's assume we return a dummy 'found' status to test the flow
    return {
        "company": company,
        "match_score": 0.85,
        "link": f"https://www.google.com/search?q={company}+careers"
    }
