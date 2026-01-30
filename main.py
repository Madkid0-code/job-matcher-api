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
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )

    try:
        matrix = vectorizer.fit_transform(documents)
        score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return round(float(score), 3)
    except Exception:
        return 0.0


@app.get("/")
def home():
    return {"status": "Smart Job Brain v3.1 is Live"}


@app.post("/match")
async def match(data: MatchRequest):
    company = data.company.strip()
    resume_text = data.resume_text

    # Highly targeted search query
    search_query = (
        f'"{company}" ("data analyst" OR "data analytics" OR "analytics") '
        f'(job OR opening OR position OR role) '
        f'(sql OR python OR tableau OR "power bi")'
    )

    best_link = f"https://www.google.com/search?q={company}+data+analyst+jobs"
    best_title = "No relevant job found"
    combined_job_text = ""

    # Scoring keywords
    job_intent_keywords = [
        "data analyst", "analytics", "job", "opening", "position",
        "apply", "requirements", "qualifications"
    ]

    skill_keywords = [
        "sql", "python", "tableau", "power bi", "bi", "dashboard"
    ]

    priority_domains = [
        "careers", "jobs", "capitalonecareers.com",
        "linkedin.com/jobs", "indeed.com",
        "greenhouse.io", "lever.co"
    ]

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=12))

        scored_results = []

        for r in results:
            title = r.get("title", "").lower()
            body = r.get("body", "").lower()
            href = r.get("href", "").lower()

            relevance_score = 0

            # Job intent weight
            relevance_score += sum(2 for kw in job_intent_keywords if kw in title or kw in body)

            # Skill match weight
            relevance_score += sum(3 for kw in skill_keywords if kw in body)

            # Domain trust bonus
            if any(domain in href for domain in priority_domains):
                relevance_score += 5

            if relevance_score >= 5:
                scored_results.append((relevance_score, r))

        if scored_results:
            scored_results.sort(key=lambda x: x[0], reverse=True)

            top_results = [r[1] for r in scored_results[:3]]
            best_result = top_results[0]

            best_title = best_result["title"]
            best_link = best_result["href"]

            combined_job_text = " ".join(
                f"{r['title']} {r['body']}" for r in top_results
            )

        elif results:
            # Soft fallback
            best_title = results[0]["title"]
            best_link = results[0]["href"]
            combined_job_text = results[0]["body"]

    except Exception as e:
        print(f"Search error: {e}")
        combined_job_text = (
            f"Data analyst role at {company} requiring SQL, Python, "
            "Tableau, and Power BI."
        )

    score = calculate_score(resume_text, combined_job_text)

    verdict = (
        "Strong Match" if score >= 0.25
        else "Possible Match" if score >= 0.12
        else "Low Match"
    )

    return {
        "company": company,
        "job_title": best_title,
        "match_score": score,
        "link": best_link,
        "verdict": verdict
    }
