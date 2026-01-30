from fastapi import FastAPI
from pydantic import BaseModel
from duckduckgo_search import DDGS

app = FastAPI()

class CompanyRequest(BaseModel):
    company: str

ROLE_TERMS = [
    "data", "analytics", "analyst", "business intelligence", "bi"
]

SKILL_TERMS = [
    "sql", "python", "tableau", "power bi"
]

CAREER_TERMS = [
    "careers", "jobs", "join our team", "open positions"
]


def detect_signals(text: str):
    text = text.lower()

    career_signal = any(t in text for t in CAREER_TERMS)
    role_signal = any(t in text for t in ROLE_TERMS)
    skill_signal = any(t in text for t in SKILL_TERMS)

    score = (
        0.4 * career_signal +
        0.3 * role_signal +
        0.3 * skill_signal
    )

    return {
        "career_signal": career_signal,
        "role_signal": role_signal,
        "skill_signal": skill_signal,
        "confidence": round(score, 2)
    }


@app.get("/")
def home():
    return {"status": "Hiring Signal Brain is Live"}


@app.post("/detect")
async def detect(data: CompanyRequest):
    company = data.company.strip()

    search_query = f'"{company}" careers data analytics jobs'

    combined_text = ""
    sample_source = "unknown"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=5))

        if results:
            combined_text = " ".join(
                f"{r.get('title','')} {r.get('body','')}"
                for r in results
            )
            sample_source = results[0].get("href", "unknown")

    except Exception as e:
        print("Search failed:", e)

    signals = detect_signals(combined_text)

    is_hiring = signals["confidence"] >= 0.5

    return {
        "company": company,
        "is_hiring_analytics": is_hiring,
        **signals,
        "sample_source": sample_source
    }
