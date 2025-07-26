from fastapi import FastAPI
from app.api import analytics
import uvicorn

app = FastAPI(
    title="Smart Analytics Agent (LangChain + Gemini)",
    description="An AI agent for data analysis and visualization, powered by LangChain and Gemini.",
    version="1.6.0"
)

# Include the analytics router
app.include_router(analytics.router)

@app.get("/", tags=["Health Check"])
async def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Welcome to the Analytics Agent!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)