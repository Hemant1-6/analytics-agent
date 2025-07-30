from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api import analytics
import uvicorn
import os

app = FastAPI(
    title="Smart Analytics Agent (Unified)",
    description="An AI agent for data analysis and visualization, powered by LangChain and OpenAI.",
    version="3.1.0"
)

# API Router - IMPORTANT: a prefix is used to avoid conflicts with frontend routes
app.include_router(analytics.router, prefix="/api")

# Serve the React Frontend
STATIC_DIR = "frontend/build"

# Mount the static files directory for CSS, JS, etc.
app.mount("/static", StaticFiles(directory=os.path.join(STATIC_DIR, "static")), name="static")

@app.get("/{catchall:path}", response_class=FileResponse)
async def serve_react_app(request: Request, catchall: str):
    """Serves the main index.html file for any route not handled by the API."""
    filepath = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(filepath):
        # This is a fallback for development environments where the build folder might not exist yet
        return FileResponse("frontend/public/index.html")
    return FileResponse(filepath)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
