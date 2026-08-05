"""Entry point shim — `python server.py` runs the lily FastAPI app."""
from lily.app import app  # noqa: F401  (uvicorn string target below is canonical)

if __name__ == "__main__":
    import uvicorn

    from lily.config import settings

    uvicorn.run("lily.app:app", host="0.0.0.0", port=settings.port, reload=False)
