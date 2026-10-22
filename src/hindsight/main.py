from fastapi import FastAPI

app = FastAPI(title="hindsight")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
