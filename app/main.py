from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import auth, chat, data_management, documents, settings, sources

app = FastAPI(title="Document Processing & Chat API")

# Local dev only — the Angular dev server's origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(sources.router, prefix="/api/sources", tags=["sources"])
app.include_router(
    data_management.router, prefix="/api/data-management", tags=["data-management"]
)
app.include_router(settings.router, prefix="/api/settings", tags=["settings"])


@app.get("/api/health")
def health_check():
    return {"status": "ok"}
