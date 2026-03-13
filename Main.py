from fastapi import FastAPI
from app.api.AnaylisyApi import router
from pydantic_settings import BaseSettings

app = FastAPI(
    title="AiMedicalSupport API",
    description="AI-powered brain radiology image analysis API",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["Analysis"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
