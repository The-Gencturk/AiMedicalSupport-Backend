from fastapi import FastAPI
from app.api.AnaylisyApi import router

app = FastAPI(
    title="Brain Radiology AI API",
    description="AI-powered brain radiology image analysis API",
    version="1.0.0"
)

# CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1", tags=["Analysis"])

# Root endpoint
@app.get("/", tags=["Health"])
async def root():
    return {"message": "Brain Radiology AI API is running", "status": "healthy"}

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "service": "Brain Radiology AI API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)