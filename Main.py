import subprocess
import socket
try:
    result = subprocess.run(
        [r"C:\Program Files\PostgreSQL\18\bin\pg_ctl.exe", "status", "-D", r"C:\Program Files\PostgreSQL\18\data"],
        capture_output=True, text=True
    )
    if "no server running" in result.stdout:
        subprocess.Popen([r"C:\Program Files\PostgreSQL\18\bin\pg_ctl.exe", "start", "-D", r"C:\Program Files\PostgreSQL\18\data"])
        import time
        time.sleep(3)
except:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.RoleController import router as role_router
from app.api.AnaylisyApi import router as analysis_router
from app.api.v1.AuthController import router as auth_router
from app.api.v1.PersonelController import router as personel_router
from app.models.User import User
from app.models.patient import Patient
from fastapi.staticfiles import StaticFiles
from app.models.AnalisyModel import Analysis, AnalysisReview
from app.api.v1.PatientController import router as patient_router
from app.db.DbContext import Base, engine
from pathlib import Path


Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="AiMedicalSupport API",
    description="AI-powered brain radiology image analysis API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
     allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
PROFILE_DIR = BASE_DIR / "wwwroot" / "UserProfile"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/wwwroot/UserProfile", StaticFiles(directory=str(PROFILE_DIR)), name="profile")
app.include_router(personel_router,prefix="/api/v1/personel", tags=["Personel"])
app.include_router(patient_router, prefix="/api/v1/Patient", tags=["Patient"])
app.include_router(analysis_router, prefix="/api/v1", tags=["Analysis"])
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(role_router, prefix="/api/v1/Role", tags=["Role"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
