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
from app.models.User import User
from app.models.rbac import Role, Permission, RolePermission, UserRole

app = FastAPI(
    title="AiMedicalSupport API",
    description="AI-powered brain radiology image analysis API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(analysis_router, prefix="/api/v1", tags=["Analysis"])
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(role_router, prefix="/api/v1/Role", tags=["Role"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)