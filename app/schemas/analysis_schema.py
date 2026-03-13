from pydantic import BaseModel
from typing import List, Optional

class AnalysisResult(BaseModel):
    """Analysis result schema"""
    status: str  # success, error
    details: str
    findings: Optional[List[str]] = None
    confidence: Optional[float] = None
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "details": "Image analyzed successfully",
                "findings": ["Finding 1", "Finding 2"],
                "confidence": 0.95
            }
        }
    }