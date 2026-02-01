# serving/app/schemas.py

from pydantic import BaseModel
from typing import List, Optional

class RecommendationItem(BaseModel):
    product_id: int
    score: float

class RecommendResponse(BaseModel):
    user_id: int
    generated_at: str
    recommendations: List[RecommendationItem]
    mode: str
    message: Optional[str] = None
