from dataclasses import Field
from pydantic import BaseModel  


class ReplayPath(BaseModel):
    replay_path: str

class ReplayResponse(BaseModel):
    content: str

class ErrorResponse(BaseModel):
    detail: str