from fastapi import APIRouter, Header, HTTPException, Depends, Request
from app.core.security import verify_api_key


router = APIRouter(
    prefix="/reconfig_model",
    tags=["reconfig_model"],
)

@router.post("/{language}")
async def reconfig_model(
    language: str,
    x_api_key: str = Header(None)
):
    if not verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Access the transcription engine through the app state
    # transcription_engine = request.app.state.transcription_engine
    # transcription_engine.reconfig_model(language) 
    return {"message": "Model reconfigured"}