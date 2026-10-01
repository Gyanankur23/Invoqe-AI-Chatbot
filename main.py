"""
FastAPI Application for AI Chatbot
REST API for the chatbot system
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import os

from chatbot_engine import ChatbotEngine

# Initialize FastAPI app
app = FastAPI(
    title="AI Chatbot API",
    description="Customer Service Chatbot with NLP Intent Classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot engine
chatbot = None

@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup"""
    global chatbot
    try:
        chatbot = ChatbotEngine()
        print("Chatbot initialized successfully")
    except Exception as e:
        print(f"Error initializing chatbot: {e}")

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    intent: str
    confidence: float
    response: str
    user_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str

class IntentsResponse(BaseModel):
    intents: list
    count: int

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="healthy",
        message="AI Chatbot API is running"
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    return HealthResponse(
        status="healthy",
        message="Chatbot is ready"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint to process user messages"""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        result = chatbot.chat(request.message)
        
        return ChatResponse(
            intent=result['intent'],
            confidence=result['confidence'],
            response=result['response'],
            user_id=request.user_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/intents", response_model=IntentsResponse)
async def get_intents():
    """Get all available intents"""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    intents = chatbot.get_all_intents()
    
    return IntentsResponse(
        intents=intents,
        count=len(intents)
    )

if __name__ == "__main__":
    import uvicorn
    
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
