"""
FastAPI Application for AI Chatbot
REST API for the chatbot system
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import json
import os

# Initialize FastAPI app
app = FastAPI(
    title="AI Chatbot API",
    description="Customer Service Chatbot (simplified version for Vercel)",
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

# Simple rule-based responses for deployment
INTENTS = {
    "greeting": ["hello", "hi", "hey", "good morning", "good afternoon"],
    "goodbye": ["bye", "goodbye", "see you", "farewell"],
    "help": ["help", "assist", "support", "need help"],
    "thanks": ["thank", "thanks", "appreciate"],
    "pricing": ["price", "cost", "how much", "pricing"],
    "contact": ["contact", "email", "phone", "reach"],
}

RESPONSES = {
    "greeting": "Hello! How can I help you today?",
    "goodbye": "Goodbye! Have a great day!",
    "help": "I'm here to help. What do you need assistance with?",
    "thanks": "You're welcome! Is there anything else I can help with?",
    "pricing": "Our pricing plans start at $10/month. Would you like more details?",
    "contact": "You can reach us at support@example.com or call 1-800-123-4567.",
    "unknown": "I'm not sure I understand. Could you please rephrase that?"
}

def classify_intent(message):
    """Simple rule-based intent classification"""
    message_lower = message.lower()
    for intent, keywords in INTENTS.items():
        for keyword in keywords:
            if keyword in message_lower:
                return intent, 0.9
    return "unknown", 0.3

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
    """Root endpoint - return HTML for frontend"""
    return FileResponse('web_interface.html')

@app.get("/api", response_model=HealthResponse)
async def api_root():
    """API root endpoint"""
    return HealthResponse(
        status="healthy",
        message="AI Chatbot API is running (simplified version for Vercel)"
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Chatbot is ready"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint to process user messages"""
    try:
        intent, confidence = classify_intent(request.message)
        response = RESPONSES.get(intent, RESPONSES["unknown"])
        
        return ChatResponse(
            intent=intent,
            confidence=confidence,
            response=response,
            user_id=request.user_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/intents", response_model=IntentsResponse)
async def get_intents():
    """Get all available intents"""
    return IntentsResponse(
        intents=list(INTENTS.keys()),
        count=len(INTENTS)
    )

if __name__ == "__main__":
    import uvicorn
    
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
