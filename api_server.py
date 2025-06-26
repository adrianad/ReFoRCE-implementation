#!/usr/bin/env python3
"""
OpenAI-compatible API server for ReFoRCE Text-to-SQL system
"""
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from reforce.workflows.reforce_workflow import ReFoRCEWorkflow
from reforce.config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI-compatible API models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="reforce-text-to-sql", description="Model name")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=4096, gt=0)
    stream: Optional[bool] = Field(default=False)
    user: Optional[str] = Field(default=None)

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

class ErrorResponse(BaseModel):
    error: Dict[str, Any]

# FastAPI app
app = FastAPI(
    title="ReFoRCE Text-to-SQL API",
    description="OpenAI-compatible API for ReFoRCE Text-to-SQL system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global workflow instance
workflow: Optional[ReFoRCEWorkflow] = None

@app.on_event("startup")
async def startup_event():
    """Initialize ReFoRCE workflow on startup"""
    global workflow
    try:
        logger.info("Initializing ReFoRCE workflow...")
        workflow = ReFoRCEWorkflow()
        
        # Health check
        health_status = await workflow.health_check()
        if not all(health_status.values()):
            logger.warning("Some components are unhealthy")
            for component, status in health_status.items():
                logger.info(f"{component}: {'✓' if status else '✗'}")
        
        logger.info("ReFoRCE API server ready")
    except Exception as e:
        logger.error(f"Failed to initialize ReFoRCE: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ReFoRCE Text-to-SQL API Server",
        "version": "1.0.0",
        "endpoints": ["/v1/chat/completions", "/v1/models", "/health"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not workflow:
        raise HTTPException(status_code=503, detail="Workflow not initialized")
    
    try:
        health_status = await workflow.health_check()
        return {
            "status": "healthy" if all(health_status.values()) else "degraded",
            "components": health_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "reforce-text-to-sql",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "reforce",
                "permission": [],
                "root": "reforce-text-to-sql",
                "parent": None
            }
        ]
    }

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    if not workflow:
        raise HTTPException(status_code=503, detail="Workflow not initialized")
    
    try:
        # Extract user request from messages
        user_message = None
        system_message = None
        
        for message in request.messages:
            if message.role == "user":
                user_message = message.content
            elif message.role == "system":
                system_message = message.content
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Process through ReFoRCE pipeline
        start_time = time.time()
        result = await workflow.process_text_to_sql_request(user_message)
        processing_time = time.time() - start_time
        
        # Format response based on system message or default
        if system_message and "json" in system_message.lower():
            # Return structured JSON response
            response_content = json.dumps({
                "sql": result.final_sql,
                "confidence": result.confidence,
                "pipeline_stage": result.pipeline_stage,
                "execution_ready": result.execution_successful,
                "statistics": {
                    "compression_ratio": result.compression_ratio,
                    "candidates_generated": result.candidates_generated,
                    "exploration_performed": result.exploration_performed,
                    "processing_time": result.processing_time
                }
            }, indent=2)
        else:
            # Return formatted text response
            confidence_indicator = "🟢" if result.confidence >= 0.8 else "🟡" if result.confidence >= 0.6 else "🔴"
            
            response_content = f"""**Generated SQL Query:**

```sql
{result.final_sql}
```

**Analysis:**
- {confidence_indicator} Confidence: {result.confidence:.1%}
- Pipeline Stage: {result.pipeline_stage.upper()}
- Processing Time: {result.processing_time:.2f}s
- Schema Compression: {result.compression_ratio:.1%}
- Candidates Generated: {result.candidates_generated}
- Column Exploration: {"Yes" if result.exploration_performed else "No"}
- Execution Ready: {"✅" if result.execution_successful else "❌"}

*Generated by ReFoRCE Text-to-SQL system*"""
        
        # Calculate token usage (approximate)
        prompt_tokens = sum(len(msg.content.split()) for msg in request.messages) * 1.3
        completion_tokens = len(response_content.split()) * 1.3
        
        # Create response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_content),
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=int(prompt_tokens + completion_tokens)
            )
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/v1/completions")
async def completions(request: dict):
    """Legacy completions endpoint (redirect to chat completions)"""
    # Convert legacy format to chat format
    prompt = request.get("prompt", "")
    
    chat_request = ChatCompletionRequest(
        model=request.get("model", "reforce-text-to-sql"),
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=request.get("temperature", 0.1),
        max_tokens=request.get("max_tokens", 4096)
    )
    
    return await chat_completions(chat_request)

@app.get("/v1/sql/direct")
async def direct_sql_generation(query: str):
    """Direct SQL generation endpoint (non-OpenAI format)"""
    if not workflow:
        raise HTTPException(status_code=503, detail="Workflow not initialized")
    
    try:
        result = await workflow.process_text_to_sql_request(query)
        return {
            "query": query,
            "sql": result.final_sql,
            "confidence": result.confidence,
            "pipeline_stage": result.pipeline_stage,
            "execution_ready": result.execution_successful,
            "processing_time": result.processing_time,
            "metadata": {
                "compression_ratio": result.compression_ratio,
                "candidates_generated": result.candidates_generated,
                "exploration_performed": result.exploration_performed
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/schema/info")
async def schema_info():
    """Get database schema information"""
    if not workflow:
        raise HTTPException(status_code=503, detail="Workflow not initialized")
    
    try:
        db_manager = workflow.db_manager
        tables = db_manager.get_all_tables()
        table_sizes = db_manager.get_table_sizes()
        
        return {
            "total_tables": len(tables),
            "tables": tables[:20],  # First 20 tables
            "largest_tables": sorted(table_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def create_api_server(host: str = "0.0.0.0", port: int = 8080, workers: int = 1):
    """Create and configure the API server"""
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True
    )
    return uvicorn.Server(config)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ReFoRCE Text-to-SQL API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False
    )