"""
API Gateway Service
===================
Routes incoming requests to appropriate backend services.
Handles load balancing, rate limiting, and request validation.
"""
import time
import logging
import os
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ============================================================================
# CONFIGURATION
# ============================================================================
# Read configuration from environment variables
# This allows us to configure the service differently in each environment
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-inference:8000")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30.0"))
# ============================================================================
# METRICS
# ============================================================================
GATEWAY_REQUESTS = Counter(
    'gateway_requests_total',
    'Total gateway requests',
    ['method', 'endpoint', 'status', 'backend']
)
GATEWAY_LATENCY = Histogram(
    'gateway_request_latency_seconds',
    'Gateway request latency',
    ['method', 'endpoint', 'backend']
)
BACKEND_ERRORS = Counter(
    'gateway_backend_errors_total',
    'Backend service errors',
    ['backend', 'error_type']
)
# ============================================================================
# HTTP CLIENT
# ============================================================================
# We create a global HTTP client for connection pooling
http_client: Optional[httpx.AsyncClient] = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global http_client
    # Startup
    logger.info("API Gateway starting up...")
    http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    logger.info(f"Configured to forward to: {ML_SERVICE_URL}")
    yield
    # Shutdown
    logger.info("API Gateway shutting down...")
    if http_client:
        await http_client.aclose()
# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=1, max_length=1000)
    model_name: Optional[str] = "default"
class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_name: str
    latency_ms: float
    gateway_latency_ms: float
class ServiceStatus(BaseModel):
    service: str
    status: str
    url: str
# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="ML Platform API Gateway",
    description="Routes requests to ML inference services",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============================================================================
# ENDPOINTS
# ============================================================================
@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "ML Platform API Gateway",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/api/v1/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }
@app.get("/health")
async def health():
    """Gateway health check."""
    return {"status": "healthy", "service": "api-gateway"}
@app.get("/ready")
async def ready():
    """
    Readiness check - verifies backend is reachable.
    """
    try:
        response = await http_client.get(f"{ML_SERVICE_URL}/health")
        if response.status_code == 200:
            return {"status": "ready", "backend": "healthy"}
        else:
            raise HTTPException(503, "Backend not healthy")
    except httpx.RequestError as e:
        logger.error(f"Backend health check failed: {e}")
        raise HTTPException(503, f"Backend unreachable: {str(e)}")
@app.get("/api/v1/services", response_model=list[ServiceStatus])
async def list_services():
    """List available backend services and their status."""
    services = []
    # Check ML service
    try:
        response = await http_client.get(f"{ML_SERVICE_URL}/health")
        status = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception:
        status = "unreachable"
    services.append(ServiceStatus(
        service="ml-inference",
        status=status,
        url=ML_SERVICE_URL
    ))
    return services
@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Forward prediction request to ML service.
    This is the main endpoint that clients use. It:
    1. Validates the request
    2. Forwards to the ML service
    3. Adds gateway metrics
    4. Returns the response
    """
    start_time = time.time()
    try:
        # Forward request to ML service
        response = await http_client.post(
            f"{ML_SERVICE_URL}/predict",
            json=request.model_dump()
        )
        # Check response
        if response.status_code != 200:
            BACKEND_ERRORS.labels(
                backend="ml-inference",
                error_type="http_error"
            ).inc()
            raise HTTPException(
                status_code=response.status_code,
                detail=f"ML service error: {response.text}"
            )
        # Parse response
        result = response.json()
        gateway_latency = (time.time() - start_time) * 1000
        # Record metrics
        GATEWAY_REQUESTS.labels(
            method="POST",
            endpoint="/api/v1/predict",
            status=200,
            backend="ml-inference"
        ).inc()
        GATEWAY_LATENCY.labels(
            method="POST",
            endpoint="/api/v1/predict",
            backend="ml-inference"
        ).observe(gateway_latency / 1000)
        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            model_name=result["model_name"],
            latency_ms=result["latency_ms"],
            gateway_latency_ms=gateway_latency
        )
    except httpx.RequestError as e:
        BACKEND_ERRORS.labels(
            backend="ml-inference",
            error_type="connection_error"
        ).inc()
        logger.error(f"Failed to reach ML service: {e}")
        raise HTTPException(503, f"ML service unavailable: {str(e)}")
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
