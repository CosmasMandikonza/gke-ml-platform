"""
ML Inference Service
====================
This service handles machine learning model predictions.
It exposes REST API endpoints for inference requests.
"""
# Standard library imports
import time
import logging
from typing import List, Optional
# Third-party imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
# Configure logging
# Logging helps us debug issues in production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# ============================================================================
# PROMETHEUS METRICS
# ============================================================================
# These metrics help us monitor our service's performance
# Counter: Tracks total number of requests
REQUEST_COUNT = Counter(
    'ml_inference_requests_total',
    'Total number of inference requests',
    ['method', 'endpoint', 'status']
)
# Histogram: Tracks request latency distribution
REQUEST_LATENCY = Histogram(
    'ml_inference_request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint']
)
# Counter: Tracks prediction counts
PREDICTION_COUNT = Counter(
    'ml_predictions_total',
    'Total number of predictions made',
    ['model_name']
)
# ============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================================
# Pydantic models validate incoming data and document our API
class PredictionRequest(BaseModel):
    """
    Schema for prediction requests.
    Attributes:
        features: List of numerical features for the model
        model_name: Which model to use for prediction (optional)
    """
    features: List[float] = Field(
        ...,  # ... means required
        description="List of input features for the model",
        min_length=1,
        max_length=1000
    )
    model_name: Optional[str] = Field(
        default="default",
        description="Name of the model to use"
    )
    # Example for API documentation
    class Config:
        json_schema_extra = {
            "example": {
                "features": [1.0, 2.0, 3.0, 4.0],
                "model_name": "default"
            }
        }
class PredictionResponse(BaseModel):
    """Schema for prediction responses."""
    prediction: float = Field(description="The model's prediction")
    confidence: float = Field(description="Confidence score (0-1)")
    model_name: str = Field(description="Model that made the prediction")
    latency_ms: float = Field(description="Inference latency in milliseconds")
class HealthResponse(BaseModel):
    """Schema for health check responses."""
    status: str
    version: str
    uptime_seconds: float
# ============================================================================
# SIMPLE ML MODEL (for demonstration)
# ============================================================================
class SimpleModel:
    """
    A simple model for demonstration purposes.
    In production, you'd load a trained model from disk or a model registry.
    """
    def __init__(self, name: str = "default"):
        self.name = name
        self.is_loaded = False
        logger.info(f"Initializing model: {name}")
    def load(self):
        """Simulate loading a model (would load from disk in production)"""
        # Simulate loading time
        time.sleep(0.1)
        self.is_loaded = True
        logger.info(f"Model {self.name} loaded successfully")
    def predict(self, features: List[float]) -> tuple:
        """
        Make a prediction.
        Args:
            features: Input features
        Returns:
            Tuple of (prediction, confidence)
        """
        if not self.is_loaded:
            self.load()
        # Simple prediction: weighted sum of features
        # In reality, this would be your trained model
        weights = np.random.RandomState(42).random(len(features))
        prediction = float(np.dot(features, weights))
        # Simulate confidence based on feature variance
        confidence = min(0.99, max(0.5, 1.0 / (1.0 + np.var(features))))
        return prediction, confidence
# ============================================================================
# FASTAPI APPLICATION
# ============================================================================
# Record start time for uptime calculation
START_TIME = time.time()
# Create FastAPI app with metadata
app = FastAPI(
    title="ML Inference Service",
    description="Microservice for machine learning model inference",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc documentation
)
# Add CORS middleware (allows requests from different domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize model
model = SimpleModel()
# ============================================================================
# MIDDLEWARE (runs on every request)
# ============================================================================
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """
    Middleware that tracks metrics for every request.
    Runs before and after each request handler.
    """
    start_time = time.time()
    # Process the request
    response = await call_next(request)
    # Calculate latency
    latency = time.time() - start_time
    # Record metrics (skip metrics endpoint to avoid recursion)
    if request.url.path != "/metrics":
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(latency)
    return response
# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/", response_model=dict)
async def root():
    """Root endpoint - returns service information."""
    return {
        "service": "ML Inference Service",
        "version": "1.0.0",
        "docs": "/docs"
    }
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Kubernetes uses this to determine if the pod is healthy.
    Returns 200 if healthy, which tells K8s the pod is working.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=time.time() - START_TIME
    )
@app.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint.
    Kubernetes uses this to determine if the pod is ready to receive traffic.
    We check if the model is loaded before accepting requests.
    """
    if not model.is_loaded:
        # Try to load the model
        try:
            model.load()
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready", "model_loaded": model.is_loaded}
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction using the ML model.
    This is the main endpoint that clients call to get predictions.
    """
    start_time = time.time()
    try:
        # Make prediction
        prediction, confidence = model.predict(request.features)
        # Record metric
        PREDICTION_COUNT.labels(model_name=request.model_name).inc()
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Prediction made: {prediction:.4f} (confidence: {confidence:.4f})")
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_name=request.model_name,
            latency_ms=latency_ms
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    Prometheus scrapes this endpoint to collect metrics.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """Runs when the service starts."""
    logger.info("ML Inference Service starting up...")
    # Pre-load model for faster first request
    model.load()
    logger.info("Service ready to accept requests")
@app.on_event("shutdown")
async def shutdown_event():
    """Runs when the service shuts down."""
    logger.info("ML Inference Service shutting down...")
# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    # Run with: python app/main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)