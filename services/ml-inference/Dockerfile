#===============================================================
#ML INFERENNCE SERVICE  DOCKERFILE
# ==============================================================
#This  Dockerfile creates a container image for our ML service.
#Think of it as a recipe :  start with a base , add ingredients , configure
# 
#------------------------------------------------------------
# STAGE 1 : Base Image 
# ------------------------------------------------------------
# We start from a slim Python image to keep our container small
# python:3.11-slim is ~45MB vs 3.11 which is ~350MB
FROM python:3.11-slim as base

#Set env variables
#PYTHONDONTWRITEBYTECODE: Dont create .pyc files (smaller image)
#PYTHONBUFFURED: Dont buffer output (see logs immediately)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # pip settings
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
#----------------------------------------------------------------
#STAGE 2: Dependencies
#----------------------------------------------------------------
FROM base as dependencies 
#install sys dependencies as needed by python packages
#gcc: Compiler for some python packages 
#well remove these after pip install to keep image small 
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*
#create and switch to app directory
WORKDIR /app
# Copy only requirements  first to leverage docker cache
#if requirements dont change, Docker wil reuse this layer
COPY requirements.txt  .
#
#install python dependencies
RUN pip install --no-cache-dir -r requirements.txt
#
#---------------------------------------------------------------
#STAGE 3: Production Image
#---------------------------------------------------------------
FROM base as production 

#Create non root user for better security
# running as root inside contianers is a securty risk
RUN groupadd -r appuser && useradd -r -g appuser appuser

#Set working directory 
WORKDIR /app

#Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=dependencies /usr/local/bin/ /usr/local/bin/

#Copy application code
COPY app/ ./app/

#Change ownership to non root user 
RUN chown -R appuser:appuser /app

# switch to non root user
USER appuser

#Expose port 

EXPOSE 8000

#Health check - Docker/k8s can use to check if container is healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
     CMD  python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

#Command to run the application
#using array form (exec form) is best practice
CMD ["uvicorn","app.main:app","--host", "0.0.0.0", "--port", "8000"]
