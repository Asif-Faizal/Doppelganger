#!/usr/bin/env python3
"""
Face Recognition API Server Runner
"""

import uvicorn
import logging
import sys
import os

# Add app directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the Face Recognition API server"""
    try:
        logger.info("Starting Face Recognition API server...")
        logger.info("Loading FaceNet model (this may take a moment on first run)...")
        
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 