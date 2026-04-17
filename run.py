"""
DraftClear Startup Script
Runs the FastAPI server with the frontend
"""
import sys
import logging
import os
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("DraftClear - Starting Application")
    logger.info("=" * 60)

    try:
        import uvicorn
        from api import app

        logger.info("Launching FastAPI server...")
        logger.info("Web Interface: http://localhost:8000")
        logger.info("API Documentation: http://localhost:8000/docs")
        logger.info("")
        logger.info("Press CTRL+C to stop the server")
        logger.info("=" * 60)

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )

    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
