"""
Modal deployment for Derivativ AI FastAPI backend.
Deploys the complete FastAPI application to Modal's serverless platform.
"""

import logging
import os

import modal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("derivativ-deploy")

# Create Modal App
app = modal.App("derivativ-ai")

# Get current directory for file paths
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load secrets from .env file if it exists
env_path = os.path.join(current_dir, ".env")
if os.path.exists(env_path):
    logger.info(f"Loading secrets from {env_path}")
    secrets = modal.Secret.from_dotenv(env_path)
else:
    logger.warning(f"Warning: .env file not found at {env_path}")
    # Create empty secret - you'll need to set up Modal secrets manually
    secrets = modal.Secret.from_dict({
        "DEMO_MODE": "true"  # Enable demo mode by default
    })

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install system dependencies
    .apt_install([
        "git",
        "curl",
        "build-essential",
        "pandoc",
        # PDF generation dependencies
        "texlive-xetex",
        "texlive-fonts-recommended",
        "texlive-plain-generic",
        "lmodern"
    ])
    # Install Python dependencies
    .pip_install_from_requirements("requirements.txt")
    # Set working directory and Python path first
    .workdir("/app")
    .env({"PYTHONPATH": "/app"})
    # Copy the entire source code (copy=True since we ran commands after)
    .add_local_dir("src", remote_path="/app/src", copy=True)
    .add_local_file("pyproject.toml", remote_path="/app/pyproject.toml", copy=True)
)

# Deploy FastAPI app as ASGI function
@app.function(
    image=image,
    secrets=[secrets],
    # Generous timeout for AI operations
    timeout=300,
    # Keep containers warm for better performance
    min_containers=1,
    # Allocate sufficient CPU and memory for AI workloads
    cpu=2,
    memory=2048,
)
@modal.asgi_app()
def fastapi_app():
    """Deploy FastAPI application to Modal."""
    import os
    import sys

    # Debug information
    logger.info("Starting Derivativ AI FastAPI application in Modal")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")

    # Verify environment setup
    required_env_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY"
    ]

    missing_vars = []
    for var in required_env_vars:
        if var not in os.environ:
            missing_vars.append(var)
            logger.warning(f"Environment variable {var} not set")
        else:
            logger.info(f"Found {var} in environment")

    # Enable demo mode if any required variables are missing
    if missing_vars:
        os.environ["DEMO_MODE"] = "true"
        logger.info(f"Demo mode enabled due to missing variables: {missing_vars}")

    # Import and return FastAPI app
    try:
        from src.api.main import app as derivativ_app
        logger.info("Successfully imported Derivativ FastAPI app")
        return derivativ_app
    except ImportError as e:
        logger.error(f"Failed to import FastAPI app: {e}")
        raise


# Standalone question generation function
@app.function(
    image=image,
    secrets=[secrets],
    timeout=120,
    cpu=2,
    memory=1024,
)
def generate_questions(
    topic: str,
    grade_level: int = 9,
    num_questions: int = 5,
    difficulty: str = "medium"
):
    """Generate questions using Modal compute - can be called independently."""
    import asyncio
    import os

    # Enable demo mode for standalone function
    os.environ["DEMO_MODE"] = "true"

    try:
        from src.core.config import get_settings
        from src.models.question_models import QuestionGenerationRequest
        from src.services.question_generation_service import QuestionGenerationService

        settings = get_settings()
        service = QuestionGenerationService(settings)

        request = QuestionGenerationRequest(
            topic=topic,
            grade_level=grade_level,
            num_questions=num_questions,
            difficulty=difficulty
        )

        async def _generate():
            result = await service.generate_questions_demo(request)
            return {
                "success": True,
                "questions": [q.dict() for q in result.questions],
                "metadata": {
                    "topic": topic,
                    "grade_level": grade_level,
                    "num_questions": num_questions,
                    "difficulty": difficulty
                }
            }

        return asyncio.run(_generate())

    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "metadata": {
                "topic": topic,
                "grade_level": grade_level,
            }
        }


# Document generation function
@app.function(
    image=image,
    secrets=[secrets],
    timeout=180,
    cpu=2,
    memory=1024,
)
def generate_document(
    material_type: str,
    topics: list,
    detail_level: int = 5,
    grade_level: int = 9
):
    """Generate educational documents using Modal compute."""
    import asyncio
    import os

    os.environ["DEMO_MODE"] = "true"

    try:
        from src.core.config import get_settings
        from src.models.document_models import DocumentGenerationRequest
        from src.services.document_generation_service import DocumentGenerationService

        settings = get_settings()
        service = DocumentGenerationService(settings)

        request = DocumentGenerationRequest(
            material_type=material_type,
            topics=topics,
            detail_level=detail_level,
            grade_level=grade_level
        )

        async def _generate():
            result = await service.generate_document(request)
            return {
                "success": True,
                "document": result.dict() if hasattr(result, 'dict') else result,
                "metadata": {
                    "material_type": material_type,
                    "topics": topics,
                    "detail_level": detail_level,
                    "grade_level": grade_level
                }
            }

        return asyncio.run(_generate())

    except Exception as e:
        logger.error(f"Document generation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "metadata": {
                "material_type": material_type,
                "topics": topics,
            }
        }


# Health check function
@app.function(image=image, timeout=30)
def health_check():
    """Health check function for monitoring."""
    import os
    return {
        "status": "healthy",
        "service": "derivativ-ai",
        "demo_mode": os.environ.get("DEMO_MODE", "false"),
        "timestamp": os.popen("date").read().strip()
    }


# Local entrypoint for testing
@app.local_entrypoint()
def main():
    """Local entrypoint for testing Modal functions."""
    print("🚀 Testing Derivativ AI Modal deployment...")

    # Test health check
    print("\n1. Testing health check...")
    health = health_check.remote()
    print(f"Health status: {health}")

    # Test question generation
    print("\n2. Testing question generation...")
    questions = generate_questions.remote(
        topic="Algebra",
        grade_level=9,
        num_questions=3,
        difficulty="medium"
    )
    print(f"Generated questions: {questions['success']}")
    if questions['success']:
        print(f"Number of questions: {len(questions['questions'])}")

    # Test document generation
    print("\n3. Testing document generation...")
    document = generate_document.remote(
        material_type="worksheet",
        topics=["Algebra", "Linear Equations"],
        detail_level=5,
        grade_level=9
    )
    print(f"Generated document: {document['success']}")

    print("\n✅ All tests completed!")


# Run the app when executed directly
if __name__ == "__main__":
    # This allows running `python deploy.py` to test locally
    main()
