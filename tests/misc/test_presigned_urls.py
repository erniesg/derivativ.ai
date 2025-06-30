#!/usr/bin/env python3
"""
Generate presigned URLs for existing R2 files.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import get_settings  # noqa: E402
from src.services.r2_storage_service import R2StorageService  # noqa: E402

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def generate_presigned_urls():
    """Generate presigned URLs for recently uploaded files."""
    print("🔗 Generating Presigned URLs for R2 Files")
    print("=" * 50)
    
    # Set demo mode
    os.environ["DEMO_MODE"] = "true"
    
    settings = get_settings()
    
    # Initialize R2 service
    r2_config = {
        "account_id": settings.cloudflare_account_id,
        "access_key_id": settings.cloudflare_r2_access_key_id,
        "secret_access_key": settings.cloudflare_r2_secret_access_key,
        "bucket_name": settings.cloudflare_r2_bucket_name,
        "region": settings.cloudflare_r2_region,
    }
    
    r2_service = R2StorageService(r2_config)
    
    # Recent file keys from our test
    test_files = [
        # Linear Equations Worksheet
        "documents/ae42c25f-6732-45f8-be4e-f501e793f14b/student.pdf",
        "documents/ae42c25f-6732-45f8-be4e-f501e793f14b/teacher.pdf",
        "documents/ae42c25f-6732-45f8-be4e-f501e793f14b/student.docx",
        "documents/ae42c25f-6732-45f8-be4e-f501e793f14b/teacher.docx",
        "documents/ae42c25f-6732-45f8-be4e-f501e793f14b/student.html",
        "documents/ae42c25f-6732-45f8-be4e-f501e793f14b/teacher.html",
        
        # Quadratic Functions Notes  
        "documents/ab3fdf43-22c4-4206-b42a-68ab38daeb21/student.pdf",
        "documents/ab3fdf43-22c4-4206-b42a-68ab38daeb21/teacher.pdf",
        "documents/ab3fdf43-22c4-4206-b42a-68ab38daeb21/student.docx", 
        "documents/ab3fdf43-22c4-4206-b42a-68ab38daeb21/teacher.docx",
    ]
    
    print("🌐 Generated Presigned URLs (valid for 24 hours):")
    print()
    
    for file_key in test_files:
        try:
            # Generate 24-hour presigned URL
            presigned_url = await r2_service.generate_presigned_url(
                file_key, expiration=86400  # 24 hours
            )
            
            # Extract document info from file key
            parts = file_key.split('/')
            doc_id = parts[1] if len(parts) > 1 else "unknown"
            version_format = parts[2] if len(parts) > 2 else "unknown"
            
            # Determine document type based on doc_id
            if doc_id == "ae42c25f-6732-45f8-be4e-f501e793f14b":
                doc_name = "Linear Equations Worksheet"
            elif doc_id == "ab3fdf43-22c4-4206-b42a-68ab38daeb21":
                doc_name = "Quadratic Functions Notes"
            else:
                doc_name = "Unknown Document"
            
            print(f"📄 **{doc_name}** ({version_format})")
            print(f"   {presigned_url}")
            print()
            
        except Exception as e:
            print(f"❌ Failed to generate URL for {file_key}: {e}")
    
    print("✅ All presigned URLs generated!")
    print("📝 These URLs are valid for 24 hours and provide direct access to download the files.")


if __name__ == "__main__":
    asyncio.run(generate_presigned_urls())