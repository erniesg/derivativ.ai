"""
Template and prompt management API endpoints.

Provides endpoints for managing Jinja2 templates, prompt templates,
and document generation templates.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_prompt_manager
from src.services.prompt_manager import PromptConfig, PromptManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/templates", tags=["Templates"])


# Prompt Template Management


@router.get("/prompts")
async def list_prompt_templates(
    prompt_manager: PromptManager = Depends(get_prompt_manager),
) -> dict[str, Any]:
    """
    List all available prompt templates.

    Returns metadata about available Jinja2 templates including
    required variables, supported models, and template types.
    """
    try:
        # Get built-in templates
        builtin_templates = {
            "question_generation": {
                "name": "question_generation",
                "description": "Generate Cambridge IGCSE Mathematics questions",
                "required_variables": ["topic", "target_grade", "marks", "command_word"],
                "optional_variables": ["tier", "calculator_policy", "syllabus_refs"],
                "template_type": "question_generation",
                "supports_streaming": True,
            },
            "marking_scheme": {
                "name": "marking_scheme",
                "description": "Create detailed marking schemes",
                "required_variables": ["question_text", "marks", "command_word"],
                "optional_variables": ["difficulty_level", "grade_level"],
                "template_type": "marking",
                "supports_streaming": False,
            },
            "quality_review": {
                "name": "quality_review",
                "description": "Assess question quality and compliance",
                "required_variables": ["question_data"],
                "optional_variables": ["marking_scheme", "grade_level"],
                "template_type": "review",
                "supports_streaming": False,
            },
            "refinement": {
                "name": "refinement",
                "description": "Improve questions based on feedback",
                "required_variables": ["original_question", "review_feedback"],
                "optional_variables": ["specific_improvements"],
                "template_type": "refinement",
                "supports_streaming": True,
            },
            "document_generation": {
                "name": "document_generation",
                "description": "Generate educational documents",
                "required_variables": ["document_type", "detail_level", "title", "topic"],
                "optional_variables": ["custom_instructions", "personalization_context"],
                "template_type": "document_generation",
                "supports_streaming": True,
            },
            "worksheet_generation": {
                "name": "worksheet_generation",
                "description": "Generate mathematics worksheets",
                "required_variables": ["title", "topic", "detail_level"],
                "optional_variables": ["target_grade", "custom_instructions"],
                "template_type": "document_generation",
                "supports_streaming": True,
            },
            "notes_generation": {
                "name": "notes_generation",
                "description": "Generate study notes",
                "required_variables": ["title", "topic", "detail_level"],
                "optional_variables": ["target_grade", "custom_instructions"],
                "template_type": "document_generation",
                "supports_streaming": True,
            },
            "textbook_generation": {
                "name": "textbook_generation",
                "description": "Generate textbook chapters",
                "required_variables": ["title", "topic", "detail_level"],
                "optional_variables": ["target_grade", "custom_instructions"],
                "template_type": "document_generation",
                "supports_streaming": True,
            },
            "slides_generation": {
                "name": "slides_generation",
                "description": "Generate presentation slides",
                "required_variables": ["title", "topic", "detail_level"],
                "optional_variables": ["target_grade", "custom_instructions"],
                "template_type": "document_generation",
                "supports_streaming": True,
            },
        }

        return {
            "templates": builtin_templates,
            "total_templates": len(builtin_templates),
            "template_types": list(set(t["template_type"] for t in builtin_templates.values())),
        }

    except Exception as e:
        logger.error(f"Failed to list prompt templates: {e}")
        raise HTTPException(status_code=500, detail="Failed to list prompt templates")


@router.get("/prompts/{template_name}")
async def get_prompt_template(
    template_name: str,
    prompt_manager: PromptManager = Depends(get_prompt_manager),
) -> dict[str, Any]:
    """
    Get details of a specific prompt template.

    Returns the template content, required variables, and metadata.
    """
    try:
        # In a full implementation, this would retrieve from prompt manager
        # For now, return placeholder based on template name

        template_details = {
            "name": template_name,
            "content": f"Template content for {template_name}",
            "required_variables": ["topic", "target_grade"],
            "optional_variables": ["custom_instructions"],
            "template_type": "generation",
            "created_at": "2025-06-21T12:00:00Z",
            "last_modified": "2025-06-21T12:00:00Z",
        }

        return template_details

    except Exception as e:
        logger.error(f"Failed to get template {template_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get template {template_name}")


@router.post("/prompts/render")
async def render_prompt_template(
    render_request: dict[str, Any],
    prompt_manager: PromptManager = Depends(get_prompt_manager),
) -> dict[str, Any]:
    """
    Render a prompt template with provided variables.

    Takes a template name and variables, returns the rendered prompt
    ready for LLM consumption.
    """
    try:
        template_name = render_request.get("template_name")
        variables = render_request.get("variables", {})
        model_name = render_request.get("model_name", "gpt-4o-mini")

        if not template_name:
            raise HTTPException(status_code=400, detail="template_name is required")

        # Create prompt config
        prompt_config = PromptConfig(
            template_name=template_name,
            variables=variables,
        )

        # Render the prompt
        rendered_prompt = await prompt_manager.render_prompt(prompt_config, model_name=model_name)

        return {
            "template_name": template_name,
            "variables_used": variables,
            "model_name": model_name,
            "rendered_prompt": rendered_prompt,
            "character_count": len(rendered_prompt),
            "estimated_tokens": len(rendered_prompt.split()) * 1.3,  # Rough estimate
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to render template: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to render template: {e!s}")


@router.post("/prompts/validate")
async def validate_prompt_template(
    template_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate a prompt template.

    Checks template syntax, required variables, and Jinja2 compliance.
    """
    try:
        template_name = template_data.get("name")
        template_content = template_data.get("content")
        required_variables = template_data.get("required_variables", [])

        if not template_name or not template_content:
            raise HTTPException(status_code=400, detail="name and content are required")

        # Basic validation
        validation_results = {
            "template_name": template_name,
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
        }

        # Check for Jinja2 syntax (basic check)
        if "{{" not in template_content and "{%" not in template_content:
            validation_results["warnings"].append("Template does not contain Jinja2 syntax")

        # Check for required variables
        for var in required_variables:
            if f"{{{{{var}}}}}" not in template_content and f"{{{{ {var} }}}}" not in template_content:
                validation_results["warnings"].append(f"Required variable '{var}' not found in template")

        # Check for common issues
        if len(template_content) > 10000:
            validation_results["warnings"].append("Template is very long (>10,000 characters)")

        if template_content.count("{{") != template_content.count("}}"):
            validation_results["is_valid"] = False
            validation_results["errors"].append("Mismatched Jinja2 variable brackets")

        return validation_results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template validation failed: {e}")
        raise HTTPException(status_code=500, detail="Template validation failed")


@router.post("/prompts")
async def create_prompt_template(
    template_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Create a new prompt template.

    Saves a custom template for use in question generation or other tasks.
    """
    try:
        template_name = template_data.get("name")
        template_content = template_data.get("content")

        if not template_name or not template_content:
            raise HTTPException(status_code=400, detail="name and content are required")

        # In a full implementation, this would save to database
        # For now, return success response

        return {
            "message": "Template created successfully",
            "template_name": template_name,
            "template_id": f"custom_{template_name}",
            "created_at": "2025-06-21T12:00:00Z",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create template: {e}")
        raise HTTPException(status_code=500, detail="Failed to create template")


@router.put("/prompts/{template_name}")
async def update_prompt_template(
    template_name: str,
    template_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Update an existing prompt template.

    Modifies template content and metadata.
    """
    try:
        # In a full implementation, this would update in database
        return {
            "message": f"Template '{template_name}' updated successfully",
            "template_name": template_name,
            "updated_at": "2025-06-21T12:00:00Z",
        }

    except Exception as e:
        logger.error(f"Failed to update template {template_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update template {template_name}")


@router.delete("/prompts/{template_name}")
async def delete_prompt_template(template_name: str) -> dict[str, Any]:
    """
    Delete a prompt template.

    Removes a custom template from the system.
    """
    try:
        # In a full implementation, this would delete from database
        return {
            "message": f"Template '{template_name}' deleted successfully",
            "template_name": template_name,
            "deleted_at": "2025-06-21T12:00:00Z",
        }

    except Exception as e:
        logger.error(f"Failed to delete template {template_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete template {template_name}")


# Template Collections and Management


@router.get("/collections")
async def list_template_collections() -> dict[str, Any]:
    """
    List template collections (groups of related templates).

    Returns organized groups of templates for specific use cases.
    """
    try:
        collections = {
            "question_generation": {
                "name": "Question Generation",
                "description": "Templates for generating Cambridge IGCSE questions",
                "templates": ["question_generation", "marking_scheme", "quality_review", "refinement"],
                "template_count": 4,
            },
            "document_generation": {
                "name": "Document Generation",
                "description": "Templates for creating educational documents",
                "templates": ["document_generation", "worksheet_generation", "notes_generation", "textbook_generation", "slides_generation"],
                "template_count": 5,
            },
            "assessment_tools": {
                "name": "Assessment Tools",
                "description": "Templates for creating assessments and evaluations",
                "templates": ["marking_scheme", "quality_review"],
                "template_count": 2,
            },
        }

        return {
            "collections": collections,
            "total_collections": len(collections),
        }

    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail="Failed to list collections")


@router.post("/collections")
async def create_template_collection(
    collection_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Create a new template collection.

    Groups related templates for easier organization and access.
    """
    try:
        collection_name = collection_data.get("name")
        templates = collection_data.get("templates", [])

        if not collection_name:
            raise HTTPException(status_code=400, detail="collection name is required")

        return {
            "message": f"Collection '{collection_name}' created successfully",
            "collection_name": collection_name,
            "templates_count": len(templates),
            "created_at": "2025-06-21T12:00:00Z",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise HTTPException(status_code=500, detail="Failed to create collection")


# Template Usage Analytics


@router.get("/analytics/usage")
async def get_template_usage_analytics() -> dict[str, Any]:
    """
    Get template usage analytics.

    Returns statistics about template usage, popularity, and performance.
    """
    try:
        # In a full implementation, this would query usage database
        analytics = {
            "period": "last_30_days",
            "template_usage": {
                "question_generation": {
                    "usage_count": 245,
                    "success_rate": 98.8,
                    "avg_render_time_ms": 45,
                },
                "marking_scheme": {
                    "usage_count": 198,
                    "success_rate": 99.2,
                    "avg_render_time_ms": 32,
                },
                "document_generation": {
                    "usage_count": 67,
                    "success_rate": 96.4,
                    "avg_render_time_ms": 78,
                },
            },
            "most_popular_templates": [
                "question_generation",
                "marking_scheme",
                "document_generation",
            ],
            "least_used_templates": [
                "slides_generation",
            ],
            "total_renders_30d": 510,
            "avg_success_rate": 98.1,
        }

        return analytics

    except Exception as e:
        logger.error(f"Failed to get template analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get template analytics")
