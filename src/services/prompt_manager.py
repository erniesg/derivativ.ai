"""
Enhanced Prompt Management System.

Provides template management with versioning, caching, async operations,
and dynamic prompt composition for different agent types.
"""

import os
import asyncio
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from pydantic import BaseModel, Field
from datetime import datetime
import jinja2
import logging

logger = logging.getLogger(__name__)


class PromptTemplate(BaseModel):
    """Individual prompt template with metadata"""
    name: str = Field(..., description="Template name")
    version: str = Field(..., description="Template version")
    content: str = Field(..., description="Template content with placeholders")
    description: Optional[str] = Field(None, description="Template description")
    required_variables: List[str] = Field(default_factory=list, description="Required template variables")
    optional_variables: List[str] = Field(default_factory=list, description="Optional template variables")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    tags: List[str] = Field(default_factory=list, description="Template tags for categorization")


class PromptConfig(BaseModel):
    """Configuration for prompt formatting"""
    template_name: str = Field(..., description="Name of template to use")
    version: str = Field(default="latest", description="Template version")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Template variables")
    model_specific_adjustments: Optional[Dict[str, Any]] = Field(None, description="Model-specific prompt adjustments")


class PromptManager:
    """
    Enhanced prompt management system with async support.
    
    Features:
    - Template versioning and caching
    - Jinja2 template engine integration
    - Async template loading and rendering
    - Model-specific prompt adjustments
    - Template validation and variable checking
    - Hot reloading in development
    """
    
    def __init__(
        self,
        templates_dir: Union[str, Path] = "prompts",
        enable_cache: bool = True,
        auto_reload: bool = False
    ):
        """
        Initialize prompt manager.
        
        Args:
            templates_dir: Directory containing prompt templates
            enable_cache: Whether to cache loaded templates
            auto_reload: Whether to auto-reload templates in development
        """
        self.templates_dir = Path(templates_dir)
        self.enable_cache = enable_cache
        self.auto_reload = auto_reload
        
        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.templates_dir)),
            undefined=jinja2.StrictUndefined,  # Fail on undefined variables
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Template cache
        self._template_cache: Dict[str, PromptTemplate] = {}
        self._file_timestamps: Dict[str, float] = {}
        
        # Built-in templates for core functionality
        self._builtin_templates = {
            "question_generation": self._get_question_generation_template(),
            "marking_scheme": self._get_marking_scheme_template(),
            "quality_review": self._get_quality_review_template(),
            "question_refinement": self._get_question_refinement_template()
        }
    
    async def render_prompt(
        self,
        config: PromptConfig,
        model_name: Optional[str] = None
    ) -> str:
        """
        Render a prompt template with provided variables.
        
        Args:
            config: Prompt configuration with template and variables
            model_name: Target model name for model-specific adjustments
            
        Returns:
            Rendered prompt string
            
        Raises:
            TemplateNotFound: If template doesn't exist
            TemplateError: If template rendering fails
        """
        try:
            # Load template
            template = await self.get_template(config.template_name, config.version)
            
            # Prepare variables with model-specific adjustments
            variables = config.variables.copy()
            if config.model_specific_adjustments and model_name:
                adjustments = config.model_specific_adjustments.get(model_name, {})
                variables.update(adjustments)
            
            # Add default variables
            variables.update({
                "timestamp": datetime.now().isoformat(),
                "model_name": model_name or "unknown"
            })
            
            # Validate required variables
            await self._validate_variables(template, variables)
            
            # Render template
            jinja_template = self.jinja_env.from_string(template.content)
            rendered = jinja_template.render(**variables)
            
            return rendered.strip()
            
        except Exception as e:
            logger.error(f"Failed to render prompt {config.template_name}: {e}")
            raise PromptError(f"Prompt rendering failed: {e}")
    
    async def get_template(self, name: str, version: str = "latest") -> PromptTemplate:
        """
        Get a prompt template by name and version.
        
        Args:
            name: Template name
            version: Template version ("latest" for most recent)
            
        Returns:
            PromptTemplate object
        """
        cache_key = f"{name}_{version}"
        
        # Check cache first
        if self.enable_cache and cache_key in self._template_cache:
            template = self._template_cache[cache_key]
            
            # Check if we need to reload (auto_reload mode)
            if self.auto_reload:
                file_path = self._get_template_file_path(name, version)
                if file_path and await self._should_reload_template(file_path):
                    template = await self._load_template_from_file(name, version)
                    self._template_cache[cache_key] = template
            
            return template
        
        # Try to load from built-in templates first
        if name in self._builtin_templates and version == "latest":
            template = self._builtin_templates[name]
            if self.enable_cache:
                self._template_cache[cache_key] = template
            return template
        
        # Load from file
        template = await self._load_template_from_file(name, version)
        if self.enable_cache:
            self._template_cache[cache_key] = template
        
        return template
    
    async def list_templates(self) -> List[PromptTemplate]:
        """List all available templates"""
        templates = []
        
        # Add built-in templates
        for name, template in self._builtin_templates.items():
            templates.append(template)
        
        # Add file-based templates
        if self.templates_dir.exists():
            for file_path in self.templates_dir.glob("*.txt"):
                try:
                    name_version = file_path.stem
                    if "_v" in name_version:
                        name, version = name_version.rsplit("_v", 1)
                        version = f"v{version}"
                    else:
                        name = name_version
                        version = "v1.0"
                    
                    template = await self._load_template_from_file(name, version)
                    templates.append(template)
                except Exception as e:
                    logger.warning(f"Failed to load template from {file_path}: {e}")
        
        return templates
    
    async def save_template(self, template: PromptTemplate) -> Path:
        """Save a template to file"""
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{template.name}_{template.version}.txt"
        file_path = self.templates_dir / filename
        
        # Create template file with metadata header
        content = f"""# Template: {template.name}
# Version: {template.version}
# Description: {template.description or 'No description'}
# Required Variables: {', '.join(template.required_variables)}
# Optional Variables: {', '.join(template.optional_variables)}
# Tags: {', '.join(template.tags)}
# Created: {template.created_at.isoformat()}
# Updated: {template.updated_at.isoformat()}
---
{template.content}
"""
        
        async with asyncio.to_thread(open, file_path, 'w', encoding='utf-8') as f:
            await asyncio.to_thread(f.write, content)
        
        # Update cache
        cache_key = f"{template.name}_{template.version}"
        if self.enable_cache:
            self._template_cache[cache_key] = template
        
        return file_path
    
    def clear_cache(self):
        """Clear template cache"""
        self._template_cache.clear()
        self._file_timestamps.clear()
    
    async def _load_template_from_file(self, name: str, version: str) -> PromptTemplate:
        """Load template from file system"""
        file_path = self._get_template_file_path(name, version)
        
        if not file_path or not file_path.exists():
            raise TemplateNotFound(f"Template {name} version {version} not found")
        
        try:
            async with asyncio.to_thread(open, file_path, 'r', encoding='utf-8') as f:
                content = await asyncio.to_thread(f.read)
            
            # Parse metadata and content
            if content.startswith("#"):
                parts = content.split("---", 1)
                if len(parts) == 2:
                    metadata_section, template_content = parts
                    metadata = self._parse_metadata(metadata_section)
                else:
                    metadata = {}
                    template_content = content
            else:
                metadata = {}
                template_content = content
            
            # Update file timestamp for auto-reload
            if self.auto_reload:
                self._file_timestamps[str(file_path)] = file_path.stat().st_mtime
            
            return PromptTemplate(
                name=name,
                version=version,
                content=template_content.strip(),
                description=metadata.get("description"),
                required_variables=metadata.get("required_variables", []),
                optional_variables=metadata.get("optional_variables", []),
                tags=metadata.get("tags", [])
            )
            
        except Exception as e:
            raise TemplateError(f"Failed to load template {name}: {e}")
    
    def _get_template_file_path(self, name: str, version: str) -> Optional[Path]:
        """Get file path for template"""
        if not self.templates_dir.exists():
            return None
        
        # Try exact version match first
        filename = f"{name}_{version}.txt"
        file_path = self.templates_dir / filename
        if file_path.exists():
            return file_path
        
        # If looking for "latest", find highest version
        if version == "latest":
            pattern = f"{name}_v*.txt"
            matches = list(self.templates_dir.glob(pattern))
            if matches:
                # Sort by version number (simple string sort should work for v1.0, v1.1, etc.)
                matches.sort(key=lambda p: p.stem)
                return matches[-1]
        
        return None
    
    async def _should_reload_template(self, file_path: Path) -> bool:
        """Check if template file has been modified"""
        if not file_path.exists():
            return False
        
        current_mtime = file_path.stat().st_mtime
        cached_mtime = self._file_timestamps.get(str(file_path), 0)
        
        return current_mtime > cached_mtime
    
    def _parse_metadata(self, metadata_section: str) -> Dict[str, Any]:
        """Parse template metadata from header comments"""
        metadata = {}
        
        for line in metadata_section.split('\n'):
            line = line.strip()
            if line.startswith('# ') and ':' in line:
                key, value = line[2:].split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                # Parse lists
                if key in ['required_variables', 'optional_variables', 'tags']:
                    metadata[key] = [v.strip() for v in value.split(',') if v.strip()]
                else:
                    metadata[key] = value
        
        return metadata
    
    async def _validate_variables(self, template: PromptTemplate, variables: Dict[str, Any]):
        """Validate that all required variables are provided"""
        missing_variables = []
        for var in template.required_variables:
            if var not in variables:
                missing_variables.append(var)
        
        if missing_variables:
            raise TemplateError(f"Missing required variables: {', '.join(missing_variables)}")
    
    # Built-in template definitions
    def _get_question_generation_template(self) -> PromptTemplate:
        """Get built-in question generation template"""
        content = """You are an expert Cambridge IGCSE Mathematics question generator. Generate a high-quality mathematics question following these specifications:

**Question Requirements:**
- Topic: {{ topic }}
- Target Grade: {{ target_grade }}
- Marks: {{ marks }}
- Calculator Policy: {{ calculator_policy }}
{% if command_word -%}
- Command Word: {{ command_word }}
{% endif -%}
{% if subject_content_references -%}
- Syllabus References: {{ subject_content_references | join(', ') }}
{% endif %}

**Cambridge Standards:**
- Use official Cambridge IGCSE command words
- Ensure question complexity matches target grade level
- Follow Cambridge marking principles
- Include clear, unambiguous wording

**Output Format:**
Return a JSON object with the following structure:
{
  "question_text": "The complete question text",
  "marks": {{ marks }},
  "command_word": "{{ command_word or 'Calculate' }}",
  "subject_content_references": {{ subject_content_references | tojson if subject_content_references else '["C1.1"]' }},
  "solution_steps": [
    "Step 1: Description",
    "Step 2: Description"
  ],
  "final_answer": "The final answer",
  "marking_criteria": [
    {
      "criterion": "Description of what earns marks",
      "marks": 1,
      "mark_type": "M"
    }
  ]
}

Generate a mathematically accurate, age-appropriate question that tests the specified topic effectively."""
        
        return PromptTemplate(
            name="question_generation",
            version="latest",
            content=content,
            description="Generate Cambridge IGCSE Mathematics questions",
            required_variables=["topic", "target_grade", "marks", "calculator_policy"],
            optional_variables=["command_word", "subject_content_references"],
            tags=["generation", "cambridge", "igcse", "mathematics"]
        )
    
    def _get_marking_scheme_template(self) -> PromptTemplate:
        """Get built-in marking scheme template"""
        content = """You are an expert Cambridge IGCSE Mathematics examiner. Create a detailed marking scheme for the following question:

**Question:** {{ question_text }}
**Total Marks:** {{ total_marks }}
**Grade Level:** {{ target_grade }}

**Marking Principles:**
- Use Cambridge mark codes (M = Method, A = Accuracy, B = Independent mark, etc.)
- Ensure marks add up to the specified total
- Provide clear, unambiguous marking criteria
- Include alternative acceptable methods where applicable

**Output Format:**
Return a JSON object with this structure:
{
  "total_marks": {{ total_marks }},
  "marking_criteria": [
    {
      "criterion_text": "What the student must do/show to earn this mark",
      "marks_value": 1,
      "mark_type": "M",
      "notes": "Additional guidance for markers"
    }
  ],
  "final_answers": ["List of acceptable final answers"],
  "alternative_methods": ["Description of alternative solution approaches"],
  "common_errors": ["Common mistakes students might make"]
}

Create a comprehensive marking scheme that ensures consistent, fair assessment."""
        
        return PromptTemplate(
            name="marking_scheme",
            version="latest",
            content=content,
            description="Generate Cambridge IGCSE marking schemes",
            required_variables=["question_text", "total_marks", "target_grade"],
            optional_variables=[],
            tags=["marking", "assessment", "cambridge", "igcse"]
        )
    
    def _get_quality_review_template(self) -> PromptTemplate:
        """Get built-in quality review template"""
        content = """You are a Cambridge IGCSE Mathematics education expert. Review the following question and marking scheme for quality and compliance:

**Question to Review:**
{{ question_data | tojson }}

**Review Criteria:**
1. Mathematical accuracy (0-1 score)
2. Cambridge syllabus compliance (0-1 score)  
3. Grade-level appropriateness (0-1 score)
4. Question clarity and wording (0-1 score)
5. Marking scheme accuracy (0-1 score)

**Output Format:**
Return a JSON object with this structure:
{
  "overall_quality_score": 0.85,
  "mathematical_accuracy": 0.95,
  "cambridge_compliance": 0.80,
  "grade_appropriateness": 0.90,
  "question_clarity": 0.85,
  "marking_accuracy": 0.90,
  "feedback_summary": "Brief overall assessment",
  "specific_issues": [
    "List any specific problems found"
  ],
  "suggested_improvements": [
    "Specific recommendations for improvement"
  ],
  "decision": "approve|needs_revision|reject"
}

Provide thorough, constructive feedback focused on educational quality and Cambridge standards."""
        
        return PromptTemplate(
            name="quality_review",
            version="latest",
            content=content,
            description="Review question quality and compliance",
            required_variables=["question_data"],
            optional_variables=[],
            tags=["review", "quality", "assessment", "cambridge"]
        )
    
    def _get_question_refinement_template(self) -> PromptTemplate:
        """Get built-in question refinement template"""
        content = """You are an expert Cambridge IGCSE Mathematics educator. Improve the following question based on the provided feedback:

**Original Question:**
{{ original_question | tojson }}

**Review Feedback:**
{{ review_feedback | tojson }}

**Improvement Guidelines:**
- Address all specific issues mentioned in the feedback
- Maintain the original learning objectives and difficulty level
- Ensure the refined question meets Cambridge standards
- Preserve the mark allocation unless feedback suggests changes

**Output Format:**
Return a JSON object with this structure:
{
  "refined_question": {
    "question_text": "Improved question text",
    "marks": {{ original_question.marks }},
    "command_word": "{{ original_question.command_word }}",
    "subject_content_references": {{ original_question.subject_content_references | tojson }},
    "solution_steps": ["Refined solution steps"],
    "final_answer": "Refined final answer",
    "marking_criteria": ["Refined marking criteria"]
  },
  "improvements_made": [
    "List of specific improvements made"
  ],
  "justification": "Explanation of why these changes improve the question",
  "quality_impact": {
    "mathematical_accuracy": 0.95,
    "cambridge_compliance": 0.90,
    "grade_appropriateness": 0.85
  }
}

Focus on creating a better question that addresses the feedback while maintaining educational value."""
        
        return PromptTemplate(
            name="question_refinement",
            version="latest",
            content=content,
            description="Refine questions based on review feedback",
            required_variables=["original_question", "review_feedback"],
            optional_variables=[],
            tags=["refinement", "improvement", "quality", "cambridge"]
        )


# Custom exceptions
class PromptError(Exception):
    """Base exception for prompt-related errors"""
    pass


class TemplateNotFound(PromptError):
    """Raised when a template cannot be found"""
    pass


class TemplateError(PromptError):
    """Raised when template processing fails"""
    pass