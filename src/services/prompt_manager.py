"""
Enhanced Prompt Management System.

Provides template management with versioning, caching, async operations,
and dynamic prompt composition for different agent types.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import jinja2
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PromptTemplate(BaseModel):
    """Individual prompt template with metadata"""

    name: str = Field(..., description="Template name")
    version: str = Field(..., description="Template version")
    content: str = Field(..., description="Template content with placeholders")
    description: Optional[str] = Field(None, description="Template description")
    required_variables: list[str] = Field(
        default_factory=list, description="Required template variables"
    )
    optional_variables: list[str] = Field(
        default_factory=list, description="Optional template variables"
    )
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    tags: list[str] = Field(default_factory=list, description="Template tags for categorization")


class PromptConfig(BaseModel):
    """Configuration for prompt formatting"""

    template_name: str = Field(..., description="Name of template to use")
    version: str = Field(default="latest", description="Template version")
    variables: dict[str, Any] = Field(default_factory=dict, description="Template variables")
    model_specific_adjustments: Optional[dict[str, Any]] = Field(
        None, description="Model-specific prompt adjustments"
    )


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
        auto_reload: bool = False,
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
            lstrip_blocks=True,
        )

        # Template cache
        self._template_cache: dict[str, PromptTemplate] = {}
        self._file_timestamps: dict[str, float] = {}

        # Built-in templates for core functionality
        self._builtin_templates = {
            "question_generation": self._get_question_generation_template(),
            "marking_scheme": self._get_marking_scheme_template(),
            "quality_review": self._get_quality_review_template(),
            "question_refinement": self._get_question_refinement_template(),
            "system_prompts": self._get_system_prompts_template(),
            # Document generation templates
            "document_generation": self._get_document_generation_template(),
            "worksheet_generation": self._get_worksheet_generation_template(),
            "notes_generation": self._get_notes_generation_template(),
            "textbook_generation": self._get_textbook_generation_template(),
            "slides_generation": self._get_slides_generation_template(),
            # V2 Document generation templates
            "block_selection": self._get_block_selection_template(),
            "document_content_generation": self._get_document_content_generation_template(),
        }

    async def render_prompt(self, config: PromptConfig, model_name: Optional[str] = None) -> str:
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
            variables.update(
                {"timestamp": datetime.now().isoformat(), "model_name": model_name or "unknown"}
            )

            # Add sensible defaults for common optional variables
            template_defaults = {
                "command_word": "Calculate",
                "subject_content_references": [],
                "grade_level": 7,
                "target_grade": variables.get("grade_level", 7),  # Alias for target_grade
                "calculator_policy": variables.get("calculator_policy", "not_allowed"),
                "tier": "Core",
                "topic": variables.get("topic", "mathematics"),
            }

            # Only add defaults for variables not already provided
            for key, default_value in template_defaults.items():
                if key not in variables:
                    variables[key] = default_value

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

    async def list_templates(self) -> list[PromptTemplate]:
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

        async with asyncio.to_thread(open, file_path, "w", encoding="utf-8") as f:
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
            async with asyncio.to_thread(open, file_path, "r", encoding="utf-8") as f:
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
                tags=metadata.get("tags", []),
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

    def _parse_metadata(self, metadata_section: str) -> dict[str, Any]:
        """Parse template metadata from header comments"""
        metadata = {}

        for line in metadata_section.split("\n"):
            line = line.strip()
            if line.startswith("# ") and ":" in line:
                key, value = line[2:].split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()

                # Parse lists
                if key in ["required_variables", "optional_variables", "tags"]:
                    metadata[key] = [v.strip() for v in value.split(",") if v.strip()]
                else:
                    metadata[key] = value

        return metadata

    async def _validate_variables(self, template: PromptTemplate, variables: dict[str, Any]):
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
{% if not include_diagrams -%}

**CRITICAL CONSTRAINT - NO DIAGRAMS:**
- Do NOT reference any diagrams, figures, charts, or visual elements
- Questions must be answerable using only the text provided
- Avoid phrases like "in the diagram," "from the figure," "the chart shows," etc.
- All necessary information must be stated in the question text
{% endif -%}
{% if cognitive_level -%}
- Cognitive Level: {{ cognitive_level }}
{% endif -%}
{% if tier -%}
- Tier: {{ tier }}
{% endif %}

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
            optional_variables=[
                "command_word",
                "subject_content_references",
                "include_diagrams",
                "cognitive_level",
                "tier",
            ],
            tags=["generation", "cambridge", "igcse", "mathematics"],
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
            tags=["marking", "assessment", "cambridge", "igcse"],
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
            tags=["review", "quality", "assessment", "cambridge"],
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
            tags=["refinement", "improvement", "quality", "cambridge"],
        )

    def _get_system_prompts_template(self) -> PromptTemplate:
        """Get built-in system prompts template"""
        content = """You are an expert {{ subject }} teacher specializing in {{ curriculum }}.

Your expertise includes:
- {{ grade }} level content
- {{ assessment_type }} design
- {{ pedagogical_approach }} methodology

Context: {{ context }}
Instructions: {{ instructions }}"""

        return PromptTemplate(
            name="system_prompts",
            version="latest",
            content=content,
            description="Generate system prompts for different educational contexts",
            required_variables=[
                "subject",
                "curriculum",
                "grade",
                "assessment_type",
                "pedagogical_approach",
            ],
            optional_variables=["context", "instructions"],
            tags=["system", "prompts", "educational", "context"],
        )

    def _get_document_generation_template(self) -> PromptTemplate:
        """Get built-in document generation template."""
        content = """You are an expert Cambridge IGCSE Mathematics educator. Generate a structured document following these specifications:

**Document Requirements:**
- Document Type: {{ document_type }}
- Detail Level: {{ detail_level }}
- Title: {{ title }}
- Topic: {{ topic }}
- Target Grade: {{ target_grade }}
{% if tier is defined and tier -%}
- Tier: {{ tier }}
{% endif -%}
{% if custom_instructions is defined and custom_instructions -%}
- Custom Instructions: {{ custom_instructions }}
{% endif -%}
{% if personalization_context is defined and personalization_context -%}
- Personalization: {{ personalization_context | tojson }}
{% endif %}

**Structure Guidelines:**
{% for section_type in structure_pattern -%}
- {{ section_type.replace('_', ' ').title() }}
{% endfor %}

**Cambridge Standards:**
- Follow Cambridge IGCSE curriculum requirements
- Use appropriate mathematical terminology and notation
- Ensure content matches the specified difficulty level
- Include clear learning objectives where appropriate

**Output Format:**
Return a JSON object with the following structure:
{
  "title": "{{ title }}",
  "document_type": "{{ document_type }}",
  "detail_level": "{{ detail_level }}",
  "sections": [
    {
      "title": "Section Title",
      "content_type": "section_type",
      "content_data": {
        "appropriate": "data for this section type"
      },
      "order_index": 0
    }
  ],
  "estimated_duration": "time in minutes",
  "total_questions": "number if applicable"
}

Generate educationally valuable content that matches the specified detail level and document type."""

        return PromptTemplate(
            name="document_generation",
            version="latest",
            content=content,
            description="Generate structured educational documents",
            required_variables=[
                "document_type",
                "detail_level",
                "title",
                "topic",
                "target_grade",
                "structure_pattern",
            ],
            optional_variables=["tier", "custom_instructions", "personalization_context"],
            tags=["document", "generation", "cambridge", "igcse"],
        )

    def _get_worksheet_generation_template(self) -> PromptTemplate:
        """Get built-in worksheet generation template."""
        # Note: Template content moved to prompts/worksheet_generation_latest.txt
        # This is a fallback for when file-based template is not available
        content = """You are an expert educational content creator specializing in Cambridge IGCSE Mathematics.

Generate a structured {{ document_type }} about {{ topic }} with the following requirements:
- Title: {{ title }}
- Detail Level: {{ detail_level }}/10
- Grade: {{ target_grade }}
- Structure: {{ structure_pattern | join(", ") }}

Return JSON with enhanced_title, introduction, and blocks array."""

        return PromptTemplate(
            name="worksheet_generation",
            version="latest",
            content=content,
            description="Generate practice worksheets for Cambridge IGCSE Mathematics (fallback template)",
            required_variables=[
                "title",
                "topic",
                "detail_level",
                "target_grade",
                "document_type",
                "structure_pattern",
            ],
            optional_variables=[
                "custom_instructions",
                "tier",
                "generate_versions",
                "export_formats",
                "personalization_context",
            ],
            tags=["worksheet", "practice", "cambridge", "igcse", "fallback"],
        )

    def _get_notes_generation_template(self) -> PromptTemplate:
        """Get built-in notes generation template."""
        content = """You are an expert Cambridge IGCSE Mathematics educator. Generate comprehensive study notes with the following specifications:

**Notes Requirements:**
- Title: {{ title }}
- Topic: {{ topic }}
- Detail Level: {{ detail_level }}
- Target Grade: {{ target_grade }}
{% if custom_instructions is defined and custom_instructions -%}
- Special Focus: {{ custom_instructions }}
{% endif -%}

**Content Structure:**
{% if detail_level == "minimal" -%}
- Key concepts summary
- Essential formulas
- Quick practice problems
{% elif detail_level == "medium" -%}
- Learning objectives
- Detailed explanations
- Key formulas with examples
- Practice problems
- Summary points
{% elif detail_level == "comprehensive" -%}
- Learning objectives
- Detailed theory with examples
- Key formulas and derivations
- Worked examples (multiple approaches)
- Practice problems with solutions
- Common misconceptions
- Exam tips and strategies
{% elif detail_level == "guided" -%}
- Step-by-step concept building
- Interactive examples with questions
- Progressive difficulty practice
- Self-check questions
- Study planning guidance
{% endif %}

**Educational Standards:**
- Follow Cambridge IGCSE curriculum sequence
- Use clear, age-appropriate explanations
- Include visual aids descriptions where helpful
- Provide multiple solution methods where applicable

**Output Format:**
Return a JSON object with structured sections containing explanatory content, examples, and practice materials.

Generate notes that effectively support student learning and exam preparation."""

        return PromptTemplate(
            name="notes_generation",
            version="latest",
            content=content,
            description="Generate study notes for Cambridge IGCSE Mathematics",
            required_variables=["title", "topic", "detail_level", "target_grade"],
            optional_variables=["custom_instructions"],
            tags=["notes", "study", "cambridge", "igcse"],
        )

    def _get_textbook_generation_template(self) -> PromptTemplate:
        """Get built-in textbook generation template."""
        content = """You are an expert Cambridge IGCSE Mathematics curriculum designer. Generate a comprehensive textbook section with the following specifications:

**Textbook Requirements:**
- Title: {{ title }}
- Topic: {{ topic }}
- Detail Level: {{ detail_level }}
- Target Grade: {{ target_grade }}
- Chapter/Section Focus: {{ custom_instructions | default("Complete topic coverage") }}

**Content Structure:**
{% if detail_level == "minimal" -%}
- Introduction to concepts
- Key definitions and formulas
- Basic examples
- Chapter exercises
{% elif detail_level == "medium" -%}
- Learning objectives
- Conceptual introduction
- Detailed explanations with examples
- Key formulas and theorems
- Worked examples
- Practice exercises
- Chapter summary
{% elif detail_level == "comprehensive" -%}
- Learning objectives and prerequisites
- Historical context and real-world applications
- Detailed theory with multiple representations
- Derivations and proofs (where appropriate)
- Extensive worked examples
- Progressive exercises (basic to advanced)
- Extension activities
- Chapter review and assessment
{% elif detail_level == "guided" -%}
- Clear learning pathway
- Step-by-step concept development
- Interactive elements and discovery activities
- Scaffolded examples
- Differentiated exercises
- Self-assessment opportunities
- Study skills integration
{% endif %}

**Academic Standards:**
- University-level pedagogical approach
- Multiple learning style accommodations
- Cross-curricular connections where relevant
- Preparation for advanced study
- Research-based teaching methods

**Output Format:**
Return a JSON object with comprehensive educational content suitable for textbook publication.

Generate content that provides thorough understanding and supports diverse learning needs."""

        return PromptTemplate(
            name="textbook_generation",
            version="latest",
            content=content,
            description="Generate comprehensive textbook content for Cambridge IGCSE Mathematics",
            required_variables=["title", "topic", "detail_level", "target_grade"],
            optional_variables=["custom_instructions"],
            tags=["textbook", "comprehensive", "cambridge", "igcse"],
        )

    def _get_slides_generation_template(self) -> PromptTemplate:
        """Get built-in slides generation template."""
        content = """You are an expert Cambridge IGCSE Mathematics educator. Generate presentation slides with the following specifications:

**Slides Requirements:**
- Title: {{ title }}
- Topic: {{ topic }}
- Detail Level: {{ detail_level }}
- Target Grade: {{ target_grade }}
- Presentation Duration: {{ estimated_duration | default(20) }} minutes
{% if custom_instructions is defined and custom_instructions -%}
- Presentation Focus: {{ custom_instructions }}
{% endif -%}

**Slide Structure:**
{% if detail_level == "minimal" -%}
- Title slide
- Key concepts (2-3 slides)
- Quick examples
- Summary slide
{% elif detail_level == "medium" -%}
- Title slide
- Learning objectives
- Concept introduction
- Key formulas
- Worked examples
- Practice opportunity
- Summary and next steps
{% elif detail_level == "comprehensive" -%}
- Title slide
- Learning objectives and context
- Detailed concept explanation
- Multiple examples and applications
- Interactive practice activities
- Real-world connections
- Assessment opportunities
- Extension activities
- Summary and reflection
{% elif detail_level == "guided" -%}
- Title slide
- Learning journey overview
- Step-by-step concept building
- Interactive discovery activities
- Guided practice with feedback
- Independent application
- Self-assessment
- Next steps planning
{% endif %}

**Presentation Best Practices:**
- Clear, concise bullet points
- Visual elements descriptions
- Interactive opportunities
- Appropriate pacing for audience
- Engagement strategies

**Output Format:**
Return a JSON object with slides structured for effective presentation delivery.

Generate slides that engage students and facilitate effective mathematics instruction."""

        return PromptTemplate(
            name="slides_generation",
            version="latest",
            content=content,
            description="Generate presentation slides for Cambridge IGCSE Mathematics",
            required_variables=["title", "topic", "detail_level", "target_grade"],
            optional_variables=["estimated_duration", "custom_instructions"],
            tags=["slides", "presentation", "cambridge", "igcse"],
        )

    def _get_block_selection_template(self) -> PromptTemplate:
        """Get built-in block selection template for V2 document generation."""
        content = """You are an expert Cambridge IGCSE Mathematics curriculum designer. Select and structure appropriate content blocks for document generation based on the following specifications:

**Document Requirements:**
- Document Type: {{ document_type }}
- Topic: {{ topic }}
- Detail Level: {{ detail_level }}
- Target Grade: {{ target_grade }}
- Time Constraint: {{ target_duration_minutes | default(30) }} minutes
{% if tier is defined and tier -%}
- Tier: {{ tier }}
{% endif -%}
{% if custom_instructions is defined and custom_instructions -%}
- Special Instructions: {{ custom_instructions }}
{% endif -%}
{% if syllabus_content is defined and syllabus_content -%}

**Syllabus Content References:**
{% for content_ref in syllabus_content -%}
- {{ content_ref.title }}: {{ content_ref.description }}
{% endfor -%}
{% endif %}

**Available Content Blocks:**
- IntroductionBlock: Learning objectives and topic overview
- ConceptExplanationBlock: Detailed theoretical explanations
- WorkedExampleBlock: Step-by-step solution demonstrations
- StepByStepGuideBlock: Practice questions with guided solutions
- PracticeQuestionsBlock: Independent practice exercises
- SummaryBlock: Key points and concept consolidation
- AssessmentBlock: Evaluation questions and rubrics
- ExtensionBlock: Advanced applications and enrichment

**Selection Guidelines:**
{% if document_type == "worksheet" -%}
- Focus on practice-oriented blocks (WorkedExample, StepByStepGuide, PracticeQuestions)
- Include minimal theory, maximum practice opportunity
- Ensure progressive difficulty in practice questions
{% elif document_type == "notes" -%}
- Emphasize explanation and understanding (Introduction, ConceptExplanation, WorkedExample)
- Include summary for consolidation
- Balance theory with practical examples
{% elif document_type == "textbook" -%}
- Comprehensive coverage (Introduction, ConceptExplanation, WorkedExample, Practice, Extension)
- Include assessment opportunities
- Provide both depth and breadth of coverage
{% elif document_type == "slides" -%}
- Presentation-friendly blocks (Introduction, WorkedExample, summary highlights)
- Keep content concise and visually structured
- Focus on key concepts and engagement
{% endif -%}

**Detail Level Adjustments:**
{% if detail_level <= 3 -%}
- Minimal blocks: Introduction + 1-2 core blocks + Summary
- Keep explanations concise and focused
{% elif detail_level <= 6 -%}
- Moderate blocks: Introduction + Explanation + Examples + Practice + Summary
- Balanced depth and breadth
{% elif detail_level <= 8 -%}
- Comprehensive blocks: Full range including Assessment and Extension
- Detailed explanations and multiple examples
{% else -%}
- Maximum coverage: All relevant blocks with advanced content
- Deep theoretical treatment and extensive practice
{% endif %}

**Time Constraint Considerations:**
- {{ target_duration_minutes }} minutes total time
- Select blocks that fit realistic completion time
- Balance content depth with time availability

**Output Format:**
Return a JSON object with this structure:
{
  "selected_blocks": [
    {
      "block_type": "IntroductionBlock",
      "title": "Learning Objectives",
      "priority": 1,
      "estimated_time_minutes": 5,
      "content_focus": "What students will learn in this session"
    },
    {
      "block_type": "ConceptExplanationBlock",
      "title": "Understanding [Topic]",
      "priority": 2,
      "estimated_time_minutes": 10,
      "content_focus": "Core theoretical concepts and definitions"
    }
  ],
  "total_estimated_time": {{ target_duration_minutes }},
  "educational_rationale": "Why these blocks were selected for this specific document type and requirements",
  "content_progression": "How the blocks build understanding from introduction to mastery"
}

Select blocks that create an effective learning progression and match the specified requirements."""

        return PromptTemplate(
            name="block_selection",
            version="latest",
            content=content,
            description="Select appropriate content blocks for V2 document generation",
            required_variables=["document_type", "topic", "detail_level", "target_grade"],
            optional_variables=[
                "target_duration_minutes",
                "tier",
                "custom_instructions",
                "syllabus_content",
            ],
            tags=["blocks", "selection", "v2", "cambridge", "igcse"],
        )

    def _get_document_content_generation_template(self) -> PromptTemplate:
        """Get built-in document content generation template for V2 document generation."""
        content = """You are an expert Cambridge IGCSE Mathematics educator. Generate comprehensive document content using the selected content blocks and following all specified requirements:

**Document Specifications:**
- Document Type: {{ document_type }}
- Title: {{ title }}
- Topic: {{ topic }}
- Grade Level: {{ grade_level }}
- Detail Level: {{ detail_level }}
- Duration: {{ target_duration_minutes }} minutes
- Tier: {{ tier }}
{% if custom_instructions -%}
- Special Instructions: {{ custom_instructions }}
{% endif -%}
{% if subtopics -%}
- Subtopics: {{ subtopics | join(', ') }}
{% endif -%}

**Syllabus Content References:**
{% for ref in syllabus_refs -%}
- {{ ref }}
{% endfor -%}

**Detailed Syllabus Content:**
{% for ref, content_detail in detailed_syllabus_content.items() -%}
**{{ ref }} - {{ content_detail.title }}:**
{{ content_detail.description }}

Key Learning Objectives:
{% for objective in content_detail.learning_objectives -%}
- {{ objective }}
{% endfor -%}

Key Concepts:
{% for concept in content_detail.key_concepts -%}
- {{ concept }}
{% endfor %}

{% endfor -%}

**Selected Content Blocks:**
{% for block in selected_blocks -%}
**{{ block.block_type }}:**
- Content Guidelines: {{ block.content_guidelines }}
- Estimated Volume: {{ block.estimated_content_volume }}
- Schema Requirements: {{ block.schema | tojson }}

{% endfor -%}

**Generation Requirements:**
1. Generate content for each selected block that matches the guidelines and estimated volume
2. Ensure mathematical accuracy and Cambridge IGCSE compliance
3. Use appropriate difficulty level for Grade {{ grade_level }}
4. Include clear explanations suitable for the {{ tier }} tier
5. Follow the exact JSON schema provided below

**CRITICAL: JSON Output Format**
You MUST return valid JSON matching this exact schema:
{{ output_schema | tojson }}

**Content Creation Guidelines:**
{% if document_type == "worksheet" -%}
- Focus on practice questions with clear instructions
- Include worked examples before independent practice
- Provide step-by-step solutions where appropriate
- Ensure progressive difficulty within each block
{% elif document_type == "notes" -%}
- Emphasize clear explanations and concept development
- Include multiple examples to illustrate concepts
- Use structured formatting with headings and bullet points
- Balance theory with practical applications
{% elif document_type == "textbook" -%}
- Provide comprehensive coverage with detailed explanations
- Include historical context where relevant
- Offer multiple solution methods for complex problems
- Add extension activities for advanced learners
{% elif document_type == "slides" -%}
- Use concise, presentation-friendly content
- Focus on key concepts and visual learning
- Include interactive elements and discussion points
- Keep text minimal but impactful
{% endif -%}

**Quality Standards:**
- All mathematical notation must be correct and consistent
- Content must align with Cambridge IGCSE curriculum requirements
- Explanations should be age-appropriate and clear
- Examples must be relevant and engaging
- Solutions must be mathematically sound and complete

Generate high-quality educational content that effectively supports student learning and meets all specified requirements."""

        return PromptTemplate(
            name="document_content_generation",
            version="latest",
            content=content,
            description="Generate structured document content using selected blocks for V2 generation",
            required_variables=[
                "document_type",
                "title",
                "topic",
                "grade_level",
                "detail_level",
                "target_duration_minutes",
                "tier",
                "selected_blocks",
                "output_schema",
            ],
            optional_variables=[
                "custom_instructions",
                "subtopics",
                "syllabus_refs",
                "detailed_syllabus_content",
                "personalization_context",
                "difficulty",
            ],
            tags=["content", "generation", "v2", "cambridge", "igcse"],
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
