"""
Prompt Loader Service - Handles loading and formatting prompt templates.

This service manages prompt templates for different purposes (generation, marking, review)
and provides template rendering with proper context substitution.
"""

import os
from typing import Dict, Any, Optional


class PromptLoader:
    """Service for loading and formatting prompt templates"""

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = prompts_dir
        self._template_cache = {}

    def load_template(self, template_name: str, version: str = "v1.0") -> str:
        """
        Load a prompt template from file.

        Args:
            template_name: Name of template (e.g., "question_generation", "marking_scheme")
            version: Template version (e.g., "v1.0", "v1.1")

        Returns:
            Raw template string with placeholders
        """
        template_key = f"{template_name}_{version}"

        # Check cache first
        if template_key in self._template_cache:
            return self._template_cache[template_key]

        # Construct filename
        filename = f"{template_name}_{version}.txt"
        filepath = os.path.join(self.prompts_dir, filename)

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                template_content = f.read()

            # Cache the template
            self._template_cache[template_key] = template_content
            return template_content

        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt template not found: {filepath}")

    def format_marking_scheme_prompt(
        self,
        template_version: str,
        question_text: str,
        target_grade: int,
        desired_marks: int,
        subject_content_references: list,
        calculator_policy: str,
        marking_principles: str,
        mark_types: str,
        expected_answer: Optional[str] = None,
        solution_steps: Optional[list] = None
    ) -> str:
        """
        Format marking scheme prompt template with provided context.

        Args:
            template_version: Version of marking template to use
            question_text: The question to create marking scheme for
            target_grade: Target grade level
            desired_marks: Number of marks for the question
            subject_content_references: List of syllabus references
            calculator_policy: Calculator policy string
            marking_principles: Cambridge marking principles text
            mark_types: Mark types reference text
            expected_answer: Expected answer (optional)
            solution_steps: Solution steps list (optional)

        Returns:
            Formatted prompt ready for LLM
        """
        template = self.load_template("marking_scheme", template_version)

        # Prepare context variables
        subject_refs_str = ', '.join(subject_content_references)

        # Handle optional context
        expected_answer_context = ""
        if expected_answer:
            expected_answer_context = f"**Expected Answer:** {expected_answer}"

        solution_steps_context = ""
        if solution_steps:
            steps_str = '\n'.join(f"{i+1}. {step}" for i, step in enumerate(solution_steps))
            solution_steps_context = f"**Solution Steps:**\n{steps_str}"

        # Format template with all variables
        formatted_prompt = template.format(
            question_text=question_text,
            target_grade=target_grade,
            desired_marks=desired_marks,
            subject_content_references=subject_refs_str,
            calculator_policy=calculator_policy,
            marking_principles=marking_principles,
            mark_types=mark_types,
            expected_answer_context=expected_answer_context,
            solution_steps_context=solution_steps_context
        )

        return formatted_prompt

    def format_question_generation_prompt(
        self,
        template_version: str,
        **context_vars
    ) -> str:
        """
        Format question generation prompt template.

        Args:
            template_version: Version of generation template to use
            **context_vars: All context variables for the template

        Returns:
            Formatted prompt ready for LLM
        """
        template = self.load_template("question_generation", template_version)

        # For v1.2 and higher, inject contextual skill tags based on subject content references
        if template_version >= "v1.2":
            from .skill_tag_mapper import SkillTagMapper

            skill_mapper = SkillTagMapper()

            # Extract subject content references from context
            subject_refs = context_vars.get('subject_content_references', [])
            if isinstance(subject_refs, str):
                # Parse if it's a string representation of a list
                import ast
                try:
                    subject_refs = ast.literal_eval(subject_refs)
                except (ValueError, SyntaxError):
                    # Fallback to splitting by comma
                    subject_refs = [ref.strip().strip('"\'') for ref in subject_refs.split(',')]

            # Get contextually relevant skill tags
            relevant_skill_tags = skill_mapper.get_relevant_skill_tags(subject_refs)
            formatted_skill_tags = self._format_skill_tags_for_prompt(relevant_skill_tags)
            context_vars['skill_tags'] = formatted_skill_tags

        return template.format(**context_vars)

    def _format_skill_tags_for_prompt(self, skill_tags: list) -> str:
        """
        Format skill tags list for display in prompt template.

        Args:
            skill_tags: List of skill tag strings

        Returns:
            Formatted string for prompt injection
        """
        # Group by topic/category for better readability
        number_tags = []
        algebra_tags = []
        geometry_tags = []
        stats_tags = []
        transform_tags = []
        general_tags = []

        for tag in sorted(skill_tags):
            tag_lower = tag.lower()
            if any(x in tag_lower for x in ['addition', 'subtraction', 'multiplication', 'division', 'number', 'fraction', 'percentage', 'place_value', 'rounding', 'time', 'prime', 'hcf', 'multiple']):
                number_tags.append(tag)
            elif any(x in tag_lower for x in ['algebra', 'equation', 'linear', 'simultaneous', 'substitution', 'factorisation', 'inequality', 'form_equation']):
                algebra_tags.append(tag)
            elif any(x in tag_lower for x in ['angle', 'triangle', 'geometry', 'area', 'bearing', 'construction', 'scale', 'parallel', 'alternate', 'symmetry']):
                geometry_tags.append(tag)
            elif any(x in tag_lower for x in ['probability', 'tree', 'scatter', 'correlation', 'mode', 'median', 'data', 'diagram']):
                stats_tags.append(tag)
            elif any(x in tag_lower for x in ['rotation', 'translation', 'enlargement', 'transformation', 'scale_factor', 'vector']):
                transform_tags.append(tag)
            else:
                general_tags.append(tag)

        formatted = ""

        if number_tags:
            formatted += "**Number & Arithmetic:**\n"
            formatted += ", ".join(number_tags) + "\n\n"

        if algebra_tags:
            formatted += "**Algebra:**\n"
            formatted += ", ".join(algebra_tags) + "\n\n"

        if geometry_tags:
            formatted += "**Geometry:**\n"
            formatted += ", ".join(geometry_tags) + "\n\n"

        if stats_tags:
            formatted += "**Statistics & Probability:**\n"
            formatted += ", ".join(stats_tags) + "\n\n"

        if transform_tags:
            formatted += "**Transformations:**\n"
            formatted += ", ".join(transform_tags) + "\n\n"

        if general_tags:
            formatted += "**General:**\n"
            formatted += ", ".join(general_tags) + "\n\n"

        return formatted.strip()

    def format_review_prompt(
        self,
        template_version: str,
        **context_vars
    ) -> str:
        """
        Format review prompt template.

        Args:
            template_version: Version of review template to use
            **context_vars: All context variables for the template

        Returns:
            Formatted prompt ready for LLM
        """
        template = self.load_template("review", template_version)
        return template.format(**context_vars)

    def format_refinement_prompt(
        self,
        original_question: Dict[str, Any],
        review_feedback: Dict[str, Any],
        template_version: str = "v1.0"
    ) -> str:
        """
        Format refinement prompt template with original question and review feedback.

        Args:
            original_question: Dictionary containing original question data
            review_feedback: Dictionary containing review feedback
            template_version: Version of refinement template to use

        Returns:
            Formatted prompt ready for LLM
        """
        template = self.load_template("refinement", template_version)

        # Extract feedback information with defaults
        overall_score = review_feedback.get('overall_score', 'Not available')
        feedback_issues = self._format_feedback_issues(review_feedback)

        # Extract individual scores with defaults
        clarity_score = review_feedback.get('clarity_score', 'Not available')
        difficulty_score = review_feedback.get('difficulty_score', 'Not available')
        curriculum_alignment_score = review_feedback.get('curriculum_alignment_score', 'Not available')
        mathematical_accuracy_score = review_feedback.get('mathematical_accuracy_score', 'Not available')

        # Format the original question
        original_question_str = self._format_original_question(original_question)

        # Format template with all variables
        formatted_prompt = template.format(
            original_question=original_question_str,
            overall_score=overall_score,
            feedback_issues=feedback_issues,
            clarity_score=clarity_score,
            difficulty_score=difficulty_score,
            curriculum_alignment_score=curriculum_alignment_score,
            mathematical_accuracy_score=mathematical_accuracy_score
        )

        return formatted_prompt

    def _format_feedback_issues(self, review_feedback: Dict[str, Any]) -> str:
        """Format feedback issues into a readable string."""
        issues = []

        # Check for specific issues based on scores and comments
        if review_feedback.get('clarity_score', 1.0) < 0.7:
            issues.append("- Question clarity needs improvement")

        if review_feedback.get('difficulty_score', 1.0) < 0.7:
            issues.append("- Difficulty level may not match intended grade")

        if review_feedback.get('curriculum_alignment_score', 1.0) < 0.7:
            issues.append("- Better alignment with curriculum objectives needed")

        if review_feedback.get('mathematical_accuracy_score', 1.0) < 0.7:
            issues.append("- Mathematical accuracy requires attention")

        # Add any specific feedback comments
        if 'feedback' in review_feedback and review_feedback['feedback']:
            issues.append(f"- Review comment: {review_feedback['feedback']}")

        return '\n'.join(issues) if issues else "No specific issues identified"

    def _format_original_question(self, original_question: Dict[str, Any]) -> str:
        """Format original question data into a readable string."""
        formatted_parts = []

        if 'question_text' in original_question:
            formatted_parts.append(f"Question: {original_question['question_text']}")

        if 'answer' in original_question:
            formatted_parts.append(f"Answer: {original_question['answer']}")

        if 'working' in original_question and original_question['working']:
            formatted_parts.append(f"Working: {original_question['working']}")

        if 'marks' in original_question:
            formatted_parts.append(f"Marks: {original_question['marks']}")

        if 'topic' in original_question:
            formatted_parts.append(f"Topic: {original_question['topic']}")

        if 'difficulty' in original_question:
            formatted_parts.append(f"Difficulty: {original_question['difficulty']}")

        return '\n'.join(formatted_parts)

    def list_available_templates(self) -> Dict[str, list]:
        """
        List all available prompt templates by type.

        Returns:
            Dictionary mapping template types to available versions
        """
        templates = {}

        if not os.path.exists(self.prompts_dir):
            return templates

        for filename in os.listdir(self.prompts_dir):
            if filename.endswith('.txt'):
                # Parse filename: template_name_version.txt
                name_parts = filename[:-4].split('_')  # Remove .txt
                if len(name_parts) >= 2:
                    version = name_parts[-1]  # Last part is version
                    template_name = '_'.join(name_parts[:-1])  # Everything else is name

                    if template_name not in templates:
                        templates[template_name] = []
                    templates[template_name].append(version)

        # Sort versions for each template
        for template_name in templates:
            templates[template_name].sort()

        return templates
