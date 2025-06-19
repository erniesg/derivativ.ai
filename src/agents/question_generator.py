"""
Question Generator Agent for Cambridge IGCSE Mathematics.

Generates high-quality, curriculum-compliant mathematics questions using
modern async patterns with proper dependency injection and error handling.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio

from .base_agent import BaseAgent
from ..models.question_models import (
    Question, GenerationRequest, AgentResult, QuestionTaxonomy,
    SolutionAndMarkingScheme, SolverAlgorithm, FinalAnswer, 
    MarkingCriterion, SolverStep
)
from ..models.enums import CommandWord, SubjectContentReference, LLMModel
from ..services import LLMService, PromptManager, JSONParser
from ..services.prompt_manager import PromptConfig
from ..services.json_parser import JSONExtractionResult

logger = logging.getLogger(__name__)


class QuestionGeneratorError(Exception):
    """Raised when question generation fails"""
    pass


class QuestionGeneratorAgent(BaseAgent):
    """
    Agent responsible for generating Cambridge IGCSE Mathematics questions.
    
    Features:
    - Async generation with proper error handling
    - Cambridge syllabus compliance validation
    - Multiple LLM provider support
    - Automatic retry logic with fallbacks
    - Comprehensive logging and observability
    """
    
    def __init__(
        self,
        name: str = "QuestionGenerator",
        llm_service: Optional[LLMService] = None,
        prompt_manager: Optional[PromptManager] = None,
        json_parser: Optional[JSONParser] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Question Generator Agent.
        
        Args:
            name: Agent name for logging and identification
            llm_service: LLM service for content generation
            prompt_manager: Prompt management service
            json_parser: JSON extraction service
            config: Agent configuration
        """
        # Initialize services if not provided (dependency injection pattern)
        if llm_service is None:
            from ..services import MockLLMService
            llm_service = MockLLMService()
        
        if prompt_manager is None:
            prompt_manager = PromptManager()
        
        if json_parser is None:
            json_parser = JSONParser()
        
        super().__init__(name, llm_service, config)
        
        self.llm_service = llm_service
        self.prompt_manager = prompt_manager
        self.json_parser = json_parser
        
        # Agent configuration with defaults
        self.agent_config = {
            "max_retries": 3,
            "generation_timeout": 60,
            "quality_threshold": 0.7,
            "enable_fallback": True,
            **self.config
        }
        
        logger.info(f"Initialized {name} with LLM service: {type(llm_service).__name__}")
    
    async def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute question generation based on input request.
        
        Args:
            input_data: Dictionary containing generation parameters
            
        Returns:
            Dictionary with generated question data
            
        Raises:
            QuestionGeneratorError: If generation fails after all retries
        """
        try:
            # Parse and validate input
            request = self._parse_generation_request(input_data)
            self._observe(f"Parsed generation request for topic: {request.topic}")
            
            # Generate question with retries
            question = await self._generate_question_with_retries(request)
            
            if not question:
                raise QuestionGeneratorError("Failed to generate question after all retry attempts")
            
            # Validate generated question
            await self._validate_generated_question(question)
            self._act("Successfully generated and validated question")
            
            return {
                "question": question.model_dump(),
                "request": request.model_dump(),
                "generation_metadata": {
                    "agent_name": self.name,
                    "timestamp": datetime.now().isoformat(),
                    "model_used": request.llm_model.value
                }
            }
            
        except Exception as e:
            self._observe(f"Generation failed: {str(e)}")
            raise QuestionGeneratorError(f"Question generation failed: {e}")
    
    def _parse_generation_request(self, input_data: Dict[str, Any]) -> GenerationRequest:
        """Parse and validate generation request"""
        try:
            self._think("Parsing generation request parameters")
            
            # Pre-validate marks before Pydantic validation
            marks = input_data.get("marks")
            if marks is not None and (marks < 1 or marks > 20):
                raise ValueError("Marks must be between 1 and 20")
            
            # Convert input data to GenerationRequest
            request = GenerationRequest(**input_data)
            
            # Additional validation
            if not request.topic:
                raise ValueError("Topic is required for question generation")
            
            self._observe(f"Validated request: topic={request.topic}, marks={request.marks}")
            return request
            
        except ValueError as e:
            # Re-raise our custom validation errors
            raise QuestionGeneratorError(str(e))
        except Exception as e:
            raise QuestionGeneratorError(f"Invalid generation request: {e}")
    
    async def _generate_question_with_retries(self, request: GenerationRequest) -> Optional[Question]:
        """Generate question with retry logic and fallbacks"""
        max_retries = self.agent_config["max_retries"]
        
        for attempt in range(1, max_retries + 1):
            try:
                self._think(f"Generation attempt {attempt}/{max_retries}")
                
                # Generate using primary method
                question = await self._generate_question_primary(request, attempt)
                
                if question:
                    self._act(f"Successfully generated question on attempt {attempt}")
                    return question
                
            except Exception as e:
                self._observe(f"Attempt {attempt} failed: {str(e)}")
                
                # Try fallback on final attempt
                if attempt == max_retries and self.agent_config.get("enable_fallback"):
                    try:
                        self._think("Trying fallback generation method")
                        question = await self._generate_question_fallback(request)
                        if question:
                            self._act("Successfully generated question using fallback method")
                            return question
                    except Exception as fallback_error:
                        self._observe(f"Fallback method also failed: {str(fallback_error)}")
                
                # Wait before retry (exponential backoff)
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
        
        return None
    
    async def _generate_question_primary(self, request: GenerationRequest, attempt: int) -> Optional[Question]:
        """Primary question generation method"""
        try:
            # Prepare prompt
            prompt_config = PromptConfig(
                template_name="question_generation",
                version="latest",
                variables={
                    "topic": request.topic,
                    "target_grade": request.grade_level or 6,
                    "marks": request.marks,
                    "calculator_policy": request.calculator_policy.value,
                    "command_word": request.command_word.value if request.command_word else None,
                    "subject_content_references": [ref.value for ref in request.subject_content_refs] if request.subject_content_refs else None
                }
            )
            
            # Render prompt
            prompt = await self.prompt_manager.render_prompt(prompt_config, request.llm_model.value)
            self._observe(f"Generated prompt for {request.topic} (length: {len(prompt)})")
            
            # Call LLM service
            llm_response = await self.llm_service.generate(
                prompt=prompt,
                model=request.llm_model,
                temperature=request.temperature,
                timeout=self.agent_config["generation_timeout"]
            )
            
            self._observe(f"Received LLM response (tokens: {llm_response.tokens_used})")
            
            # Parse JSON response
            extraction_result = await self.json_parser.extract_json(
                llm_response.content,
                model_name=request.llm_model.value
            )
            
            if not extraction_result.success:
                raise QuestionGeneratorError(f"Failed to extract JSON: {extraction_result.error}")
            
            # Convert to Question object
            question = await self._convert_to_question_object(extraction_result.data, request)
            
            return question
            
        except Exception as e:
            logger.warning(f"Primary generation failed on attempt {attempt}: {e}")
            raise
    
    async def _generate_question_fallback(self, request: GenerationRequest) -> Optional[Question]:
        """Fallback generation method with simpler prompt"""
        try:
            self._think("Using simplified fallback generation")
            
            # Use simpler prompt for fallback
            simplified_prompt = f"""Generate a Cambridge IGCSE Mathematics question about {request.topic} 
for grade {request.grade_level or 6} worth {request.marks} marks.

Return JSON with: question_text, marks, command_word, solution_steps, final_answer"""
            
            llm_response = await self.llm_service.generate(
                prompt=simplified_prompt,
                model=LLMModel.GPT_4O_MINI,  # Use faster model for fallback
                temperature=0.5,
                timeout=30
            )
            
            extraction_result = await self.json_parser.extract_json(
                llm_response.content,
                model_name=LLMModel.GPT_4O_MINI.value
            )
            
            if extraction_result.success:
                return await self._convert_to_question_object(extraction_result.data, request, is_fallback=True)
            
            return None
            
        except Exception as e:
            logger.warning(f"Fallback generation failed: {e}")
            return None
    
    async def _convert_to_question_object(
        self,
        json_data: Dict[str, Any],
        request: GenerationRequest,
        is_fallback: bool = False
    ) -> Question:
        """Convert parsed JSON to Question object"""
        try:
            self._think("Converting JSON response to Question object")
            
            # Extract basic question data
            question_text = json_data.get("question_text", "")
            marks = json_data.get("marks", request.marks)
            command_word = json_data.get("command_word", "Calculate")
            
            # Generate unique IDs
            question_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Parse command word
            try:
                cmd_word = CommandWord(command_word)
            except ValueError:
                cmd_word = CommandWord.CALCULATE
            
            # Create taxonomy
            taxonomy = QuestionTaxonomy(
                topic_path=[request.topic],
                subject_content_references=request.subject_content_refs or [SubjectContentReference.C1_1],
                skill_tags=[request.topic.lower().replace(" ", "_")],
                cognitive_level=request.cognitive_level,
                difficulty_estimate=0.5  # Default difficulty
            )
            
            # Create solution and marking scheme
            solution_steps = json_data.get("solution_steps", ["Solution step 1"])
            final_answer = json_data.get("final_answer", "Answer")
            marking_criteria = json_data.get("marking_criteria", [])
            
            # Convert marking criteria
            mark_criteria = []
            for i, criterion in enumerate(marking_criteria):
                if isinstance(criterion, dict):
                    mark_criteria.append(MarkingCriterion(
                        criterion_id=f"crit_{i+1}",
                        criterion_text=criterion.get("criterion", f"Criterion {i+1}"),
                        mark_code_display=f"M{i+1}",
                        marks_value=criterion.get("marks", 1),
                        mark_type_primary=None
                    ))
            
            # If no criteria provided, create default
            if not mark_criteria:
                mark_criteria = [MarkingCriterion(
                    criterion_id="crit_1",
                    criterion_text="Correct method and answer",
                    mark_code_display="M1",
                    marks_value=marks,
                    mark_type_primary=None
                )]
            
            solution_and_marking = SolutionAndMarkingScheme(
                final_answers_summary=[FinalAnswer(answer_text=final_answer)],
                mark_allocation_criteria=mark_criteria,
                total_marks_for_part=marks
            )
            
            # Create solver algorithm
            solver_steps = [
                SolverStep(
                    step_number=i+1,
                    description_text=step,
                    skill_applied_tag=request.topic.lower()
                )
                for i, step in enumerate(solution_steps)
            ]
            
            solver_algorithm = SolverAlgorithm(steps=solver_steps)
            
            # Create Question object
            question = Question(
                question_id_local=question_id,
                question_id_global=f"derivativ_{question_id}",
                question_number_display="1",
                marks=marks,
                command_word=cmd_word,
                raw_text_content=question_text,
                taxonomy=taxonomy,
                solution_and_marking_scheme=solution_and_marking,
                solver_algorithm=solver_algorithm
            )
            
            self._act(f"Converted to Question object: {question.question_id_global}")
            return question
            
        except Exception as e:
            raise QuestionGeneratorError(f"Failed to convert JSON to Question: {e}")
    
    async def _validate_generated_question(self, question: Question):
        """Validate the generated question meets requirements"""
        try:
            self._think("Validating generated question")
            
            # Basic validation
            if not question.raw_text_content:
                raise ValueError("Question text cannot be empty")
            
            if len(question.raw_text_content) < 10:
                raise ValueError("Question text too short")
            
            if question.marks < 1 or question.marks > 20:
                raise ValueError("Invalid mark allocation")
            
            # Check marking scheme consistency
            scheme_total = question.solution_and_marking_scheme.total_marks_for_part
            if question.marks != scheme_total:
                logger.warning(f"Mark mismatch: question={question.marks}, scheme={scheme_total}")
            
            # Validate syllabus references
            if not question.taxonomy.subject_content_references:
                logger.warning("No subject content references provided")
            
            self._observe("Question validation completed successfully")
            
        except Exception as e:
            raise QuestionGeneratorError(f"Question validation failed: {e}")
    
    async def generate_multiple_questions(
        self,
        request: GenerationRequest
    ) -> List[Question]:
        """Generate multiple questions for a single request"""
        if request.count <= 1:
            result = await self._execute(request.model_dump())
            return [Question(**result["question"])]
        
        self._observe(f"Generating {request.count} questions")
        
        # Generate questions concurrently
        tasks = []
        for i in range(request.count):
            # Create individual request for each question
            individual_request = request.model_copy()
            individual_request.count = 1
            task = self._execute(individual_request.model_dump())
            tasks.append(task)
        
        # Wait for all generations to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        questions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Question {i+1} generation failed: {result}")
                continue
            
            questions.append(Question(**result["question"]))
        
        self._act(f"Generated {len(questions)}/{request.count} questions successfully")
        return questions
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about generation performance"""
        # This could be enhanced with actual metrics tracking
        return {
            "total_generations": getattr(self, '_generation_count', 0),
            "success_rate": getattr(self, '_success_rate', 0.0),
            "average_generation_time": getattr(self, '_avg_time', 0.0),
            "cache_stats": self.json_parser.get_extraction_stats() if hasattr(self.json_parser, 'get_extraction_stats') else {}
        }