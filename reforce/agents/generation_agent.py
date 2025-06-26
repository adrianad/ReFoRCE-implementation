"""
Candidate Generation and Self-Refinement Agent for ReFoRCE
Stage 2: Generates SQL candidates and iteratively refines them
"""
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Sequence
from dataclasses import dataclass
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import ChatMessage, TextMessage
from autogen_agentchat.base import Response, TaskResult
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

from ..core.database_manager import DatabaseManager
from ..core.sql_executor import SQLExecutor, ExecutionResult
from ..models.llm_client import LLMClient
from ..models.prompt_templates import PromptTemplates
from ..config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class SQLCandidate:
    """Represents a SQL candidate query with metadata"""
    sql: str
    confidence: float
    iteration: int
    execution_result: Optional[ExecutionResult]
    refinement_history: List[str]
    validation_score: float = 0.0

class GenerationAgent(AssistantAgent):
    """
    AutoGen agent for SQL candidate generation and self-refinement
    Implements ReFoRCE Stage 2: Candidate Generation with Self-Refinement
    """
    
    def __init__(
        self,
        name: str = "GenerationAgent",
        description: str = "SQL generation and refinement specialist",
        db_manager: Optional[DatabaseManager] = None,
        llm_client: Optional[LLMClient] = None,
        model_client: Optional[OpenAIChatCompletionClient] = None
    ):
        # Initialize components
        self.db_manager = db_manager or DatabaseManager()
        self.llm_client = llm_client or LLMClient()
        
        # Create AutoGen model client if not provided
        if model_client is None:
            model_info = ModelInfo(
                family="qwen",
                vision=False,
                function_calling=False,
                json_output=False
            )
            model_client = OpenAIChatCompletionClient(
                model=settings.llm.model_name,
                api_key="dummy",  # vLLM doesn't require real API key
                base_url=settings.llm.base_url,
                model_info=model_info
            )
        
        # Initialize base AssistantAgent with model client
        super().__init__(name=name, description=description, model_client=model_client)
        self.sql_executor = SQLExecutor(self.db_manager)
        
        # Configuration
        self.config = settings.reforce
        self.max_refinement_iterations = self.config.max_refinement_iterations
        self.num_candidates = self.config.num_candidates
        
        # State tracking
        self.current_request = None
        self.compressed_schema = None
        self.candidates = []
        self.refinement_results = {}
    
    async def on_messages(
        self, 
        messages: Sequence[ChatMessage], 
        cancellation_token: CancellationToken
    ) -> Response:
        """Handle incoming messages for SQL generation and refinement"""
        try:
            latest_message = messages[-1]
            request_content = latest_message.content
            
            logger.info(f"GenerationAgent received request: {request_content[:100]}...")
            
            # Parse request type
            if "generate_candidates" in request_content.lower():
                result = await self._generate_sql_candidates(request_content)
            elif "refine_sql" in request_content.lower():
                result = await self._refine_sql_candidate(request_content)
            elif "validate_candidates" in request_content.lower():
                result = await self._validate_all_candidates()
            elif "get_best_candidate" in request_content.lower():
                result = await self._get_best_candidate()
            else:
                result = await self._handle_general_generation_request(request_content)
            
            return Response(
                chat_message=TextMessage(content=result, source=self.name)
            )
            
        except Exception as e:
            error_msg = f"GenerationAgent error: {str(e)}"
            logger.error(error_msg)
            return Response(
                chat_message=TextMessage(content=error_msg, source=self.name)
            )
    
    def set_compressed_schema(self, schema_text: str):
        """Set compressed schema from CompressionAgent"""
        self.compressed_schema = schema_text
        logger.info("Compressed schema received from CompressionAgent")
    
    async def _generate_sql_candidates(self, request: str) -> str:
        """Generate multiple SQL candidates for the given request"""
        try:
            # Extract user request from the message
            user_request = self._extract_user_request(request)
            self.current_request = user_request
            
            if not self.compressed_schema:
                return "Error: No compressed schema available. Please run CompressionAgent first."
            
            logger.info(f"Generating {self.num_candidates} SQL candidates for: {user_request[:100]}...")
            
            # Generate candidates concurrently
            candidate_tasks = []
            for i in range(self.num_candidates):
                task = self._generate_single_candidate(user_request, i)
                candidate_tasks.append(task)
            
            candidates = await asyncio.gather(*candidate_tasks, return_exceptions=True)
            
            # Process results
            valid_candidates = []
            for i, candidate in enumerate(candidates):
                if isinstance(candidate, Exception):
                    logger.error(f"Candidate {i} generation failed: {candidate}")
                else:
                    valid_candidates.append(candidate)
            
            self.candidates = valid_candidates
            
            # Start self-refinement process
            await self._self_refinement_process()
            
            result = f"""Generated {len(valid_candidates)} SQL candidates successfully!
            
Request: {user_request}

Candidates generated:
"""
            
            for i, candidate in enumerate(valid_candidates):
                result += f"""
Candidate {i+1} (Confidence: {candidate.confidence:.2f}):
```sql
{candidate.sql}
```
Refinement iterations: {candidate.iteration}
"""
            
            return result
            
        except Exception as e:
            logger.error(f"Candidate generation failed: {e}")
            raise
    
    async def _generate_single_candidate(self, user_request: str, candidate_id: int) -> SQLCandidate:
        """Generate a single SQL candidate"""
        try:
            # Use different temperature for diversity
            temperature = 0.1 + (candidate_id * 0.1)  # 0.1 to 0.8
            
            # Generate SQL
            response = await self.llm_client.generate_completion(
                prompt=PromptTemplates.get_generation_prompt(
                    schema_text=self.compressed_schema,
                    user_request=user_request,
                    context=f"Generate SQL candidate #{candidate_id + 1}"
                ),
                system_prompt=PromptTemplates.GENERATION_SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=4000  # Generous limit for Qwen thinking + SQL generation
            )
            
            # Extract SQL from response
            sql = self._extract_sql_from_response(response.content)
            
            # Initial validation
            validation_result = self.sql_executor.validate_sql_syntax(sql)
            validation_score = 1.0 if validation_result.is_valid else 0.5
            
            candidate = SQLCandidate(
                sql=sql,
                confidence=0.7,  # Initial confidence
                iteration=0,
                execution_result=None,
                refinement_history=[f"Initial generation with temperature {temperature}"],
                validation_score=validation_score
            )
            
            logger.info(f"Generated candidate {candidate_id + 1}: {len(sql)} characters")
            return candidate
            
        except Exception as e:
            logger.error(f"Single candidate generation failed: {e}")
            raise
    
    async def _self_refinement_process(self):
        """Apply self-refinement to all candidates"""
        logger.info("Starting self-refinement process...")
        
        for i, candidate in enumerate(self.candidates):
            try:
                refined_candidate = await self._refine_candidate(candidate, i)
                self.candidates[i] = refined_candidate
            except Exception as e:
                logger.error(f"Refinement failed for candidate {i}: {e}")
        
        logger.info("Self-refinement process completed")
    
    async def _refine_candidate(self, candidate: SQLCandidate, candidate_id: int) -> SQLCandidate:
        """Apply iterative refinement to a single candidate"""
        current_candidate = candidate
        consecutive_failures = 0
        
        for iteration in range(self.max_refinement_iterations):
            try:
                # Execute current SQL
                execution_result = await self.sql_executor.execute_sql_safe(current_candidate.sql)
                current_candidate.execution_result = execution_result
                
                if execution_result.success:
                    # Success - check for self-consistency
                    if iteration > 0:
                        # Compare with previous result for consistency
                        consistency_check = await self._check_self_consistency(current_candidate)
                        if consistency_check:
                            logger.info(f"Candidate {candidate_id} achieved self-consistency at iteration {iteration}")
                            break
                    
                    # Update confidence based on success
                    current_candidate.confidence = min(0.95, current_candidate.confidence + 0.1)
                    consecutive_failures = 0
                
                else:
                    # Failure - attempt refinement
                    consecutive_failures += 1
                    if consecutive_failures >= 2:
                        logger.warning(f"Candidate {candidate_id} failed {consecutive_failures} times, stopping refinement")
                        break
                    
                    refined_sql = await self._refine_sql_with_feedback(
                        current_candidate.sql,
                        execution_result.error,
                        self.current_request
                    )
                    
                    if refined_sql != current_candidate.sql:
                        current_candidate.sql = refined_sql
                        current_candidate.iteration = iteration + 1
                        current_candidate.refinement_history.append(
                            f"Iteration {iteration + 1}: Fixed error - {execution_result.error[:100]}"
                        )
                    else:
                        logger.warning(f"No refinement suggested for candidate {candidate_id}")
                        break
                
            except Exception as e:
                logger.error(f"Refinement iteration {iteration} failed for candidate {candidate_id}: {e}")
                break
        
        return current_candidate
    
    async def _refine_sql_with_feedback(self, sql: str, error: str, original_request: str) -> str:
        """Refine SQL based on execution feedback"""
        try:
            refinement_response = await self.llm_client.generate_completion(
                prompt=PromptTemplates.get_refinement_prompt(
                    original_request=original_request,
                    sql_query=sql,
                    feedback="Execution failed",
                    error_details=error,
                    schema_text=self.compressed_schema
                ),
                system_prompt=PromptTemplates.REFINEMENT_SYSTEM_PROMPT,
                temperature=0.1,  # Low temperature for refinement
                max_tokens=3000  # Generous limit for Qwen thinking + SQL refinement
            )
            
            refined_sql = self._extract_sql_from_response(refinement_response.content)
            return refined_sql
            
        except Exception as e:
            logger.error(f"SQL refinement failed: {e}")
            return sql  # Return original if refinement fails
    
    async def _check_self_consistency(self, candidate: SQLCandidate) -> bool:
        """Check if candidate has achieved self-consistency"""
        try:
            # Re-execute the query
            second_result = await self.sql_executor.execute_sql_safe(candidate.sql)
            
            if not second_result.success:
                return False
            
            # Compare results (simplified consistency check)
            if candidate.execution_result and candidate.execution_result.success:
                first_data = candidate.execution_result.data
                second_data = second_result.data
                
                # Check if row counts match
                if len(first_data or []) == len(second_data or []):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Self-consistency check failed: {e}")
            return False
    
    async def _validate_all_candidates(self) -> str:
        """Validate all generated candidates"""
        if not self.candidates:
            return "No candidates available for validation. Generate candidates first."
        
        validation_results = []
        
        for i, candidate in enumerate(self.candidates):
            try:
                # LLM-based validation
                validation = await self.llm_client.validate_sql_generation(
                    prompt=self.current_request or "SQL generation request",
                    sql=candidate.sql
                )
                
                candidate.validation_score = validation.get('confidence', 0.5)
                
                validation_results.append(f"""
Candidate {i+1} Validation:
- Confidence: {candidate.confidence:.2f}
- Validation Score: {candidate.validation_score:.2f}
- Execution Success: {'✓' if candidate.execution_result and candidate.execution_result.success else '✗'}
- Refinement Iterations: {candidate.iteration}
- Issues: {', '.join(validation.get('issues', []))}
""")
                
            except Exception as e:
                validation_results.append(f"Candidate {i+1}: Validation failed - {e}")
        
        return "Candidate Validation Results:\n" + "\n".join(validation_results)
    
    async def _get_best_candidate(self) -> str:
        """Select and return the best candidate"""
        if not self.candidates:
            return "No candidates available. Generate candidates first."
        
        # Score candidates based on multiple factors
        scored_candidates = []
        
        for i, candidate in enumerate(self.candidates):
            score = self._calculate_candidate_score(candidate)
            scored_candidates.append((score, i, candidate))
        
        # Sort by score (highest first)
        scored_candidates.sort(reverse=True)
        
        best_score, best_idx, best_candidate = scored_candidates[0]
        
        result = f"""Best SQL Candidate (Score: {best_score:.2f}):

```sql
{best_candidate.sql}
```

Candidate Details:
- Confidence: {best_candidate.confidence:.2f}
- Validation Score: {best_candidate.validation_score:.2f}
- Refinement Iterations: {best_candidate.iteration}
- Execution Success: {'✓' if best_candidate.execution_result and best_candidate.execution_result.success else '✗'}

Refinement History:
{chr(10).join(f"- {h}" for h in best_candidate.refinement_history)}
"""
        
        return result
    
    def _calculate_candidate_score(self, candidate: SQLCandidate) -> float:
        """Calculate overall score for a candidate"""
        score = 0.0
        
        # Base confidence score
        score += candidate.confidence * 0.3
        
        # Validation score
        score += candidate.validation_score * 0.3
        
        # Execution success bonus
        if candidate.execution_result and candidate.execution_result.success:
            score += 0.25
        
        # Performance bonus (faster execution)
        if candidate.execution_result and candidate.execution_result.execution_time < 1.0:
            score += 0.1
        
        # Refinement penalty (fewer iterations preferred)
        score -= candidate.iteration * 0.05
        
        return max(0.0, min(1.0, score))
    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from LLM response"""
        # Look for SQL code blocks
        import re
        
        # Try to find SQL in code blocks
        sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        
        # Try to find SQL in generic code blocks
        code_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Look for SELECT statements
        select_match = re.search(r'(SELECT\s+.*?)(?:\n\n|\Z)', response, re.DOTALL | re.IGNORECASE)
        if select_match:
            return select_match.group(1).strip()
        
        # Return cleaned response as fallback
        return response.strip()
    
    def _extract_user_request(self, message: str) -> str:
        """Extract user request from agent message"""
        # Simple extraction - look for request after "user_request:" or similar
        lines = message.split('\n')
        for line in lines:
            if 'request:' in line.lower():
                return line.split(':', 1)[1].strip()
        
        # Fallback to full message
        return message
    
    async def _handle_general_generation_request(self, request: str) -> str:
        """Handle general generation requests"""
        return f"""I'm the SQL Generation and Refinement Agent. I can help with:

1. Generate multiple SQL candidates (generate_candidates)
2. Refine SQL queries based on feedback (refine_sql)
3. Validate generated candidates (validate_candidates)
4. Select the best candidate (get_best_candidate)

Your request: {request}

Please specify what generation task you'd like me to perform, and provide:
- The natural language SQL request
- Compressed database schema (from CompressionAgent)
"""
    
    def get_generation_results(self) -> Dict[str, Any]:
        """Get current generation results for other agents"""
        return {
            "candidates": [
                {
                    "sql": c.sql,
                    "confidence": c.confidence,
                    "validation_score": c.validation_score,
                    "iteration": c.iteration,
                    "success": c.execution_result.success if c.execution_result else False
                }
                for c in self.candidates
            ],
            "best_candidate": self.candidates[0].sql if self.candidates else None,
            "total_candidates": len(self.candidates)
        }