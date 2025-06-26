"""
vLLM OpenAI-compatible client for ReFoRCE Text-to-SQL system
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import openai
from openai import AsyncOpenAI
from ..config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Response from LLM containing generated content and metadata"""
    content: str
    usage: Dict[str, int]
    model: str
    finish_reason: str
    confidence: Optional[float] = None

class LLMClient:
    """Client for interacting with vLLM via OpenAI-compatible API"""
    
    def __init__(self, config=None):
        self.config = config or settings.llm
        self.client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key
        )
        self.sync_client = openai.OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key
        )
    
    async def generate_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> LLMResponse:
        """Generate single completion from prompt"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                stop=stop_sequences
            )
            
            choice = response.choices[0]
            return LLMResponse(
                content=choice.message.content,
                usage=response.usage.model_dump() if response.usage else {},
                model=response.model,
                finish_reason=choice.finish_reason
            )
            
        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            raise
    
    def generate_completion_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """Synchronous completion generation"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.sync_client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens
            )
            
            choice = response.choices[0]
            return LLMResponse(
                content=choice.message.content,
                usage=response.usage.model_dump() if response.usage else {},
                model=response.model,
                finish_reason=choice.finish_reason
            )
            
        except Exception as e:
            logger.error(f"Sync LLM completion failed: {e}")
            raise
    
    async def generate_multiple_completions(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        num_completions: int = 3,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> List[LLMResponse]:
        """Generate multiple completions concurrently"""
        tasks = []
        for _ in range(num_completions):
            task = self.generate_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            tasks.append(task)
        
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            valid_responses = []
            
            for response in responses:
                if isinstance(response, Exception):
                    logger.error(f"Completion failed: {response}")
                else:
                    valid_responses.append(response)
            
            return valid_responses
            
        except Exception as e:
            logger.error(f"Multiple completions failed: {e}")
            raise
    
    async def generate_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        temperature: Optional[float] = None
    ) -> LLMResponse:
        """Generate completion with retry logic"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return await self.generate_completion(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature
                )
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} attempts failed")
        
        raise last_exception
    
    async def estimate_confidence(self, prompt: str, response: str) -> float:
        """
        Estimate confidence of response using a separate validation prompt
        """
        confidence_prompt = f"""
        Given the following prompt and response, rate the confidence of the response on a scale of 0.0 to 1.0.
        Consider factors like:
        - Completeness of the response
        - Correctness based on the prompt
        - Specificity and detail level
        
        Prompt: {prompt[:500]}...
        Response: {response[:500]}...
        
        Provide only a numerical confidence score between 0.0 and 1.0:
        """
        
        try:
            confidence_response = await self.generate_completion(
                prompt=confidence_prompt,
                temperature=0.1,
                max_tokens=10
            )
            
            # Parse confidence score
            confidence_text = confidence_response.content.strip()
            try:
                confidence = float(confidence_text)
                return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            except ValueError:
                logger.warning(f"Could not parse confidence score: {confidence_text}")
                return 0.5  # Default medium confidence
                
        except Exception as e:
            logger.error(f"Confidence estimation failed: {e}")
            return 0.5  # Default medium confidence
    
    async def validate_sql_generation(self, prompt: str, sql: str) -> Dict[str, Any]:
        """
        Validate generated SQL against the original prompt
        """
        validation_prompt = f"""
        Analyze the following SQL query against the original request:
        
        Original Request: {prompt}
        Generated SQL: {sql}
        
        Evaluate:
        1. Does the SQL correctly address the request?
        2. Are there any syntax errors?
        3. Are the table/column references appropriate?
        4. Is the query structure logical?
        
        Respond in JSON format:
        {{
            "is_valid": true/false,
            "confidence": 0.0-1.0,
            "issues": ["list", "of", "issues"],
            "suggestions": ["list", "of", "improvements"]
        }}
        """
        
        try:
            response = await self.generate_completion(
                prompt=validation_prompt,
                temperature=0.1,
                max_tokens=512
            )
            
            # Try to parse JSON response
            import json
            try:
                validation_result = json.loads(response.content)
                return validation_result
            except json.JSONDecodeError:
                logger.warning(f"Could not parse validation JSON: {response.content}")
                return {
                    "is_valid": False,
                    "confidence": 0.3,
                    "issues": ["Could not validate response"],
                    "suggestions": ["Manual review required"]
                }
                
        except Exception as e:
            logger.error(f"SQL validation failed: {e}")
            return {
                "is_valid": False,
                "confidence": 0.0,
                "issues": [f"Validation error: {str(e)}"],
                "suggestions": ["Retry validation"]
            }
    
    async def health_check(self) -> bool:
        """Check if the LLM service is healthy"""
        try:
            test_response = await self.generate_completion(
                prompt="Return 'OK' if you can process this request.",
                temperature=0.0,
                max_tokens=10
            )
            return "OK" in test_response.content.upper()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        return {
            "base_url": self.config.base_url,
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout
        }