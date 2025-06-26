"""
Configuration settings for ReFoRCE Text-to-SQL system
"""
import os
from typing import Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration"""
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    database: str = os.getenv("DB_NAME", "reforce_db")
    username: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "")
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class LLMConfig:
    """vLLM OpenAI-compatible API configuration"""
    base_url: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key: str = os.getenv("VLLM_API_KEY", "not-needed")
    model_name: str = os.getenv("VLLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    temperature: float = float(os.getenv("VLLM_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("VLLM_MAX_TOKENS", "4096"))
    timeout: int = int(os.getenv("VLLM_TIMEOUT", "60"))

@dataclass
class ReFoRCEConfig:
    """ReFoRCE algorithm configuration"""
    # Database compression settings
    compression_threshold: int = int(os.getenv("COMPRESSION_THRESHOLD", "100000"))  # 100KB
    max_tables_per_group: int = int(os.getenv("MAX_TABLES_PER_GROUP", "50"))
    
    # Candidate generation settings
    num_candidates: int = int(os.getenv("NUM_CANDIDATES", "8"))
    max_refinement_iterations: int = int(os.getenv("MAX_REFINEMENT_ITERATIONS", "5"))
    
    # Voting and consensus settings
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    min_votes_required: int = int(os.getenv("MIN_VOTES_REQUIRED", "3"))
    
    # Column exploration settings
    max_exploration_queries: int = int(os.getenv("MAX_EXPLORATION_QUERIES", "10"))
    exploration_result_limit: int = int(os.getenv("EXPLORATION_RESULT_LIMIT", "100"))
    exploration_size_limit: str = os.getenv("EXPLORATION_SIZE_LIMIT", "5KB")

class Settings:
    """Global settings container"""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.llm = LLMConfig()
        self.reforce = ReFoRCEConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "username": self.database.username
            },
            "llm": {
                "base_url": self.llm.base_url,
                "model_name": self.llm.model_name,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens
            },
            "reforce": {
                "num_candidates": self.reforce.num_candidates,
                "confidence_threshold": self.reforce.confidence_threshold,
                "max_refinement_iterations": self.reforce.max_refinement_iterations
            }
        }

# Global settings instance
settings = Settings()