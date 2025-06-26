# ReFoRCE Text-to-SQL Implementation - Claude AI Assistant Guide

This document provides context and instructions for AI assistants working with the ReFoRCE Text-to-SQL system.

## Project Overview

This is a complete implementation of the ReFoRCE (Refinement, Format Restriction, and Column Exploration) Text-to-SQL system using the AutoGen multi-agent framework. ReFoRCE is a state-of-the-art approach that achieved top results on the Spider 2.0 benchmark by implementing a sophisticated 4-stage pipeline.

## Architecture Summary

### Four-Stage Pipeline

1. **Database Information Compression (Stage 1)**
   - Pattern-based table grouping using prefix/suffix matching
   - DDL compression retaining representative samples
   - Achieves ~96% schema size reduction for large databases
   - Handles databases with 1000+ columns efficiently

2. **Candidate Generation with Self-Refinement (Stage 2)**
   - Generates k=8 SQL candidates using different temperature settings
   - Iterative refinement based on execution feedback
   - Self-consistency checking (terminates when same result achieved twice)
   - Syntax and semantic error correction

3. **Majority Voting and Consensus Enforcement (Stage 3)**
   - Multi-round voting: weight-based, pairwise comparison, holistic evaluation
   - Confidence scoring and consensus strength measurement
   - Automatic routing to Stage 4 if confidence below threshold (0.7)

4. **Column Exploration (Stage 4)**
   - Progressive query execution from simple to complex
   - Column content analysis and relationship discovery
   - Fuzzy matching and data type inference
   - Enhanced SQL generation based on discovered insights

### Core Components

- **DatabaseManager**: PostgreSQL connection, schema operations, async/sync query execution
- **SchemaCompressor**: Pattern detection, table grouping, compression algorithms
- **LLMClient**: vLLM integration via OpenAI-compatible API with retry logic
- **SQLExecutor**: Safe execution with validation, caching, performance analysis
- **PromptTemplates**: Structured prompts for each pipeline stage

### AutoGen Agent Architecture

- **CompressionAgent**: Orchestrates schema compression and pattern analysis
- **GenerationAgent**: Manages candidate generation and self-refinement cycles
- **VotingAgent**: Implements voting mechanisms and consensus determination
- **ExplorationAgent**: Handles column exploration for low-confidence cases
- **ReFoRCEWorkflow**: Main orchestrator coordinating all agents

## Key Implementation Details

### Database Schema Compression
- Identifies patterns: prefixes (GA_SESSIONS_*), suffixes (*_LOG), dates (TABLE_20240101), numerics (TABLE_1)
- Selects representative tables avoiding temporal/numeric suffixes
- Creates compressed DDL with metadata about grouped tables
- Maintains essential schema information while reducing context size

### Self-Refinement Process
- Executes candidates and captures detailed error feedback
- Uses LLM to analyze errors and suggest corrections
- Implements consecutive failure detection (stops after 2 failures)
- Tracks refinement history and iteration counts

### Voting Mechanisms
- **Weight-based**: Scores based on correctness, confidence, execution success
- **Pairwise**: Compares candidates head-to-head based on combined metrics
- **Holistic**: LLM evaluates all candidates together for best selection
- Aggregates votes across rounds and calculates consensus strength

### Column Exploration Strategy
- Generates exploratory queries with progressive complexity (1-5 scale)
- Limits results to 100 rows/5KB per query to avoid overwhelming
- Extracts insights: data types, sample values, null counts, unique counts
- Uses insights to generate improved SQL candidates

## Configuration and Setup

### Environment Variables
```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database
DB_USER=postgres
DB_PASSWORD=your_password

# vLLM
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
VLLM_TEMPERATURE=0.1

# Algorithm Parameters
NUM_CANDIDATES=8
CONFIDENCE_THRESHOLD=0.7
MAX_REFINEMENT_ITERATIONS=5
```

### Usage Patterns
```bash
# Interactive mode
python main.py --interactive

# Single request
python main.py --request "Show all users who registered last month"

# Batch processing
python main.py --batch queries.txt --output results.json

# System diagnostics
python main.py --diagnostics
```

## Common Tasks for AI Assistants

### Debugging Issues

1. **Database Connection Problems**
   - Check `reforce.log` for connection errors
   - Verify PostgreSQL service status
   - Test connection: `python -c "from reforce.core.database_manager import DatabaseManager; dm = DatabaseManager(); print(dm.get_all_tables())"`

2. **vLLM Service Issues**
   - Verify service at configured URL
   - Test health: `python -c "import asyncio; from reforce.models.llm_client import LLMClient; asyncio.run(LLMClient().health_check())"`

3. **Low Confidence Results**
   - Review if Column Exploration was triggered
   - Check compression ratio in logs
   - Examine candidate generation diversity

4. **Performance Issues**
   - Monitor query execution times in logs
   - Check LLM response latency
   - Consider reducing NUM_CANDIDATES

### Code Modifications

When modifying the system:

1. **Adding New Agents**: Extend `AssistantAgent` base class, implement `on_messages()` method
2. **Custom Prompts**: Modify `PromptTemplates` class for new prompt patterns
3. **Database Support**: Extend `DatabaseManager` for other database types
4. **LLM Providers**: Modify `LLMClient` for different API endpoints

### Testing and Validation

The system includes comprehensive validation:
- Security validation (dangerous SQL patterns)
- Syntax validation (balanced parentheses, required keywords)
- Schema validation (column/table references)
- Performance validation (SELECT *, missing WHERE clauses)

Test individual components:
```bash
# Test compression
python -c "from reforce.core.schema_compressor import SchemaCompressor; sc = SchemaCompressor(); print(sc.get_table_groups_summary())"

# Test validation
python -c "from reforce.utils.validation import SQLValidator; print(SQLValidator.validate_sql_syntax('SELECT * FROM users'))"
```

## Performance Characteristics

- **Schema Compression**: Reduces 50MB schemas to <100KB (96% reduction)
- **Candidate Generation**: Generates 8 candidates in ~10-15 seconds
- **Voting Process**: Multi-round evaluation in ~5-10 seconds
- **Column Exploration**: 5-10 queries executed progressively
- **Total Pipeline**: Typically 30-60 seconds for complex requests

## Integration Points

### With External Systems
- PostgreSQL database (primary data source)
- vLLM service (LLM inference)
- AutoGen framework (agent coordination)

### API Compatibility
- OpenAI-compatible endpoints for LLM integration
- Standard PostgreSQL connection protocols
- Async/await patterns throughout for concurrency

## Troubleshooting Commands

```bash
# Full system health check
python main.py --diagnostics

# Verbose logging
python main.py --verbose --request "test query"

# Check configuration
python -c "from reforce.config.settings import settings; print(settings.to_dict())"

# Test specific components
python -c "from reforce.workflows.reforce_workflow import ReFoRCEWorkflow; import asyncio; workflow = ReFoRCEWorkflow(); asyncio.run(workflow.health_check())"
```

## File Structure Context

```
reforce/
├── agents/          # AutoGen agents for each pipeline stage
├── core/           # Core functionality (DB, compression, execution)
├── models/         # LLM client and prompt templates
├── workflows/      # Main pipeline orchestration
├── utils/          # Utilities (validation, schema analysis)
├── config/         # Configuration management
└── tests/          # Integration and unit tests

main.py             # CLI entry point
requirements.txt    # Python dependencies
.env.example        # Configuration template
```

This implementation faithfully reproduces the ReFoRCE algorithm described in the research paper while adapting it for practical use with PostgreSQL and self-hosted LLM deployments.