# ReFoRCE Text-to-SQL Implementation

A comprehensive implementation of the ReFoRCE (Refinement, Format Restriction, and Column Exploration) Text-to-SQL system using AutoGen multi-agent framework.

## Overview

ReFoRCE is a state-of-the-art Text-to-SQL system that achieved top results on the Spider 2.0 benchmark. This implementation provides a complete 4-stage pipeline:

1. **Database Information Compression** - Reduces schema size by ~96% using pattern-based table grouping
2. **Candidate Generation with Self-Refinement** - Generates multiple SQL candidates with iterative improvement
3. **Majority Voting and Consensus Enforcement** - Selects best candidates through multi-round voting
4. **Column Exploration** - Handles low-confidence cases through progressive database exploration

## Features

- 🔄 **Multi-Agent Architecture**: Built with AutoGen for robust agent coordination
- 🗜️ **Schema Compression**: Handles large databases (1000+ columns) efficiently  
- 🔄 **Self-Refinement**: Iterative SQL improvement based on execution feedback
- 🗳️ **Consensus Voting**: Multi-round voting for reliable candidate selection
- 🔍 **Column Exploration**: Progressive exploration for ambiguous cases
- 🐘 **PostgreSQL Support**: Optimized for PostgreSQL databases
- 🤖 **vLLM Integration**: Works with self-hosted models via OpenAI-compatible API

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd ReFoRCE-implementation
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your database and vLLM settings
```

## Configuration

### Database Setup
Configure your PostgreSQL database connection in `.env`:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database
DB_USER=postgres
DB_PASSWORD=your_password
```

### vLLM Setup
Configure your vLLM endpoint:

```env
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
VLLM_TEMPERATURE=0.1
```

### ReFoRCE Parameters
Tune the algorithm parameters:

```env
NUM_CANDIDATES=8
CONFIDENCE_THRESHOLD=0.7
MAX_REFINEMENT_ITERATIONS=5
```

## Usage

### Interactive Mode
```bash
python main.py --interactive
```

### Single Request
```bash
python main.py --request "Show all users who registered last month"
```

### Batch Processing
```bash
python main.py --batch queries.txt --output results.json
```

### System Diagnostics
```bash
python main.py --diagnostics
```

## Architecture

### Core Components

- **DatabaseManager**: PostgreSQL connection and schema operations
- **SchemaCompressor**: Pattern-based table grouping and compression
- **LLMClient**: vLLM integration with OpenAI-compatible API
- **SQLExecutor**: Safe SQL execution with validation

### Agent Architecture

- **CompressionAgent**: Stage 1 - Database schema compression
- **GenerationAgent**: Stage 2 - SQL candidate generation and refinement
- **VotingAgent**: Stage 3 - Majority voting and consensus
- **ExplorationAgent**: Stage 4 - Column exploration for low-confidence cases

### Workflow Orchestration

The `ReFoRCEWorkflow` class coordinates all agents through the complete pipeline:

```python
# Initialize workflow
workflow = ReFoRCEWorkflow()

# Process request
result = await workflow.process_text_to_sql_request("Your request here")

# Access results
print(f"SQL: {result.final_sql}")
print(f"Confidence: {result.confidence}")
```

## Example Usage

### Basic Example
```python
import asyncio
from reforce.workflows.reforce_workflow import ReFoRCEWorkflow

async def example():
    workflow = ReFoRCEWorkflow()
    
    result = await workflow.process_text_to_sql_request(
        "Find customers who spent more than $1000 last year"
    )
    
    print(f"Generated SQL: {result.final_sql}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Processing time: {result.processing_time:.2f}s")

asyncio.run(example())
```

### Advanced Example with Custom Configuration
```python
from reforce.workflows.reforce_workflow import ReFoRCEWorkflow
from reforce.core.database_manager import DatabaseManager
from reforce.models.llm_client import LLMClient
from reforce.config.settings import LLMConfig, DatabaseConfig

# Custom configuration
db_config = DatabaseConfig(
    host="custom-host",
    database="custom-db"
)

llm_config = LLMConfig(
    base_url="http://custom-vllm:8000/v1",
    model_name="custom-model"
)

# Initialize with custom config
db_manager = DatabaseManager(db_config)
llm_client = LLMClient(llm_config)
workflow = ReFoRCEWorkflow(db_manager, llm_client)

# Process request
result = await workflow.process_text_to_sql_request("Your query here")
```

## Pipeline Stages

### Stage 1: Database Information Compression
- Analyzes table naming patterns (prefixes, suffixes, dates)
- Groups similar tables together
- Retains representative DDL samples
- Achieves ~96% size reduction for large schemas

### Stage 2: Candidate Generation & Self-Refinement
- Generates k=8 SQL candidates with different approaches
- Iterative refinement based on execution feedback
- Self-consistency checking
- Syntax and semantic error correction

### Stage 3: Majority Voting & Consensus
- Multi-round voting with different evaluation methods
- Weight-based, pairwise comparison, and holistic evaluation
- Confidence scoring and consensus strength measurement
- Automatic routing to exploration if low confidence

### Stage 4: Column Exploration
- Progressive query execution (simple → complex)
- Column content analysis and relationship discovery
- Fuzzy matching and data type inference
- Enhanced SQL generation based on insights

## Performance and Scalability

- **Large Schema Support**: Handles 1000+ column databases efficiently
- **Compression Efficiency**: ~96% schema size reduction
- **Parallel Processing**: Concurrent candidate generation and voting
- **Caching**: Query result caching for improved performance
- **Timeout Handling**: Configurable timeouts for all operations

## Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/ -v
```

Individual component testing:
```bash
# Test database connection
python -c "from reforce.core.database_manager import DatabaseManager; dm = DatabaseManager(); print(dm.get_all_tables())"

# Test LLM connection
python -c "import asyncio; from reforce.models.llm_client import LLMClient; asyncio.run(LLMClient().health_check())"

# Test schema compression
python -c "from reforce.core.schema_compressor import SchemaCompressor; sc = SchemaCompressor(); print(sc.get_table_groups_summary())"
```

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check PostgreSQL is running
   - Verify connection parameters in `.env`
   - Ensure database exists and user has permissions

2. **vLLM Connection Failed**
   - Verify vLLM service is running at specified URL
   - Check model name matches your vLLM deployment
   - Ensure API endpoint is accessible

3. **Low Confidence Results**
   - Check if Column Exploration is being triggered
   - Review database schema complexity
   - Consider adjusting confidence threshold

4. **Performance Issues**
   - Monitor database query performance
   - Check vLLM response times
   - Consider reducing number of candidates

### Debugging

Enable verbose logging:
```bash
python main.py --verbose --diagnostics
```

Check the `reforce.log` file for detailed execution logs.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

This implementation is based on the ReFoRCE research paper and is provided for educational and research purposes.

## References

- [ReFoRCE Paper](https://arxiv.org/pdf/2502.00675)
- [Original ReFoRCE Repository](https://github.com/Snowflake-Labs/ReFoRCE)
- [AutoGen Framework](https://github.com/microsoft/autogen)
- [Spider 2.0 Benchmark](https://spider2-sql.github.io/)

## Acknowledgments

- ReFoRCE research team for the original algorithm
- Microsoft AutoGen team for the multi-agent framework
- vLLM project for high-performance LLM serving