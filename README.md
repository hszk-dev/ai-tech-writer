# AI Tech Writer

AI-powered technical article writer for Zenn/Qiita, inspired by [AI Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2).

## Features

- **Automated Article Generation**: Generate complete technical articles from just a topic
- **Multi-Stage Pipeline**: Ideation → Outline → Draft → Review
- **Web Search Integration**: Uses Tavily API to gather latest information
- **Multiple Platforms**: Supports Zenn and Qiita markdown formats
- **Quality Review**: AI-powered review and improvement suggestions

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ai-tech-writer.git
cd ai-tech-writer

# Install dependencies
pip install -e .

# Or with uv
uv pip install -e .
```

## Setup

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Add your API keys to `.env`:
```bash
ANTHROPIC_API_KEY=sk-ant-xxx   # Required: Claude 3.5 Sonnet
TAVILY_API_KEY=tvly-xxx        # Required: Web search
```

## Usage

### Generate an article

```bash
# Basic usage
ai-tech-writer generate "LangChainでRAGを実装する"

# Specify platform
ai-tech-writer generate "Next.js 15の新機能" --platform qiita

# Use different model
ai-tech-writer generate "Rust入門" --model gpt-4o

# Skip web search (faster, but less current info)
ai-tech-writer generate "Pythonの基本" --skip-search

# Skip review stage
ai-tech-writer generate "TypeScript Tips" --skip-review
```

### Other commands

```bash
# List available models
ai-tech-writer list-models

# Initialize project structure
ai-tech-writer init
```

## Pipeline Stages

### 1. Ideation
- Takes a topic as input
- Searches the web for related information
- Generates article idea with title, target audience, and structure

### 2. Outline
- Creates detailed outline from the idea
- Defines key points for each section
- Identifies sections that need code examples

### 3. Draft
- Writes full article content
- Generates code examples where needed
- Creates proper markdown structure

### 4. Review
- Reviews the draft for quality
- Suggests improvements
- Optionally applies improvements

## Configuration

Configuration is stored in `config/default.yaml`:

```yaml
llm:
  default_model: "claude-3-5-sonnet-20241022"
  temperature: 0.7
  max_tokens: 4096

output:
  platform: "zenn"
  output_dir: "outputs"

web_search:
  provider: "tavily"
  num_results: 10
```

## Project Structure

```
ai-tech-writer/
├── src/ai_tech_writer/
│   ├── cli.py              # CLI entry point
│   ├── models/             # Domain models
│   ├── llm/                # LLM client
│   ├── pipeline/           # Pipeline stages
│   ├── web/                # Web search
│   └── output/             # Markdown rendering
├── config/
│   ├── default.yaml        # Default config
│   └── prompts/            # Prompt templates
├── templates/              # Jinja2 templates
│   ├── zenn/
│   └── qiita/
└── outputs/                # Generated articles
```

## License

MIT
