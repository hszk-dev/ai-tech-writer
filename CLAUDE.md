# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Tech Writer is an AI-powered technical article generator for Japanese tech platforms (Zenn/Qiita). It uses a multi-stage pipeline to generate complete articles from just a topic.

## Commands

```bash
# Install dependencies (development mode)
uv pip install -e .
# or
pip install -e .

# Generate an article
ai-tech-writer generate "LangChainでRAGを実装する"
ai-tech-writer generate "Next.js 15の新機能" --platform qiita
ai-tech-writer generate "Rust入門" --model gpt-4o
ai-tech-writer generate "Pythonの基本" --skip-search  # Skip web search
ai-tech-writer generate "TypeScript Tips" --skip-review  # Skip review stage
ai-tech-writer generate "topic" --verbose  # Show detailed progress

# Other commands
ai-tech-writer list-models
ai-tech-writer init

# Linting and type checking
ruff check src/
ruff format src/
mypy src/
```

## Environment Variables

Required in `.env`:
- `ANTHROPIC_API_KEY` - For Claude models (or `OPENAI_API_KEY` for GPT models)
- `TAVILY_API_KEY` - For web search in ideation stage

## Architecture

### Pipeline System

The core architecture is a sequential pipeline in `src/ai_tech_writer/pipeline/`:

1. **IdeationStage** - Takes topic, searches web (Tavily), generates article idea (title, emoji, topics, target audience)
2. **OutlineStage** - Creates detailed outline with sections and key points
3. **DraftStage** - Writes full article content with code examples
4. **ReviewStage** - Reviews and optionally improves the draft

Each stage extends `PipelineStage` (base.py) with `execute()` method. `ArticlePipeline` (orchestrator.py) runs stages sequentially, passing `StageContext` with shared resources.

### Key Components

- **LLMClient** (`llm/client.py`) - Uses LiteLLM for multi-provider support (Claude, GPT). Has `complete()` and `complete_json()` async methods.
- **PromptLoader** (`llm/prompts.py`) - Loads Jinja2 prompt templates from `config/prompts/`
- **Domain Models** (`models/article.py`) - `Article`, `ArticleFrontmatter`, `ArticleSection`, `ArticleIdea`, `ArticleOutline`
- **MarkdownRenderer** (`output/markdown.py`) - Renders articles with platform-specific frontmatter (Zenn vs Qiita)

### Configuration

- `config/default.yaml` - Main config (model, temperature, web search settings)
- `config/prompts/*.md` - Prompt templates for each pipeline stage
- `templates/{zenn,qiita}/article.md.j2` - Jinja2 templates for output formatting
