# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OneFileLLM is a command-line tool that aggregates content from diverse sources into single, LLM-ready XML files. It processes GitHub repos, web pages, PDFs, YouTube transcripts, local files/directories, and more.

## Key Architecture

### Core Components
- **onefilellm.py**: Main entry point with async processing, source detection, web crawling, and XML output generation
- **utils.py**: Shared utilities for file I/O, text format detection, URL validation, and text parsing
- **extras/web_app.py**: Optional Flask web interface for processing via browser
- **tests/test_all.py**: Comprehensive test suite with 57+ tests covering all major functionality

### Processing Flow
1. Input detection (URL/path/DOI/alias) → Source type identification
2. Async processing orchestration for multiple inputs simultaneously
3. Type-specific processor invoked (e.g., `process_github_repo`, `crawl_and_extract_text`)
4. Content extraction and transformation with format detection
5. XML wrapping with semantic structure (`<onefilellm_output>` root)
6. Token counting and clipboard copy

### Important Patterns
- **Async/await architecture**: Main processing uses asyncio for concurrent operations
- **XML output preserves readability**: Content is NOT XML-escaped to maintain code readability
- **Rich console UI**: All terminal output uses Rich library for consistent formatting
- **Modular utilities**: Common operations factored into utils.py for reusability
- **Environment configuration**: Uses python-dotenv for .env file support

## Common Commands

### Running the Tool
```bash
# Basic usage (interactive prompt)
python onefilellm.py

# Single input
python onefilellm.py https://github.com/user/repo

# Multiple inputs processed concurrently
python onefilellm.py input1.txt https://example.com input2.pdf

# Stream processing
cat file.txt | python onefilellm.py -
python onefilellm.py --clipboard

# Advanced web crawling with configuration
python onefilellm.py https://docs.example.com --crawl-max-depth 4 --crawl-include-pattern "/docs/"
```

### Alias Management 2.0
```bash
# Add single source alias
python onefilellm.py --alias-add repo_name "https://github.com/user/repo"

# Add multi-source alias (MUST quote the command string)
python onefilellm.py --alias-add mcp "https://modelcontextprotocol.io/llms-full.txt https://github.com/modelcontextprotocol/python-sdk"

# Dynamic placeholders
python onefilellm.py --alias-add gh-search "https://github.com/search?q={}"
python onefilellm.py gh-search "machine learning"

# Alias management
python onefilellm.py --alias-list              # Show all aliases
python onefilellm.py --alias-list-core         # Show core aliases only
python onefilellm.py --alias-remove old-alias  # Remove user alias
```

### Running Tests
```bash
# Run all tests (excluding integration/slow tests)
python tests/test_all.py

# Run with network-dependent integration tests
python tests/test_all.py --integration

# Run with slow tests (comprehensive web crawling)
python tests/test_all.py --integration --slow

# Run specific test classes
python -m unittest tests.test_all.TestUtilityFunctions
python -m unittest tests.test_all.TestGitHubProcessing

# Run specific test methods
python -m unittest tests.test_all.TestUtilityFunctions.test_safe_file_read
```

### Development Setup
```bash
# Install dependencies
pip install -U -r requirements.txt

# Set environment variables (optional .env file)
export GITHUB_TOKEN="your_token_here"
export RUN_INTEGRATION_TESTS="true"
export RUN_SLOW_TESTS="false"

# Run web interface (optional)
cd extras && python web_app.py
```

## Alias System Architecture

### Storage
- **User Aliases**: `~/.onefilellm_aliases/aliases.json`
- **Core Aliases**: Defined in `CORE_ALIASES` dict (onefilellm.py:73)

### Key Classes
- **AliasManager** (onefilellm.py:87): Handles all alias operations
  - `load_aliases()`: Loads user and core aliases with precedence
  - `add_or_update_alias()`: Adds/updates user aliases
  - `get_command()`: Retrieves expanded command for an alias
  - `expand_placeholder()`: Handles `{}` token replacement

### Alias Expansion Flow
1. Early expansion in `main()` before full argparse (line 3033)
2. Checks first non-flag argument as potential alias
3. Handles placeholder (`{}`) substitution if present
4. Reconstructs `sys.argv` with expanded command
5. Proceeds with normal argument parsing

## Configuration

### File Type Filtering
Modify `allowed_extensions` in onefilellm.py:
```python
allowed_extensions = ['.py', '.txt', '.js', '.rst', '.sh', '.md', '.pyx', '.html', '.yaml','.json', '.jsonl', '.ipynb', '.h', '.c', '.sql', '.csv']
```

### Directory Exclusion
Modify `EXCLUDED_DIRS` in onefilellm.py:
```python
EXCLUDED_DIRS = ["dist", "node_modules", ".git", "__pycache__"]
```

### Web Crawling Configuration
All crawler settings are CLI-configurable:
```bash
# Depth and limits
--crawl-max-depth 3 --crawl-max-pages 1000

# URL filtering
--crawl-include-pattern "/docs/" --crawl-exclude-pattern "\.(css|js)$"

# Content control
--crawl-include-images --crawl-no-include-code --crawl-include-pdfs

# Performance tuning
--crawl-concurrency 5 --crawl-delay 0.5 --crawl-timeout 30

# Compliance
--crawl-respect-robots --crawl-restrict-path
```

### Environment Variables
```bash
# Core functionality
GITHUB_TOKEN=your_github_token
ENABLE_COMPRESSION_AND_NLTK=false

# Testing configuration
RUN_INTEGRATION_TESTS=true
RUN_SLOW_TESTS=false
```

## Testing Architecture

### Test Organization
- **TestUtilityFunctions**: Tests for utils.py functions (file I/O, format detection, parsing)
- **TestGitHubProcessing**: GitHub repos, PRs, issues (requires GITHUB_TOKEN for integration tests)
- **TestWebCrawling**: Advanced async crawler testing with mock and real scenarios
- **TestStreamProcessing**: stdin, clipboard, format detection
- **TestMultiInput**: Multiple input processing and XML combination
- **TestAliasSystem**: Current alias functionality

### Test Execution Modes
- **Unit tests**: Run without `--integration` flag (uses mocking)
- **Integration tests**: Require `--integration` flag and network access
- **Slow tests**: Require `--slow` flag for comprehensive crawling tests

### Mock Strategy
- External APIs (GitHub, YouTube, ArXiv) mocked for unit tests
- Network requests mocked with predictable responses
- File system operations use temporary directories
- Integration tests hit real endpoints when `--integration` specified

## Key Implementation Notes

### XML Output Structure
- Root element: `<onefilellm_output>`
- Source elements: `<source type="..." [attributes]>`
- Content preservation: Code and text NOT XML-escaped for LLM readability
- Multiple inputs combined into single XML document

### Async Processing
- Main function is async using `asyncio.run()`
- Concurrent processing of multiple inputs
- Async web crawler with `aiohttp` and controlled concurrency
- Progress tracking with Rich progress bars

### Error Handling
- Graceful fallbacks for encoding issues (`safe_file_read`)
- Network timeouts and retry logic in web crawling
- Rich console for user-friendly error messages
- Test environment isolation with temporary directories

### Dependencies
- **Core**: requests, beautifulsoup4, PyPDF2, tiktoken, rich
- **Async**: aiohttp, asyncio (built-in)
- **Text processing**: nltk (optional), PyYAML, pandas
- **Web crawling**: readability-lxml, python-dotenv
- **Development**: unittest, mock (built-in testing framework)