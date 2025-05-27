# OneFileLLM Refactoring Plan: 3-File Architecture with Plugin System

## Overview

This document outlines the plan to refactor OneFileLLM from a single 1700+ line file into a modular 3-file architecture with a plugin system for input processors. This will make the codebase more maintainable, testable, and extensible.

## Current Architecture Problems

1. **Monolithic Structure**: All functionality in one large file (onefilellm.py)
2. **Tight Coupling**: Processing logic mixed with CLI, utilities, and orchestration
3. **Hard to Extend**: Adding new input types requires modifying the main file
4. **Difficult Testing**: Can't easily test individual processors in isolation
5. **Code Duplication**: Similar patterns repeated across processors

## Proposed 3-File Structure

### File 1: `onefilellm.py` (Main Entry Point & Orchestration)
**Purpose**: CLI interface, orchestration, and output management

**Responsibilities**:
- Command-line argument parsing
- Plugin discovery and loading
- Routing inputs to appropriate processors
- Combining outputs from multiple sources
- XML output formatting and structure
- Clipboard operations
- Token counting and reporting
- Alias management system
- Progress reporting

**Key Components**:
```python
# Main orchestration functions
def main()
def process_input(input_path, progress=None, task=None)  # Modified to use plugins
def combine_xml_outputs(outputs)

# Output management
def get_token_count(text, disallowed_special=[], chunk_size=1000)
def escape_xml(text)

# Alias system
def handle_add_alias(args, console)
def handle_alias_from_clipboard(args, console)
def load_alias(alias_name, console)
def resolve_single_input_source(source_string, console)

# Plugin management (new)
def load_processors()
def find_processor_for_input(input_path, processors)
```

### File 2: `processors.py` (Input Processors as Plugins)
**Purpose**: All input type processors implemented as plugin classes

**Base Architecture**:
```python
from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    """Base class for all input processors"""
    
    @staticmethod
    @abstractmethod
    def can_handle(input_path: str) -> bool:
        """Check if this processor can handle the given input"""
        pass
    
    @property
    def priority(self) -> int:
        """Priority for processor selection (higher = checked first)"""
        return 0
    
    @property
    @abstractmethod
    def processor_name(self) -> str:
        """Human-readable name for this processor"""
        pass
    
    @abstractmethod
    def process(self, input_path: str, **kwargs) -> str:
        """Process the input and return XML-formatted content"""
        pass
```

**Processor Classes** (each implementing BaseProcessor):
1. **GitHubRepoProcessor**
   - Handles: GitHub repository URLs
   - Current function: `process_github_repo()`

2. **GitHubPullRequestProcessor**
   - Handles: GitHub PR URLs
   - Current function: `process_github_pull_request()`

3. **GitHubIssueProcessor**
   - Handles: GitHub issue URLs
   - Current function: `process_github_issue()`

4. **ArXivProcessor**
   - Handles: ArXiv paper URLs
   - Current function: `process_arxiv_pdf()`

5. **YouTubeProcessor**
   - Handles: YouTube video URLs
   - Current function: `fetch_youtube_transcript()`

6. **LocalFileProcessor**
   - Handles: Local file paths (PDFs, text files, etc.)
   - Current logic: Part of `process_input()`

7. **LocalDirectoryProcessor**
   - Handles: Local directory paths
   - Current function: `process_local_folder()`

8. **WebCrawlProcessor**
   - Handles: General web URLs
   - Current function: `crawl_and_extract_text()`

9. **SciHubProcessor**
   - Handles: DOI/PMID identifiers
   - Current function: `process_doi_or_pmid()`

10. **TextStreamProcessor**
    - Handles: stdin (`-`) and clipboard (`--clipboard`)
    - Current function: `process_text_stream()`

11. **ExcelProcessor**
    - Handles: .xls/.xlsx files (local or URL)
    - Current function: `excel_to_markdown()`

### File 3: `utils.py` (Shared Utilities)
**Purpose**: Common utilities used across processors

**Categories of Utilities**:

1. **File I/O Utilities**:
   ```python
   def safe_file_read(filepath, fallback_encoding='latin1')
   def read_from_clipboard() -> str | None
   def read_from_stdin() -> str | None
   def is_allowed_filetype(filename)
   def is_excluded_file(filename)
   ```

2. **Network Utilities**:
   ```python
   def download_file(url, target_path)
   def is_same_domain(base_url, new_url)
   def is_within_depth(base_url, current_url, max_depth)
   ```

3. **Text Processing Utilities**:
   ```python
   def detect_text_format(text_sample: str) -> str
   def parse_as_plaintext(text_content: str) -> str
   def parse_as_markdown(text_content: str) -> str
   def parse_as_json(text_content: str) -> str
   def parse_as_html(text_content: str) -> str
   def parse_as_yaml(text_content: str) -> str
   def preprocess_text(input_file, output_file)  # If NLTK enabled
   ```

4. **Specialized Format Utilities**:
   ```python
   def process_ipynb_file(temp_file)
   def excel_to_markdown(file_path, as_dict, sheet_name)
   def process_web_pdf(url)
   ```

## How the Plugin System Works

### 1. Processor Registration
When `processors.py` is imported, all processor classes are automatically available:

```python
# In processors.py
ALL_PROCESSORS = [
    GitHubRepoProcessor,
    GitHubPullRequestProcessor,
    GitHubIssueProcessor,
    ArXivProcessor,
    YouTubeProcessor,
    LocalFileProcessor,
    LocalDirectoryProcessor,
    WebCrawlProcessor,
    SciHubProcessor,
    TextStreamProcessor,
    ExcelProcessor,
]
```

### 2. Input Routing
The main `process_input()` function will:
1. Load all available processors
2. Check each processor's `can_handle()` method
3. Use the first processor that returns True (ordered by priority)
4. Fall back to error if no processor matches

```python
def process_input(input_path, progress=None, task=None):
    processors = load_processors()
    
    # Find appropriate processor
    for processor_class in sorted(processors, key=lambda p: p().priority, reverse=True):
        processor = processor_class()
        if processor.can_handle(input_path):
            if progress and task:
                progress.update(task, description=f"Processing with {processor.processor_name}")
            return processor.process(input_path)
    
    # No processor found
    return f'<source type="unknown"><error>No processor available for: {input_path}</error></source>'
```

### 3. Adding New Processors
To add a new input type:
1. Create a new class in `processors.py` inheriting from `BaseProcessor`
2. Implement the required methods
3. Add it to `ALL_PROCESSORS` list
4. No changes needed to main code!

Example:
```python
class TwitterThreadProcessor(BaseProcessor):
    @staticmethod
    def can_handle(input_path: str) -> bool:
        return "twitter.com" in input_path or "x.com" in input_path
    
    @property
    def processor_name(self) -> str:
        return "Twitter Thread"
    
    def process(self, input_path: str, **kwargs) -> str:
        # Implementation here
        return f'<source type="twitter_thread">...</source>'
```

## Migration Strategy

### Phase 1: Create New Files
1. Create `utils.py` with all utility functions
2. Create `processors.py` with BaseProcessor class
3. Update imports in `onefilellm.py`

### Phase 2: Migrate Processors
1. Convert each `process_*` function into a processor class
2. Move processor-specific logic into the class
3. Update each processor to use utilities from `utils.py`

### Phase 3: Refactor Main
1. Remove migrated functions from `onefilellm.py`
2. Implement plugin loading and routing
3. Update `process_input()` to use the plugin system

### Phase 4: Testing
1. Ensure all existing tests pass
2. Add tests for the plugin system
3. Add tests for individual processors

## Benefits of This Architecture

### 1. **Modularity**
- Clear separation of concerns
- Each file has a specific purpose
- Easier to navigate and understand

### 2. **Extensibility**
- New processors can be added without touching core code
- Processors can be developed and tested independently
- Easy to disable/enable specific processors

### 3. **Maintainability**
- Smaller, focused files are easier to maintain
- Changes to one processor don't affect others
- Utilities can be updated without touching processors

### 4. **Testability**
- Individual processors can be unit tested
- Mock processors can be created for testing
- Utilities can be tested in isolation

### 5. **Reusability**
- Utilities can be used by any processor
- Processors can share common patterns through inheritance
- Base processor class ensures consistency

## Configuration Considerations

### Current Configuration
```python
ENABLE_COMPRESSION_AND_NLTK = False
EXCLUDED_DIRS = ["dist", "node_modules", ".git", "__pycache__"]
allowed_extensions = ['.py', '.txt', '.js', ...]
```

### Proposed Configuration
- Move to a configuration section in `onefilellm.py` or separate `config.py`
- Allow processors to define their own configuration
- Make configuration accessible to all processors through kwargs

## Backward Compatibility

- The command-line interface remains unchanged
- All existing functionality is preserved
- Output format remains the same
- Only internal structure changes

## Future Enhancements

This architecture enables:
1. **Dynamic Plugin Loading**: Load processors from a plugins directory
2. **Processor Pipelines**: Chain multiple processors together
3. **Async Processing**: Process multiple inputs concurrently
4. **Processor Configuration**: Per-processor settings and options
5. **Plugin Package Management**: Install processors as separate packages

## Summary

This refactoring will transform OneFileLLM from a monolithic script into a modular, extensible application while preserving all existing functionality. The plugin architecture will make it easy to add new input types and maintain the codebase as it grows.