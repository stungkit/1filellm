# OneFileLLM: Efficient Data Aggregation for LLM Ingestion
OneFileLLM is a command-line tool designed to streamline the creation of information-dense prompts for large language models (LLMs). It aggregates and preprocesses data from a variety of sources, compiling them into a single text file that is automatically copied to your clipboard for quick use.

## Features

- Automatic source type detection based on provided path, URL, or identifier
- Support for local files and/or directories, GitHub repositories, GitHub pull requests, GitHub issues, academic papers from ArXiv, YouTube transcripts, web page documentation, Sci-Hub hosted papers via DOI or PMID
- Handling of multiple file formats, including Jupyter Notebooks (.ipynb), PDFs, and Excel files (.xls/.xlsx)
- Web crawling functionality to extract content from linked pages up to a specified depth
- Integration with Sci-Hub for automatic downloading of research papers using DOIs or PMIDs
- Text preprocessing, including compressed and uncompressed outputs, stopword removal, and lowercase conversion
- Automatic copying of uncompressed text to the clipboard for easy pasting into LLMs
- Token count reporting for both compressed and uncompressed outputs
- XML encapsulation of output for improved LLM performance
- Text stream input processing directly from stdin or clipboard
- Format detection and processing for plain text, Markdown, JSON, HTML, and YAML
- Format override option to control input processing
- Excel spreadsheet (.xls/.xlsx) processing to Markdown tables
- Alias system for frequently used sources
- Proper PDF text extraction from local files
- Cross-platform launcher scripts for easy execution

![image](https://github.com/jimmc414/1filellm/assets/6346529/73c24bcb-7be7-4b67-8591-3f1404b98fba)

## Installation

### Prerequisites

Install the required dependencies:

```bash
pip install -U -r requirements.txt
```

### GitHub Personal Access Token

To access private GitHub repositories, generate a personal access token as described in the 'Obtaining a GitHub Personal Access Token' section here: 

[Obtaining a GitHub Personal Access Token](https://github.com/jjmc414/onefilellm?tab=readme-ov-file#obtaining-a-github-personal-access-token)
### Setup

Clone the repository or download the source code.

## Usage

Run the script using the following command:

```bash
python onefilellm.py
```

![image](https://github.com/jimmc414/1filellm/assets/6346529/b4e281eb-8a41-4612-9d73-b2c115691013)


You can pass a single URL or path as a command line argument for faster processing:

```bash
python onefilellm.py https://github.com/jimmc414/1filellm
```

### Text Stream Processing

OneFileLLM now supports processing text directly from standard input (stdin) or the system clipboard:

#### Processing from Standard Input

Use the `-` argument to process text from standard input:

```bash
# Process text from a file via pipe
cat README.md | python onefilellm.py -

# Process output from another command
git diff | python onefilellm.py -
```

#### Processing from Clipboard

Use the `--clipboard` or `-c` argument to process text from the system clipboard:

```bash
# Copy text to clipboard first, then run:
python onefilellm.py --clipboard

# Or using the short form:
python onefilellm.py -c
```

#### Format Detection and Override

OneFileLLM automatically detects the format of input text (plain text, Markdown, JSON, HTML, YAML) and processes it accordingly. You can override this detection with the `--format` or `-f` option:

```bash
# Force processing as JSON
cat data.txt | python onefilellm.py - --format json

# Force processing clipboard content as Markdown 
python onefilellm.py --clipboard -f markdown
```

Supported format types: `text`, `markdown`, `json`, `html`, `yaml`, `doculing`, `markitdown`

### Multiple Inputs

OneFileLLM supports processing multiple inputs at once. Simply provide multiple paths or URLs as command line arguments:

```bash
python onefilellm.py https://github.com/jimmc414/1filellm test_file1.txt test_file2.txt
```

When multiple inputs are provided, OneFileLLM will:
1. Process each input separately according to its type
2. Combine all outputs into a single XML document with the `<onefilellm_output>` root tag
3. Save the combined output to `output.xml`
4. Copy the content to your clipboard for immediate use with LLMs

### Using Aliases

OneFileLLM now includes an alias system to save you from typing the same URLs or paths repeatedly:

#### Creating Aliases

You can create aliases for single or multiple sources:

```bash
# Create an alias for a single source
python onefilellm.py --add-alias github_repo https://github.com/jimmc414/onefilellm

# Create an alias for multiple sources
python onefilellm.py --add-alias mixed_sources test_file.txt https://github.com/jimmc414/onefilellm/blob/main/README.md
```

#### Creating Aliases from Clipboard

You can also create aliases from content in your clipboard, with one source per line:

```bash
# First copy multiple URLs or paths to your clipboard (one per line)
python onefilellm.py --alias-from-clipboard research_sources
```

#### Using Aliases

Use your defined aliases just like any other input:

```bash
# Use a single alias
python onefilellm.py github_repo

# Mix aliases with direct sources
python onefilellm.py github_repo test_file.txt

# Use multiple aliases together
python onefilellm.py github_repo research_sources
```

Aliases are stored in your home directory at `~/.onefilellm_aliases/` for easy access from any location.

### Expected Inputs and Resulting Outputs
The tool supports the following input options:

- Local file path (e.g., C:\documents\report.pdf)
- Local directory path (e.g., C:\projects\research) -> (files of selected filetypes segmented into one flat text file)
- GitHub repository URL (e.g., https://github.com/jimmc414/onefilellm) -> (Repo files of selected filetypes segmented into one flat text file)
- GitHub pull request URL (e.g., https://github.com/dear-github/dear-github/pull/102) -> (Pull request diff detail and comments and entire repository content concatenated into one flat text file)
- GitHub issue URL (e.g., https://github.com/isaacs/github/issues/1191) -> (Issue details, comments, and entire repository content concatenated into one flat text file)
- ArXiv paper URL (e.g., https://arxiv.org/abs/2401.14295) -> (Full paper PDF to text file)
- YouTube video URL (e.g., https://www.youtube.com/watch?v=KZ_NlnmPQYk) -> (Video transcript to text file)
- Webpage URL (e.g., https://llm.datasette.io/en/stable/) -> (To scrape pages to x depth in segmented text file)
- Sci-Hub Paper DOI (Digital Object Identifier of Sci-Hub hosted paper) (e.g., 10.1053/j.ajkd.2017.08.002) -> (Full Sci-Hub paper PDF to text file)
- Sci-Hub Paper PMID (PubMed Identifier of Sci-Hub hosted paper) (e.g., 29203127) -> (Full Sci-Hub paper PDF to text file)
- Standard input via pipe (e.g., `cat file.txt | python onefilellm.py -`)
- Clipboard content (e.g., `python onefilellm.py --clipboard`)

The tool supports the following input options, with their corresponding output actions. Note that the input file extensions are selected based on the following section of code (Applicable to Repos only):

```python
allowed_extensions = ['.xyz', '.pdq', '.example']
```

**The output for all options is encapsulated in LLM prompt-appropriate XML and automatically copied to the clipboard.**

1. **Local file path**
   - **Example Input:** `C:\documents\report.pdf`
   - **Output:** The contents of the PDF file are extracted and saved into a single text file.

2. **Local directory path**
   - **Example Input:** `C:\projects\research`
   - **Output:** Files of selected file types within the directory are segmented and saved into a single flat text file.

3. **GitHub repository URL**
   - **Example Input:** `https://github.com/jimmc414/onefilellm`
   - **Output:** Repository files of selected file types are segmented and saved into a single flat text file.

4. **GitHub pull request URL**
   - **Example Input:** `https://github.com/dear-github/dear-github/pull/102`
   - **Output:** Pull request diff details, comments, and the entire repository content are concatenated into a single flat text file.

5. **GitHub issue URL**
   - **Example Input:** `https://github.com/isaacs/github/issues/1191`
   - **Output:** Issue details, comments, and the entire repository content are concatenated into a single flat text file.

6. **ArXiv paper URL**
   - **Example Input:** `https://arxiv.org/abs/2401.14295`
   - **Output:** The full paper PDF is converted into a text file.

7. **YouTube video URL**
   - **Example Input:** `https://www.youtube.com/watch?v=KZ_NlnmPQYk`
   - **Output:** The video transcript is extracted and saved into a text file.

8. **Webpage URL**
   - **Example Input:** `https://llm.datasette.io/en/stable/`
   - **Output:** The webpage content and linked pages up to a specified depth are scraped and segmented into a text file.

9. **Sci-Hub Paper DOI**
   - **Example Input:** `10.1053/j.ajkd.2017.08.002`
   - **Output:** The full Sci-Hub paper PDF is converted into a text file.

10. **Sci-Hub Paper PMID**
    - **Example Input:** `29203127`
    - **Output:** The full Sci-Hub paper PDF is converted into a text file.

11. **Standard Input**
    - **Example Input:** `cat file.txt | python onefilellm.py -`
    - **Output:** The piped text content is processed according to its detected format (or format override).

12. **Clipboard**
    - **Example Input:** `python onefilellm.py --clipboard`
    - **Output:** The clipboard text content is processed according to its detected format (or format override).

The script generates the following output files:

- `output.xml`: The full XML-structured output, automatically copied to the clipboard.
- `compressed_output.txt`: Cleaned and compressed text (when NLTK processing is enabled).
- `processed_urls.txt`: A list of all processed URLs during web crawling.

## Configuration

- To modify the allowed file types for repository processing, update the `allowed_extensions` list in the code.
- To change the depth of web crawling, adjust the `max_depth` variable in the code.

## Obtaining a GitHub Personal Access Token

To access private GitHub repositories, you need a personal access token. Follow these steps:

1. Log in to your GitHub account and go to Settings.
2. Navigate to Developer settings > Personal access tokens.
3. Click on "Generate new token" and provide a name.
4. Select the necessary scopes (at least `repo` for private repositories).
5. Click "Generate token" and copy the token value.

In the `onefilellm.py` script, replace `GITHUB_TOKEN` with your actual token or set it as an environment variable:

- For Windows:
  ```shell
  setx GITHUB_TOKEN "YourGitHubToken"
  ```

- For Linux:
  ```shell
  echo 'export GITHUB_TOKEN="YourGitHubToken"' >> ~/.bashrc
  source ~/.bashrc
  ```

## XML Output Format

All output is encapsulated in XML tags. This structure was implemented based on evaluations showing that LLMs perform better with prompts structured in XML. The general structure of the output is as follows:

### Single Source Output

```xml
<onefilellm_output>
  <source type="[source_type]" [additional_attributes]>
    <[content_type]>
      [Extracted content]
    </[content_type]>
  </source>
</onefilellm_output>
```

### Multiple Sources Output

```xml
<onefilellm_output>
  <source type="[source_type_1]" [additional_attributes]>
    <[content_type]>
      [Extracted content 1]
    </[content_type]>
  </source>
  <source type="[source_type_2]" [additional_attributes]>
    <[content_type]>
      [Extracted content 2]
    </[content_type]>
  </source>
  <!-- Additional sources as needed -->
</onefilellm_output>
```

Where `[source_type]` could be one of: "github_repository", "github_pull_request", "github_issue", "arxiv_paper", "youtube_transcript", "web_documentation", "sci_hub_paper", "local_directory", "local_file", "stdin", or "clipboard".

This XML structure provides clear delineation of different content types and sources, improving the LLM's understanding and processing of the input.


## Data Flow Diagram

```
                                 +--------------------------------+
                                 |      External Services         |
                                 |--------------------------------|
                                 |  GitHub API  | YouTube API     |
                                 |  Sci-Hub     | ArXiv           |
                                 +--------------------------------+
                                           |
                                           |
                                           v
 +----------------------+          +---------------------+         +----------------------+
 |                      |          |                     |         |                      |
 |        User          |          |  Command Line Tool  |         |  External Libraries  |
 |----------------------|          |---------------------|         |----------------------|
 | - Provides input URL |--------->| - Handles user input|         | - Requests           |
 | - Provides text via  |          | - Detects source    |<--------| - BeautifulSoup      |
 |   pipe or clipboard  |          |   type              |         | - PyPDF2             |
 | - Receives text      |          | - Calls appropriate |         | - Tiktoken           |
 |   in clipboard       |<---------| - processing modules|         | - NLTK               |
 |                      |          | - Preprocesses text |         | - Nbformat           |
 +----------------------+          | - Generates output  |         | - Nbconvert          |
                                   |   files             |         | - YouTube Transcript |
                                   | - Copies text to    |         |   API                |
                                   |   clipboard         |         | - Pyperclip          |
                                   | - Reports token     |         | - Wget               |
                                   |   count             |         | - Tqdm               |
                                   +---------------------+         | - Rich               |
                                           |                       | - PyYAML             |
                                           |                       +----------------------+
                                           v
                                    +---------------------+
                                    | Source Type         |
                                    | Detection           |
                                    |---------------------|
                                    | - Determines type   |
                                    |   of source         |
                                    +---------------------+
                                           |
                                           v
                                    +---------------------+
                                    | Processing Modules  |
                                    |---------------------|
                                    | - GitHub Repo Proc  |
                                    | - Local Dir Proc    |
                                    | - YouTube Transcript|
                                    |   Proc              |
                                    | - ArXiv PDF Proc    |
                                    | - Sci-Hub Paper Proc|
                                    | - Webpage Crawling  |
                                    |   Proc              |
                                    | - Text Stream Proc  |
                                    +---------------------+
                                           |
                                           v
                                    +---------------------+
                                    | Text Preprocessing  |
                                    |---------------------|
                                    | - Stopword removal  |
                                    | - Lowercase         |
                                    |   conversion        |
                                    | - Text cleaning     |
                                    +---------------------+
                                           |
                                           v
                                    +---------------------+
                                    | Output Generation   |
                                    |---------------------|
                                    | - Compressed text   |
                                    |   file output       |
                                    | - Uncompressed text |
                                    |   file output       |
                                    +---------------------+
                                           |
                                           v
                                    +---------------------+
                                    | Token Count         |
                                    | Reporting           |
                                    |---------------------|
                                    | - Report token count|
                                    |                     |
                                    | - Copies text to    |
                                    |   clipboard         |
                                    +---------------------+
```



## Recent Changes

- **2025-05-14:**
  - Added text stream input processing directly from stdin or clipboard
  - Implemented format detection for plain text, Markdown, JSON, HTML, and YAML
  - Added format override option with `--format TYPE` or `-f TYPE` flags
  - Updated help messages and error handling for stream processing
  - Added comprehensive test suite for stream processing features

- **2025-05-10:**
  - Added Excel spreadsheet (.xls/.xlsx) processing with conversion to Markdown tables
  - Support for both local Excel files and Excel files via URL
  - Each sheet in an Excel workbook is converted to a separate Markdown table
  - Added intelligent header detection for tables with varying formats

- **2025-05-07:**
  - Added alias management system for frequently used sources
  - Added `--add-alias` and `--alias-from-clipboard` commands
  - Fixed PDF text extraction for local PDF files
  - Changed root XML tag from `<combined_sources>` to `<onefilellm_output>`
  - Added cross-platform launcher scripts for Windows and Linux/macOS
  - Improved user feedback during alias operations

- **2025-05-03:**
  - Added support for processing multiple inputs as command line arguments
  - Implemented functionality to combine multiple outputs into a cohesive XML document
  - Refactored code to improve modularity and reusability
  - Added test files to demonstrate multi-input capabilities

- **2025-01-20:**
  - Added file and directory exclusion functionality to reduce context window usage
  - Added ability to exclude auto-generated files (*.pb.go, etc.)
  - Added ability to exclude mock files and test directories
  - Updated documentation with exclusion configuration instructions
- **2025-01-17:**
  - Added ability to exclude specific directories from processing
  - Updated directory traversal logic to respect exclusion rules
- **2024-07-29:**
  - Updated output format to encapsulate content in XML tags. This change was implemented due to evaluations showing that LLMs perform better with prompts structured in XML.
  - Added tests for GitHub issues and GitHub pull requests to improve robustness and reliability.
  - Updated various processing functions to return formatted content instead of writing directly to files, improving consistency and testability.
- **2024-05-17:** Added ability to pass path or URL as command line argument.
- **2024-05-16:** Updated text colors.
- **2024-05-11:** 
  - Updated requirements.txt.
  - Added Rich library to `onefilellm.py`.
- **2024-04-04:**
  - Added GitHub PR and issue tests.
  - Added GitHub PR and issues.
  - Added tests for GitHub PRs and issues.
  - Added ability to concatenate specific GitHub issue and repo when GitHub issue URL is passed.
  - Updated tests to include pull request changes.
  - Added ability to concatenate pull request and repo when GitHub pull request URL is passed.
- **2024-04-03:**
  - Included the ability to pull a complete GitHub pull request given the GitHub pull request URL.
  - Updated `onefilellm.py` to return an error when Sci-hub is inaccessible or no document is found.


## Notes
- For Repos, Modify this line of code to add or remove filetypes processed: ``` allowed_extensions = ['.py', '.txt', '.js', '.rst', '.sh', '.md', '.pyx', '.html', '.yaml','.json', '.jsonl', '.ipynb', '.h', '.c', '.sql', '.csv'] ```
- For excluding files, modify the excluded_patterns list to customize which files are filtered out
- For excluding directories, modify the EXCLUDED_DIRS list to customize which directories are skipped
- For Web scraping, Modify this line of code to change how many links deep from the starting URL to include ``` max_depth = 2 ```
- Token counts are displayed in the console for both output files.

