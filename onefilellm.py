import asyncio
import time
import requests
from bs4 import BeautifulSoup, Comment
from urllib.parse import urljoin, urlparse
from PyPDF2 import PdfReader
import os
import sys
import json
import tiktoken
import nltk
from nltk.corpus import stopwords
import re
import nbformat
from nbconvert import PythonExporter
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import pyperclip
from pathlib import Path
import wget
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
import xml.etree.ElementTree as ET # Keep for preprocess_text if needed
import pandas as pd
from typing import Union, List, Dict, Optional, Set, Tuple
from dotenv import load_dotenv
from urllib.robotparser import RobotFileParser
import aiohttp
from readability import Document
import io
import argparse

# Import utility functions
from utils import (
    safe_file_read, read_from_clipboard, read_from_stdin,
    detect_text_format, parse_as_plaintext, parse_as_markdown,
    parse_as_json, parse_as_html, parse_as_yaml,
    download_file, is_same_domain, is_within_depth,
    is_excluded_file, is_allowed_filetype, escape_xml
)

# Try to import yaml, but don't fail if not available
try:
    import yaml
except ImportError:
    yaml = None
    print("[Warning] PyYAML module not found. YAML format detection/parsing will be limited.", file=sys.stderr)

# --- Configuration Flags ---
ENABLE_COMPRESSION_AND_NLTK = False # Set to True to enable NLTK download, stopword removal, and compressed output
# --- End Configuration Flags ---

# --- Output Format Notes ---
# This script produces output wrapped in XML-like tags for structure (e.g., <source>, <file>).
# However, the *content* within these tags (especially code) is NOT XML-escaped.
# This means characters like < > & within code blocks are preserved as-is for readability
# and correct interpretation by LLMs. The escape_xml function currently returns text unchanged.
# --- End Output Format Notes ---

# --- Configuration Directories ---
EXCLUDED_DIRS = ["dist", "node_modules", ".git", "__pycache__"]

# --- Alias Configuration ---
ALIAS_DIR_NAME = ".onefilellm_aliases"
ALIAS_DIR = Path.home() / ALIAS_DIR_NAME
# --- End Alias Configuration ---

def ensure_alias_dir_exists():
    """Ensures the alias directory exists, creating it if necessary."""
    ALIAS_DIR.mkdir(parents=True, exist_ok=True)

# --- Placeholders for custom formats ---
def parse_as_doculing(text_content: str) -> str:
    """Placeholder for Doculing parsing. Returns text as is for V1."""
    # TODO: Implement actual Doculing parsing logic when specifications are available.
    return text_content

def parse_as_markitdown(text_content: str) -> str:
    """Placeholder for Markitdown parsing. Returns text as is for V1."""
    # TODO: Implement actual Markitdown parsing logic when specifications are available.
    return text_content

def get_parser_for_format(format_name: str) -> callable:
    """
    Returns the appropriate parser function based on the format name.
    Defaults to parse_as_plaintext if format is unknown.
    """
    parsers = {
        "text": parse_as_plaintext,
        "markdown": parse_as_markdown,
        "json": parse_as_json,
        "html": parse_as_html,
        "yaml": parse_as_yaml,
        "doculing": parse_as_doculing,       # Placeholder
        "markitdown": parse_as_markitdown,   # Placeholder
    }
    return parsers.get(format_name, parse_as_plaintext) # Default to plaintext parser

def process_text_stream(raw_text_content: str, source_info: dict, console: Console, format_override: str | None = None) -> str | None:
    """
    Processes text from a stream (stdin or clipboard).
    Detects format, parses, and builds the XML structure.

    Args:
        raw_text_content (str): The raw text from the input stream.
        source_info (dict): Information about the source, e.g., {'type': 'stdin'}.
        console (Console): The Rich console object for printing messages.
        format_override (str | None): User-specified format, if any.

    Returns:
        str | None: The XML structured output string, or None if processing fails.
    """
    actual_format = ""
    parsed_content = ""

    if format_override:
        actual_format = format_override.lower()
        console.print(f"[green]Processing input as [bold]{actual_format}[/bold] (user override).[/green]")
    else:
        actual_format = detect_text_format(raw_text_content)
        console.print(f"[green]Detected format: [bold]{actual_format}[/bold][/green]")

    parser_function = get_parser_for_format(actual_format)

    try:
        parsed_content = parser_function(raw_text_content)
    except json.JSONDecodeError as e:
        console.print(f"[bold red]Error:[/bold red] Input specified or detected as JSON, but it's not valid JSON. Details: {e}")
        return None
    except yaml.YAMLError as e: # Assuming PyYAML is used
        console.print(f"[bold red]Error:[/bold red] Input specified or detected as YAML, but it's not valid YAML. Details: {e}")
        return None
    except Exception as e: # Catch-all for other parsing errors
        console.print(f"[bold red]Error:[/bold red] Failed to parse content as {actual_format}. Details: {e}")
        return None

    # XML Generation for the stream
    # This XML structure should be consistent with how single files/sources are wrapped.
    # The escape_xml function currently does nothing, which is correct for content.
    # Attributes of XML tags *should* be escaped if they could contain special chars,
    # but 'stdin', 'clipboard', and format names are safe.
    
    source_type_attr = escape_xml(source_info.get('type', 'unknown_stream'))
    format_attr = escape_xml(actual_format)

    # Build the XML for this specific stream source
    # This part creates the XML for THIS stream.
    # It will be wrapped by <onefilellm_output> in main() if it's the only input,
    # or combined with other sources by combine_xml_outputs() if multiple inputs are supported later.
    
    # For now, let's assume process_text_stream provides the content for a single <source> tag
    # and main() will handle the <onefilellm_output> wrapper.
    
    # XML structure should mirror existing <source> tags for files/URLs where possible
    # but with type="stdin" or type="clipboard".
    # Instead of <file path="...">, we might have a <content_block> or similar.

    # Let's create a simple XML structure for the stream content.
    # The content itself (parsed_content) is NOT XML-escaped, preserving its raw form.
    xml_parts = [
        f'<source type="{source_type_attr}" processed_as_format="{format_attr}">',
        f'<content>{escape_xml(parsed_content)}</content>', # escape_xml does nothing to parsed_content
        f'</source>'
    ]
    final_xml_for_stream = "\n".join(xml_parts)
    
    return final_xml_for_stream

stop_words = set()
if ENABLE_COMPRESSION_AND_NLTK:
    try:
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))
    except Exception as e:
        print(f"[bold yellow]Warning:[/bold yellow] Failed to download or load NLTK stopwords. Compression will proceed without stopword removal. Error: {e}")

TOKEN = os.getenv('GITHUB_TOKEN', 'default_token_here')
if TOKEN == 'default_token_here':
    # Consider making this a non-fatal warning or prompt if interactive use is common
    print("[bold red]Warning:[/bold red] GITHUB_TOKEN environment variable not set. GitHub API requests may fail or be rate-limited.")
    # raise EnvironmentError("GITHUB_TOKEN environment variable not set.") # Keep commented out if you want it to proceed

headers = {"Authorization": f"token {TOKEN}"} if TOKEN != 'default_token_here' else {}

def process_ipynb_file(temp_file):
    try:
        with open(temp_file, "r", encoding='utf-8', errors='ignore') as f:
            notebook_content = f.read()
        exporter = PythonExporter()
        python_code, _ = exporter.from_notebook_node(nbformat.reads(notebook_content, as_version=4))
        return python_code
    except Exception as e:
        print(f"[bold red]Error processing notebook {temp_file}: {e}[/bold red]")
        # Return error message instead of raising, wrapped in comments
        return f"# ERROR PROCESSING NOTEBOOK: {e}\n"


# --- XML Handling ---
# --- End XML Handling ---


def process_github_repo(repo_url):
    """
    Processes a GitHub repository, extracting file contents and wrapping them in XML structure.
    """
    api_base_url = "https://api.github.com/repos/"
    repo_url_parts = repo_url.split("https://github.com/")[-1].split("/")
    repo_name = "/".join(repo_url_parts[:2])
    branch_or_tag = ""
    subdirectory = ""

    if len(repo_url_parts) > 2 and repo_url_parts[2] == "tree":
        if len(repo_url_parts) > 3:
            branch_or_tag = repo_url_parts[3]
        if len(repo_url_parts) > 4:
            subdirectory = "/".join(repo_url_parts[4:])
    
    contents_url = f"{api_base_url}{repo_name}/contents"
    if subdirectory:
        contents_url = f"{contents_url}/{subdirectory}"
    if branch_or_tag:
        contents_url = f"{contents_url}?ref={branch_or_tag}"

    # Start XML structure
    repo_content = [f'<source type="github_repository" url="{escape_xml(repo_url)}">']

    def process_directory_recursive(url, repo_content_list):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            files = response.json()

            for file_info in files:
                if file_info["type"] == "dir" and file_info["name"] in EXCLUDED_DIRS:
                    continue

                if file_info["type"] == "file" and is_allowed_filetype(file_info["name"]):
                    print(f"Processing {file_info['path']}...")
                    temp_file = f"temp_{file_info['name']}"
                    try:
                        download_file(file_info["download_url"], temp_file)
                        repo_content_list.append(f'\n<file path="{escape_xml(file_info["path"])}">')
                        if file_info["name"].endswith(".ipynb"):
                            # Append raw code - escape_xml not needed as it does nothing
                            repo_content_list.append(process_ipynb_file(temp_file))
                        else:
                            # Append raw code - escape_xml not needed here
                            repo_content_list.append(safe_file_read(temp_file))
                        repo_content_list.append('</file>')
                    except Exception as e:
                        print(f"[bold red]Error processing file {file_info['path']}: {e}[/bold red]")
                        repo_content_list.append(f'\n<file path="{escape_xml(file_info["path"])}">')
                        repo_content_list.append(f'<error>Failed to download or process: {escape_xml(str(e))}</error>')
                        repo_content_list.append('</file>')
                    finally:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)

                elif file_info["type"] == "dir":
                    process_directory_recursive(file_info["url"], repo_content_list)
        except requests.exceptions.RequestException as e:
            print(f"[bold red]Error fetching directory {url}: {e}[/bold red]")
            repo_content_list.append(f'<error>Failed to fetch directory {escape_xml(url)}: {escape_xml(str(e))}</error>')
        except Exception as e: # Catch other potential errors like JSON parsing
             print(f"[bold red]Error processing directory {url}: {e}[/bold red]")
             repo_content_list.append(f'<error>Failed processing directory {escape_xml(url)}: {escape_xml(str(e))}</error>')


    process_directory_recursive(contents_url, repo_content)
    repo_content.append('\n</source>') # Close source tag
    print("GitHub repository processing finished.")
    return "\n".join(repo_content)

def process_local_folder(local_path):
    """
    Processes a local directory, extracting file contents and wrapping them in XML structure.
    """
    def process_local_directory_recursive(current_path, content_list):
        try:
            for item in os.listdir(current_path):
                item_path = os.path.join(current_path, item)
                relative_path = os.path.relpath(item_path, local_path)

                if os.path.isdir(item_path):
                    if item not in EXCLUDED_DIRS:
                        process_local_directory_recursive(item_path, content_list)
                elif os.path.isfile(item_path):
                    if is_allowed_filetype(item):
                        print(f"Processing {item_path}...")
                        content_list.append(f'\n<file path="{escape_xml(relative_path)}">')
                        try:
                            if item.lower().endswith(".ipynb"): # Case-insensitive check
                                content_list.append(process_ipynb_file(item_path))
                            elif item.lower().endswith(".pdf"): # Case-insensitive check
                                content_list.append(_process_pdf_content_from_path(item_path))
                            elif item.lower().endswith(('.xls', '.xlsx')): # Case-insensitive check for Excel files
                                # Need to pop the opening file tag we already added
                                content_list.pop()  # Remove the <file> tag
                                
                                # Generate Markdown for each sheet
                                try:
                                    for sheet, md in excel_to_markdown(item_path).items():
                                        virtual_name = f"{os.path.splitext(relative_path)[0]}_{sheet}.md"
                                        content_list.append(f'\n<file path="{escape_xml(virtual_name)}">')
                                        content_list.append(md)      # raw Markdown table
                                        content_list.append('</file>')
                                except Exception as e:
                                    print(f"[bold red]Error processing Excel file {item_path}: {e}[/bold red]")
                                    # Re-add the original file tag for the error message
                                    content_list.append(f'\n<file path="{escape_xml(relative_path)}">')
                                    content_list.append(f'<e>Failed to process Excel file: {escape_xml(str(e))}</e>')
                                    content_list.append('</file>')
                                continue  # Skip the final </file> for Excel files
                            else:
                                content_list.append(safe_file_read(item_path))
                        except Exception as e:
                            print(f"[bold red]Error reading file {item_path}: {e}[/bold red]")
                            content_list.append(f'<error>Failed to read file: {escape_xml(str(e))}</error>')
                        content_list.append('</file>')
        except Exception as e:
             print(f"[bold red]Error reading directory {current_path}: {e}[/bold red]")
             content_list.append(f'<error>Failed reading directory {escape_xml(current_path)}: {escape_xml(str(e))}</error>')


    # Start XML structure
    content = [f'<source type="local_folder" path="{escape_xml(local_path)}">']
    process_local_directory_recursive(local_path, content)
    content.append('\n</source>') # Close source tag

    print("Local folder processing finished.")
    return '\n'.join(content)


def _process_pdf_content_from_path(file_path):
    """
    Extracts text content from a local PDF file.
    Returns the extracted text or an error message string.
    """
    print(f"  Extracting text from local PDF: {file_path}")
    text_list = []
    try:
        with open(file_path, 'rb') as pdf_file_obj:
            pdf_reader = PdfReader(pdf_file_obj)
            if not pdf_reader.pages:
                print(f"  [bold yellow]Warning:[/bold yellow] PDF file has no pages or is encrypted: {file_path}")
                return "<e>PDF file has no pages or could not be read (possibly encrypted).</e>"
            
            for i, page_obj in enumerate(pdf_reader.pages):
                try:
                    page_text = page_obj.extract_text()
                    if page_text:
                        text_list.append(page_text)
                except Exception as page_e: # Catch error extracting from a specific page
                     print(f"  [bold yellow]Warning:[/bold yellow] Could not extract text from page {i+1} of {file_path}: {page_e}")
                     text_list.append(f"\n<e>Could not extract text from page {i+1}.</e>\n")
        
        if not text_list:
             print(f"  [bold yellow]Warning:[/bold yellow] No text could be extracted from PDF: {file_path}")
             return "<e>No text could be extracted from PDF.</e>"

        return ' '.join(text_list)
    except Exception as e:
        print(f"[bold red]Error reading PDF file {file_path}: {e}[/bold red]")
        return f"<e>Failed to read or process PDF file: {escape_xml(str(e))}</e>"

def _download_and_read_file(url):
    """
    Downloads and reads the content of a file from a URL.
    Returns the content as text or an error message string.
    """
    print(f"  Downloading and reading content from: {url}")
    try:
        # Add headers conditionally
        response = requests.get(url, headers=headers if TOKEN != 'default_token_here' else None)
        response.raise_for_status()
        
        # Try to determine encoding
        encoding = response.encoding or 'utf-8'
        
        try:
            # Try to decode as text
            content = response.content.decode(encoding)
            return content
        except UnicodeDecodeError:
            # If that fails, try a fallback encoding
            try:
                content = response.content.decode('latin-1')
                return content
            except Exception as decode_err:
                print(f"  [bold yellow]Warning:[/bold yellow] Could not decode content: {decode_err}")
                return f"<e>Failed to decode content: {escape_xml(str(decode_err))}</e>"
                
    except requests.RequestException as e:
        print(f"[bold red]Error downloading file from {url}: {e}[/bold red]")
        return f"<e>Failed to download file: {escape_xml(str(e))}</e>"
    except Exception as e:
        print(f"[bold red]Unexpected error processing file from {url}: {e}[/bold red]")
        return f"<e>Unexpected error: {escape_xml(str(e))}</e>"


def excel_to_markdown(
    file_path: Union[str, Path],
    *,
    skip_rows: int = 0,  # Changed from 3 to 0 to not skip potential headers
    min_header_cells: int = 2,
    sheet_filter: List[str] | None = None,
) -> Dict[str, str]:
    """
    Convert an Excel workbook (.xls / .xlsx) to Markdown.

    Parameters
    ----------
    file_path :
        Path to the workbook.
    skip_rows :
        How many leading rows to ignore before we start hunting for a header row.
        Default is 0 to ensure we don't miss any potential headers.
    min_header_cells :
        Minimum number of non-NA cells that makes a row "look like" a header.
    sheet_filter :
        Optional list of sheet names to include (exact match, case-sensitive).

    Returns
    -------
    Dict[str, str]
        Mapping of ``sheet_name → markdown_table``.
        Empty dict means the workbook had no usable sheets by the above rules.

    Raises
    ------
    ValueError
        If the file extension is not .xls or .xlsx.
    RuntimeError
        If *none* of the sheets meet the header-detection criteria.
    """
    file_path = Path(file_path).expanduser().resolve()
    if file_path.suffix.lower() not in {".xls", ".xlsx"}:
        raise ValueError("Only .xls/.xlsx files are supported")

    print(f"Processing Excel file: {file_path}")
    
    # For simple Excel files, it's often better to use header=0 directly
    # Try both approaches: first with automatic header detection, then fallback to header=0
    try:
        # Let pandas pick the right engine (openpyxl for xlsx, xlrd/pyxlsb if installed for xls)
        wb = pd.read_excel(file_path, sheet_name=None, header=None)

        md_tables: Dict[str, str] = {}

        for name, df in wb.items():
            if sheet_filter and name not in sheet_filter:
                continue

            df = df.iloc[skip_rows:].reset_index(drop=True)
            try:
                # Try to find a header row
                header_idx = next(i for i, row in df.iterrows() if row.count() >= min_header_cells)
                
                # Use ffill instead of deprecated method parameter
                header = df.loc[header_idx].copy()
                header = header.ffill()  # Forward-fill NaN values
                
                body = df.loc[header_idx + 1:].copy()
                body.columns = header
                body.dropna(how="all", inplace=True)
                
                # Convert to markdown
                md_tables[name] = body.to_markdown(index=False)
                print(f"  Processed sheet '{name}' with detected header")
                
            except StopIteration:
                # No row looked like a header - skip for now, we'll try again with header=0
                print(f"  No header detected in sheet '{name}', will try fallback")
                continue

        # If no headers were found with our heuristic, try again with header=0
        if not md_tables:
            print("  No headers detected with heuristic, trying with fixed header row")
            wb = pd.read_excel(file_path, sheet_name=None, header=0)
            
            for name, df in wb.items():
                if sheet_filter and name not in sheet_filter:
                    continue
                    
                # Drop rows that are all NaN
                df = df.dropna(how="all")
                
                # Convert to markdown
                md_tables[name] = df.to_markdown(index=False)
                print(f"  Processed sheet '{name}' with fixed header")

        if not md_tables:
            raise RuntimeError("Workbook contained no sheets with usable data")

        return md_tables
        
    except Exception as e:
        print(f"Error processing Excel file: {e}")
        # Last resort: try with the most basic approach
        wb = pd.read_excel(file_path, sheet_name=None)
        md_tables = {name: df.to_markdown(index=False) for name, df in wb.items() 
                    if not (sheet_filter and name not in sheet_filter)}
                    
        if not md_tables:
            raise RuntimeError(f"Failed to extract any usable data from Excel file: {e}")
            
        return md_tables


def excel_to_markdown_from_url(
    url: str,
    *,
    skip_rows: int = 0,  # Changed from 3 to 0 to not skip potential headers
    min_header_cells: int = 2,
    sheet_filter: List[str] | None = None,
) -> Dict[str, str]:
    """
    Download an Excel workbook from a URL and convert it to Markdown.
    
    This function downloads the Excel file from the URL to a BytesIO buffer
    and then processes it using excel_to_markdown.
    
    Parameters are the same as excel_to_markdown.
    
    Returns
    -------
    Dict[str, str]
        Mapping of ``sheet_name → markdown_table``.
    
    Raises
    ------
    ValueError, RuntimeError, RequestException
        Various errors that might occur during downloading or processing.
    """
    import io
    print(f"  Downloading Excel file from URL: {url}")
    
    try:
        # Add headers conditionally
        response = requests.get(url, headers=headers if TOKEN != 'default_token_here' else None)
        response.raise_for_status()
        
        # Create a BytesIO buffer from the downloaded content
        excel_buffer = io.BytesIO(response.content)
        
        # For simple Excel files, it's often better to use header=0 directly
        # Try both approaches: first with automatic header detection, then fallback to header=0
        try:
            # Let pandas read from the buffer
            wb = pd.read_excel(excel_buffer, sheet_name=None, header=None)
            
            md_tables: Dict[str, str] = {}
            
            for name, df in wb.items():
                if sheet_filter and name not in sheet_filter:
                    continue
                    
                df = df.iloc[skip_rows:].reset_index(drop=True)
                try:
                    # Try to find a header row
                    header_idx = next(i for i, row in df.iterrows() if row.count() >= min_header_cells)
                    
                    # Use ffill instead of deprecated method parameter
                    header = df.loc[header_idx].copy()
                    header = header.ffill()  # Forward-fill NaN values
                    
                    body = df.loc[header_idx + 1:].copy()
                    body.columns = header
                    body.dropna(how="all", inplace=True)
                    
                    # Convert to markdown
                    md_tables[name] = body.to_markdown(index=False)
                    print(f"  Processed sheet '{name}' with detected header")
                    
                except StopIteration:
                    # No row looked like a header - skip for now, we'll try again with header=0
                    print(f"  No header detected in sheet '{name}', will try fallback")
                    continue

            # If no headers were found with our heuristic, try again with header=0
            if not md_tables:
                print("  No headers detected with heuristic, trying with fixed header row")
                excel_buffer.seek(0)  # Reset the buffer position
                wb = pd.read_excel(excel_buffer, sheet_name=None, header=0)
                
                for name, df in wb.items():
                    if sheet_filter and name not in sheet_filter:
                        continue
                        
                    # Drop rows that are all NaN
                    df = df.dropna(how="all")
                    
                    # Convert to markdown
                    md_tables[name] = df.to_markdown(index=False)
                    print(f"  Processed sheet '{name}' with fixed header")

            if not md_tables:
                raise RuntimeError("Workbook contained no sheets with usable data")

            return md_tables
            
        except Exception as e:
            print(f"Error processing Excel file: {e}")
            # Last resort: try with the most basic approach
            excel_buffer.seek(0)  # Reset the buffer position
            wb = pd.read_excel(excel_buffer, sheet_name=None)
            md_tables = {name: df.to_markdown(index=False) for name, df in wb.items() 
                        if not (sheet_filter and name not in sheet_filter)}
                        
            if not md_tables:
                raise RuntimeError(f"Failed to extract any usable data from Excel file: {e}")
                
            return md_tables
        
    except requests.RequestException as e:
        print(f"[bold red]Error downloading Excel file from {url}: {e}[/bold red]")
        raise
    except Exception as e:
        print(f"[bold red]Error processing Excel file from {url}: {e}[/bold red]")
        raise

def process_arxiv_pdf(arxiv_abs_url):
    """
    Downloads and extracts text from an ArXiv PDF, wrapped in XML.
    """
    pdf_url = arxiv_abs_url.replace("/abs/", "/pdf/") + ".pdf"
    temp_pdf_path = 'temp_arxiv.pdf'
    try:
        print(f"Downloading ArXiv PDF from {pdf_url}...")
        response = requests.get(pdf_url)
        response.raise_for_status()

        with open(temp_pdf_path, 'wb') as pdf_file:
            pdf_file.write(response.content)

        print("Extracting text from PDF...")
        text_list = []
        with open(temp_pdf_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            for i, page in enumerate(range(len(pdf_reader.pages))):
                print(f"  Processing page {i+1}/{len(pdf_reader.pages)}")
                page_text = pdf_reader.pages[page].extract_text()
                if page_text: # Add text only if extraction was successful
                    text_list.append(page_text)

        # Use XML structure
        formatted_text = f'<source type="arxiv" url="{escape_xml(arxiv_abs_url)}">\n'
        formatted_text += ' '.join(text_list) # Append raw extracted text
        formatted_text += '\n</source>' # Close source tag
        print("ArXiv paper processed successfully.")
        return formatted_text

    except requests.RequestException as e:
        print(f"[bold red]Error downloading ArXiv PDF {pdf_url}: {e}[/bold red]")
        return f'<source type="arxiv" url="{escape_xml(arxiv_abs_url)}"><error>Failed to download PDF: {escape_xml(str(e))}</error></source>'
    except Exception as e: # Catch PdfReader errors or others
        print(f"[bold red]Error processing ArXiv PDF {arxiv_abs_url}: {e}[/bold red]")
        return f'<source type="arxiv" url="{escape_xml(arxiv_abs_url)}"><error>Failed to process PDF: {escape_xml(str(e))}</error></source>'
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


def fetch_youtube_transcript(url):
    """
    Fetches YouTube transcript using youtube_transcript_api, wrapped in XML.
    """
    def extract_video_id(url):
        pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
        match = re.search(pattern, url)
        return match.group(1) if match else None

    video_id = extract_video_id(url)
    if not video_id:
        print(f"[bold red]Could not extract YouTube video ID from URL: {url}[/bold red]")
        # Use XML for errors
        return f'<source type="youtube_transcript" url="{escape_xml(url)}">\n<error>Could not extract video ID from URL.</error>\n</source>'

    try:
        print(f"Fetching transcript for YouTube video ID: {video_id}")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = TextFormatter()
        transcript = formatter.format_transcript(transcript_list)
        print("Transcript fetched successfully.")

        # Use XML structure for success
        formatted_text = f'<source type="youtube_transcript" url="{escape_xml(url)}">\n'
        formatted_text += transcript # Append raw transcript text
        formatted_text += '\n</source>' # Close source tag
        return formatted_text
    except Exception as e:
        print(f"[bold red]Error fetching YouTube transcript for {url}: {e}[/bold red]")
        # Use XML structure for errors
        return f'<source type="youtube_transcript" url="{escape_xml(url)}">\n<error>{escape_xml(str(e))}</error>\n</source>'

def preprocess_text(input_file, output_file):
    """
    Preprocesses text, optionally removing stopwords if NLTK is enabled.
    Handles potential XML structure if present (intended for compressed output).
    """
    print("Preprocessing text for compression (if enabled)...")
    with open(input_file, "r", encoding="utf-8") as infile:
        input_text = infile.read()

    def process_content(text):
        text = re.sub(r"[\n\r]+", "\n", text)
        text = re.sub(r"[^a-zA-Z0-9\s_.,!?:;@#$%^&*()+\-=[\]{}|\\<>`~'\"/]+", "", text)
        text = re.sub(r"\s+", " ", text)
        text = text.lower()
        if ENABLE_COMPRESSION_AND_NLTK and stop_words:
            words = text.split()
            words = [word for word in words if word not in stop_words]
            text = " ".join(words)
        return text

    try:
        # Attempt to parse as XML - this is mainly relevant if the INPUT
        # already had some structure we wanted to preserve during compression
        root = ET.fromstring(input_text)
        for elem in root.iter():
            if elem.text:
                elem.text = process_content(elem.text)
            if elem.tail:
                elem.tail = process_content(elem.tail)
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        print("Text preprocessing with XML structure preservation completed.")
    except ET.ParseError:
        # If input is not valid XML (likely our case with raw content), process as plain text
        processed_text = process_content(input_text)
        with open(output_file, "w", encoding="utf-8") as out_file:
            out_file.write(processed_text)
        print("Input was not XML. Text preprocessing completed as plain text.")
    except Exception as e:
        print(f"[bold red]Error during text preprocessing: {e}[/bold red]")
        # Fallback: write the original text if preprocessing fails
        with open(output_file, "w", encoding="utf-8") as out_file:
             out_file.write(input_text)
        print("[bold yellow]Warning:[/bold yellow] Preprocessing failed, writing original content to compressed file.")


def is_potential_alias(arg_string):
    """Checks if an argument string looks like a potential alias name."""
    if not arg_string:
        return False
    # An alias should not contain typical path or URL characters
    return not any(char in arg_string for char in ['.', '/', ':', '\\'])

def handle_add_alias(args, console):
    """Handles the --add-alias command."""
    ensure_alias_dir_exists()
    if len(args) < 2: # Need at least '--add-alias', 'alias_name', 'target'
        console.print("[bold red]Error:[/bold red] --add-alias requires an alias name and at least one target URL/path.")
        console.print("Usage: python onefilellm.py --add-alias <alias_name> <url_or_path1> [url_or_path2 ...]")
        return True # Indicate handled and should exit

    alias_name_index = args.index("--add-alias") + 1
    if alias_name_index >= len(args):
         console.print("[bold red]Error:[/bold red] Alias name not provided after --add-alias.")
         return True

    alias_name = args[alias_name_index]
    
    # Basic validation for alias name (no path-like chars)
    if '/' in alias_name or '\\' in alias_name or '.' in alias_name or ':' in alias_name:
        console.print(f"[bold red]Error:[/bold red] Invalid alias name '{alias_name}'. Avoid using '/', '\\', '.', or ':'.")
        return True

    targets = args[alias_name_index + 1:]

    if not targets:
        console.print("[bold red]Error:[/bold red] No target URLs/paths provided for the alias.")
        return True

    alias_file_path = ALIAS_DIR / alias_name
    try:
        with open(alias_file_path, "w", encoding="utf-8") as f:
            for target in targets:
                f.write(target + "\n")
        console.print(f"[green]Alias '{alias_name}' created/updated successfully.[/green]")
        for i, target in enumerate(targets):
            console.print(f"  {i+1}. {target}")
    except IOError as e:
        console.print(f"[bold red]Error creating alias file {alias_file_path}: {e}[/bold red]")
    return True # Indicate handled and should exit

def handle_alias_from_clipboard(args, console):
    """Handles the --alias-from-clipboard command."""
    ensure_alias_dir_exists()
    
    if len(args) < 2: # Need at least '--alias-from-clipboard', 'alias_name'
        console.print("[bold red]Error:[/bold red] --alias-from-clipboard requires an alias name.")
        console.print("Usage: python onefilellm.py --alias-from-clipboard <alias_name>")
        return True

    alias_name_index = args.index("--alias-from-clipboard") + 1
    if alias_name_index >= len(args):
         console.print("[bold red]Error:[/bold red] Alias name not provided after --alias-from-clipboard.")
         return True

    alias_name = args[alias_name_index]

    if '/' in alias_name or '\\' in alias_name or '.' in alias_name or ':' in alias_name:
        console.print(f"[bold red]Error:[/bold red] Invalid alias name '{alias_name}'. Avoid using '/', '\\', '.', or ':'.")
        return True

    try:
        clipboard_content = pyperclip.paste()
        if not clipboard_content or not clipboard_content.strip():
            console.print("[bold yellow]Warning:[/bold yellow] Clipboard is empty. Alias not created.")
            return True
        
        # Treat each line in clipboard as a separate target
        targets = [line.strip() for line in clipboard_content.splitlines() if line.strip()]

        if not targets:
            console.print("[bold yellow]Warning:[/bold yellow] Clipboard content did not yield any valid targets (after stripping whitespace). Alias not created.")
            return True

        alias_file_path = ALIAS_DIR / alias_name
        with open(alias_file_path, "w", encoding="utf-8") as f:
            for target in targets:
                f.write(target + "\n")
        console.print(f"[green]Alias '{alias_name}' created/updated successfully from clipboard content:[/green]")
        for i, target in enumerate(targets):
            console.print(f"  {i+1}. {target}")
    except pyperclip.PyperclipException as e:
        console.print(f"[bold red]Error accessing clipboard: {e}[/bold red]")
        console.print("Please ensure you have a copy/paste mechanism installed (e.g., xclip or xsel on Linux).")
    except IOError as e:
        console.print(f"[bold red]Error creating alias file {alias_file_path}: {e}[/bold red]")
    return True # Indicate handled and should exit

def load_alias(alias_name, console):
    """Loads target paths from an alias file."""
    ensure_alias_dir_exists() # Ensure directory is checked, though mostly for creation
    alias_file_path = ALIAS_DIR / alias_name
    if alias_file_path.is_file():
        try:
            with open(alias_file_path, "r", encoding="utf-8") as f:
                targets = [line.strip() for line in f if line.strip()]
            if not targets:
                console.print(f"[yellow]Warning: Alias '{alias_name}' file is empty.[/yellow]")
                return []
            return targets
        except IOError as e:
            console.print(f"[bold red]Error reading alias file {alias_file_path}: {e}[/bold red]")
            return None # Indicate error
    return None # Alias not found

def resolve_single_input_source(source_string, console):
    """
    Resolves a single input string. If it's an alias, expands it.
    Otherwise, returns the string as a single-item list.
    Returns a list of actual source strings, or an empty list if an alias is empty or fails to load.
    """
    resolved_sources = []
    if is_potential_alias(source_string):
        # The following lines are more aligned with existing logging in main() for CLI args
        # so we adapt the logging slightly to match.
        existing_log_message = f"[dim]Checking if '{source_string}' is an alias...[/dim]"
        if console: # Check if console object is passed (it might not be in all call contexts)
            console.print(existing_log_message)
        else:
            print(existing_log_message) # Fallback to standard print

        resolved_targets = load_alias(source_string, console)
        if resolved_targets is not None:  # Alias found and loaded (could be empty list)
            resolved_sources.extend(resolved_targets)
            if resolved_targets:
                success_message = f"[cyan]Alias '{source_string}' expanded to: {', '.join(resolved_targets)}[/cyan]"
                if console:
                    console.print(success_message)
                else:
                    print(success_message)
            else:
                empty_alias_message = f"[yellow]Alias '{source_string}' is defined but empty. Skipping.[/yellow]"
                if console:
                    console.print(empty_alias_message)
                else:
                    print(empty_alias_message)
        else:  # Not found as an alias or error reading it
            not_found_message = f"[dim]'{source_string}' is not a known alias or could not be read. Treating as a direct input.[/dim]"
            if console:
                console.print(not_found_message)
            else:
                print(not_found_message)
            resolved_sources.append(source_string)
    else:
        resolved_sources.append(source_string)
    return resolved_sources

def get_token_count(text, disallowed_special=[], chunk_size=1000):
    """
    Counts tokens using tiktoken, stripping XML tags first.
    """
    enc = tiktoken.get_encoding("cl100k_base")

    # Restore XML tag removal before counting tokens
    # This gives a count of the actual content, not the structural tags
    text_without_tags = re.sub(r'<[^>]+>', '', text)

    # Split the text without tags into smaller chunks for more robust encoding
    chunks = [text_without_tags[i:i+chunk_size] for i in range(0, len(text_without_tags), chunk_size)]
    total_tokens = 0

    for chunk in chunks:
        try:
            tokens = enc.encode(chunk, disallowed_special=disallowed_special)
            total_tokens += len(tokens)
        except Exception as e:
            print(f"[bold yellow]Warning:[/bold yellow] Error encoding chunk for token count: {e}")
            # Estimate token count for problematic chunk (e.g., len/4)
            total_tokens += len(chunk) // 4

    return total_tokens


def process_web_pdf(url):
    """Downloads and extracts text from a PDF found during web crawl."""
    temp_pdf_path = 'temp_web.pdf'
    try:
        print(f"  Downloading PDF: {url}")
        response = requests.get(url, timeout=30) # Add timeout
        response.raise_for_status()

        # Basic check for PDF content type
        if 'application/pdf' not in response.headers.get('Content-Type', '').lower():
             print(f"  [bold yellow]Warning:[/bold yellow] URL doesn't report as PDF, skipping: {url}")
             return None # Or return an error string

        with open(temp_pdf_path, 'wb') as pdf_file:
            pdf_file.write(response.content)

        print(f"  Extracting text from PDF: {url}")
        text_list = []
        with open(temp_pdf_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            for page in range(len(pdf_reader.pages)):
                page_text = pdf_reader.pages[page].extract_text()
                if page_text:
                    text_list.append(page_text)
        return ' '.join(text_list)
    except requests.RequestException as e:
        print(f"  [bold red]Error downloading PDF {url}: {e}[/bold red]")
        return f"<error>Failed to download PDF: {escape_xml(str(e))}</error>"
    except Exception as e:
        print(f"  [bold red]Error processing PDF {url}: {e}[/bold red]")
        return f"<error>Failed to process PDF: {escape_xml(str(e))}</error>"
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


def crawl_and_extract_text(base_url, max_depth, include_pdfs, ignore_epubs):
    """
    Crawls a website starting from base_url, extracts text, and wraps in XML.
    """
    visited_urls = set()
    urls_to_visit = [(base_url, 0)]
    processed_urls_content = {} # Store URL -> content/error
    # Start XML structure
    all_text = [f'<source type="web_crawl" base_url="{escape_xml(base_url)}">']

    print(f"Starting crawl from: {base_url} (Max Depth: {max_depth}, Include PDFs: {include_pdfs})")

    while urls_to_visit:
        current_url, current_depth = urls_to_visit.pop(0)
        # Normalize URL: remove fragment and ensure scheme
        parsed_url = urlparse(current_url)
        clean_url = urlparse(current_url)._replace(fragment="").geturl()
        if not parsed_url.scheme:
             clean_url = "http://" + clean_url # Default to http if missing

        if clean_url in visited_urls:
            continue

        # Check domain and depth *after* cleaning URL
        if not is_same_domain(base_url, clean_url) or not is_within_depth(base_url, clean_url, max_depth):
             # print(f"Skipping (domain/depth): {clean_url}") # Optional debug
             continue

        if ignore_epubs and clean_url.lower().endswith('.epub'):
            print(f"Skipping (EPUB): {clean_url}")
            visited_urls.add(clean_url)
            continue

        print(f"Processing (Depth {current_depth}): {clean_url}")
        visited_urls.add(clean_url)
        page_content = f'\n<page url="{escape_xml(clean_url)}">' # Start page tag

        try:
            # Handle PDFs separately
            if clean_url.lower().endswith('.pdf'):
                if include_pdfs:
                    pdf_text = process_web_pdf(clean_url)
                    if pdf_text: # Append text or error message from process_web_pdf
                        page_content += f'\n{pdf_text}\n'
                    else: # process_web_pdf returned None (e.g., wrong content type)
                        page_content += '\n<error>Skipped non-PDF content reported at PDF URL.</error>\n'
                else:
                    print(f"  Skipping PDF (include_pdfs=False): {clean_url}")
                    page_content += '\n<skipped>PDF ignored by configuration</skipped>\n'

            # Handle HTML pages
            else:
                 # Add timeout to requests
                response = requests.get(clean_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
                response.raise_for_status()

                # Basic check for HTML content type
                if 'text/html' not in response.headers.get('Content-Type', '').lower():
                    print(f"  [bold yellow]Warning:[/bold yellow] Skipping non-HTML page: {clean_url} (Content-Type: {response.headers.get('Content-Type')})")
                    page_content += f'\n<skipped>Non-HTML content type: {escape_xml(response.headers.get("Content-Type", "N/A"))}</skipped>\n'
                else:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    # Remove scripts, styles, etc.
                    for element in soup(['script', 'style', 'head', 'title', 'meta', '[document]', 'nav', 'footer', 'aside']): # Added common noise tags
                        element.decompose()
                    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
                    for comment in comments:
                        comment.extract()
                    # Get text, try to preserve some structure with newlines
                    text = soup.get_text(separator='\n', strip=True)
                    page_content += f'\n{text}\n' # Append raw extracted text

                    # Find links for the next level if depth allows
                    if current_depth < max_depth:
                        for link in soup.find_all('a', href=True):
                            try:
                                new_url_raw = link['href']
                                if new_url_raw and not new_url_raw.startswith(('mailto:', 'javascript:', '#')):
                                    new_url = urljoin(clean_url, new_url_raw)
                                    parsed_new = urlparse(new_url)
                                    # Add scheme if missing for domain/depth checks
                                    if not parsed_new.scheme:
                                         new_url = parsed_new._replace(scheme=urlparse(clean_url).scheme).geturl()

                                    new_clean_url = urlparse(new_url)._replace(fragment="").geturl()

                                    if new_clean_url not in visited_urls:
                                        # Check domain/depth *before* adding to queue
                                        if is_same_domain(base_url, new_clean_url) and is_within_depth(base_url, new_clean_url, max_depth):
                                             if not (ignore_epubs and new_clean_url.lower().endswith('.epub')):
                                                # Add only if valid and not already visited
                                                if (new_clean_url, current_depth + 1) not in urls_to_visit:
                                                     urls_to_visit.append((new_clean_url, current_depth + 1))
                            except Exception as link_err: # Catch errors parsing individual links
                                print(f"  [bold yellow]Warning:[/bold yellow] Error parsing link '{link.get('href')}': {link_err}")


        except requests.exceptions.Timeout:
             print(f"[bold red]Timeout retrieving {clean_url}[/bold red]")
             page_content += f'\n<error>Timeout during request</error>\n'
        except requests.RequestException as e:
            print(f"[bold red]Failed to retrieve {clean_url}: {e}[/bold red]")
            page_content += f'\n<error>Failed to retrieve URL: {escape_xml(str(e))}</error>\n'
        except Exception as e: # Catch other errors like BeautifulSoup issues
             print(f"[bold red]Error processing page {clean_url}: {e}[/bold red]")
             page_content += f'\n<error>Error processing page: {escape_xml(str(e))}</error>\n'

        page_content += '</page>' # Close page tag
        all_text.append(page_content)
        processed_urls_content[clean_url] = page_content # Store for processed list


    all_text.append('\n</source>') # Close source tag
    print("Web crawl finished.")
    formatted_content = '\n'.join(all_text)

    return {
        'content': formatted_content,
        'processed_urls': list(processed_urls_content.keys()) # Return URLs we attempted to process
    }


# --- Helper functions for DocCrawler ---
def _detect_code_language_heuristic(code: str) -> str:
    """Attempt to detect programming language of code block with naive heuristics."""
    if re.search(r'^\s*(import|from)\s+\w+\s+import|def\s+\w+\s*\(|class\s+\w+[:\(]', code, re.MULTILINE):
        return "python"
    elif re.search(r'^\s*(function|const|let|var|import)\s+|=\>|{\s*\n|export\s+', code, re.MULTILINE):
        return "javascript"
    elif re.search(r'^\s*(#include|int\s+main|using\s+namespace)', code, re.MULTILINE):
        return "cpp"
    elif re.search(r'^\s*(public\s+class|import\s+java|@Override)', code, re.MULTILINE):
        return "java"
    elif re.search(r'<\?php|\$\w+\s*=', code, re.MULTILINE):
        return "php"
    elif re.search(r'^\s*(use\s+|fn\s+\w+|let\s+mut|impl)', code, re.MULTILINE):
        return "rust"
    elif re.search(r'^\s*(package\s+main|import\s+\(|func\s+\w+\s*\()', code, re.MULTILINE):
        return "go"
    elif re.search(r'<html|<body|<div|<script|<style', code, re.IGNORECASE | re.MULTILINE):
        return "html"
    elif re.search(r'^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE TABLE)', code, re.IGNORECASE | re.MULTILINE):
        return "sql"
    return "code"  # Default if no strong signal


def _clean_text_content(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]', ' ', text)  # Various space chars
    text = re.sub(r'[\u2018\u2019]', "'", text)  # Smart quotes to standard
    text = re.sub(r'[\u201C\u201D]', '"', text)  # Smart quotes to standard
    return text


class DocCrawler:
    """Advanced web crawler for extracting structured content from websites."""
    
    def __init__(self, start_url: str, cli_args: object, console_obj):
        self.start_url = start_url
        self.config = cli_args  # This will be the argparse namespace or similar
        self.console = console_obj
        
        self.output_xml_parts: List[str] = []
        self.visited_urls: Set[str] = set()
        self.pages_crawled = 0
        self.failed_urls: List[Tuple[str, str]] = []
        
        parsed_start = urlparse(self.start_url)
        self.domain = parsed_start.netloc
        self.start_url_path_prefix = parsed_start.path.rstrip('/') or "/"  # For --crawl-restrict-path
        
        self.robots_parsers: Dict[str, RobotFileParser] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.rich_progress = None  # For Rich progress bar
        self.progress_task_id = None
        
        # Map CLI args to attributes for convenience
        # Using defaults from change.md since CLI args aren't implemented yet
        self.max_depth = getattr(self.config, 'crawl_max_depth', 3)
        self.max_pages = getattr(self.config, 'crawl_max_pages', 1000)  # Increased default from 100 to 1000
        self.user_agent = getattr(self.config, 'crawl_user_agent', "OneFileLLMCrawler/1.1")
        self.delay = getattr(self.config, 'crawl_delay', 0.25)
        self.include_pattern = re.compile(self.config.crawl_include_pattern) if getattr(self.config, 'crawl_include_pattern', None) else None
        self.exclude_pattern = re.compile(self.config.crawl_exclude_pattern) if getattr(self.config, 'crawl_exclude_pattern', None) else None
        self.timeout = getattr(self.config, 'crawl_timeout', 20)
        self.include_images = getattr(self.config, 'crawl_include_images', False)
        self.include_code = getattr(self.config, 'crawl_include_code', True)
        self.extract_headings = getattr(self.config, 'crawl_extract_headings', True)
        self.follow_links = getattr(self.config, 'crawl_follow_links', False)
        self.clean_html = getattr(self.config, 'crawl_clean_html', True)
        self.strip_js = getattr(self.config, 'crawl_strip_js', True)
        self.strip_css = getattr(self.config, 'crawl_strip_css', True)
        self.strip_comments = getattr(self.config, 'crawl_strip_comments', True)
        self.respect_robots = getattr(self.config, 'crawl_respect_robots', False)  # Changed to False for backward compatibility
        self.concurrency = getattr(self.config, 'crawl_concurrency', 3)
        self.restrict_path = getattr(self.config, 'crawl_restrict_path', False)
        self.include_pdfs = getattr(self.config, 'crawl_include_pdfs', True)
        self.ignore_epubs = getattr(self.config, 'crawl_ignore_epubs', True)

    async def _init_session(self):
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }
        self.session = aiohttp.ClientSession(headers=headers)

    async def _close_session(self):
        if self.session:
            await self.session.close()

    async def _can_fetch_robots(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        if domain not in self.robots_parsers:
            robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"
            parser = RobotFileParser()
            parser.set_url(robots_url)
            try:
                # RobotFileParser.read() is synchronous. Run in executor.
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, parser.read)
                self.robots_parsers[domain] = parser
            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not fetch/parse robots.txt for {domain}: {e}[/yellow]")
                return True  # Default to allow if robots.txt is inaccessible
        
        return self.robots_parsers[domain].can_fetch(self.user_agent, url)

    def _should_crawl_url(self, url: str) -> bool:
        parsed_url = urlparse(url)
        
        if parsed_url.scheme not in ('http', 'https'):
            return False
        
        # Handle external links based on --crawl-follow-links
        if not self.follow_links and parsed_url.netloc != self.domain:
            return False

        if self.restrict_path:
            # Ensure current URL's path starts with the initial URL's path prefix
            current_path_normalized = parsed_url.path.rstrip('/') or "/"
            if not current_path_normalized.startswith(self.start_url_path_prefix):
                return False

        if url in self.visited_urls:
            return False
        
        if self.pages_crawled >= self.max_pages:
            return False
        
        if self.include_pattern and not self.include_pattern.search(url):
            return False
        
        if self.exclude_pattern and self.exclude_pattern.search(url):
            return False

        if self.ignore_epubs and url.lower().endswith('.epub'):
            self.console.print(f"  [dim]Skipping EPUB: {url}[/dim]")
            return False
            
        # Note: robots.txt check is async, so it's done in the worker.
        return True

    async def _fetch_url_content(self, url: str) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
        if not self.session:
            await self._init_session()
        
        try:
            async with self.session.get(url, timeout=self.timeout) as response:
                content_type_header = response.headers.get('Content-Type', '')
                if response.status != 200:
                    return None, f"HTTP Error {response.status}: {response.reason}", content_type_header
                
                # Read content as bytes first to handle different types
                content_bytes = await response.read()
                return content_bytes, None, content_type_header
                
        except asyncio.TimeoutError:
            return None, "Request timed out", None
        except aiohttp.ClientError as e:
            return None, f"Client error: {e}", None
        except Exception as e:
            return None, f"Unexpected fetch error: {e}", None

    def _extract_page_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        links = []
        for link_tag in soup.find_all('a', href=True):
            href = link_tag['href'].strip()
            if not href or href.startswith(('#', 'javascript:', 'mailto:')):
                continue
            
            full_url = urljoin(base_url, href)
            # Normalize: remove fragment, ensure scheme for external links
            parsed_new_url = urlparse(full_url)
            normalized_url = parsed_new_url._replace(fragment="").geturl()
            
            links.append(normalized_url)
        return links

    def _process_html_to_structured_data(self, html_content: str, url: str) -> Dict:
        try:
            doc = Document(html_content)
            title = _clean_text_content(doc.title())
            
            if self.clean_html:
                # Use readability's cleaned HTML body
                main_content_html = doc.summary()
                soup = BeautifulSoup(main_content_html, 'lxml') 
            else:
                soup = BeautifulSoup(html_content, 'lxml')

            if self.strip_js:
                for script_tag in soup.find_all('script'):
                    script_tag.decompose()
            if self.strip_css:
                for style_tag in soup.find_all('style'):
                    style_tag.decompose()
            if self.strip_comments:
                for comment_tag in soup.find_all(string=lambda text_node: isinstance(text_node, Comment)):
                    comment_tag.extract()
            
            meta_tags = {}
            for meta in soup.find_all('meta'):
                name = meta.get('name') or meta.get('property')
                content = meta.get('content')
                if name and content:
                    meta_tags[_clean_text_content(name)] = _clean_text_content(content)
            
            structured_content_blocks = self._extract_structured_content_from_soup(soup, url)
            
            return {
                'url': url,
                'title': title,
                'meta': meta_tags,
                'content_blocks': structured_content_blocks
            }
        except Exception as e:
            self.console.print(f"[bold red]Error processing HTML for {url}: {e}[/bold red]")
            return {
                'url': url,
                'title': f"Error processing page: {url}",
                'meta': {},
                'content_blocks': [{'type': 'error', 'text': f"Failed to process HTML: {e}"}]
            }

    def _extract_structured_content_from_soup(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        content_blocks = []
        
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'pre', 'table']):
            if element.name.startswith('h'):
                level = int(element.name[1])
                text = _clean_text_content(element.get_text())
                if text:
                    content_blocks.append({'type': 'heading', 'level': level, 'text': text})
            elif element.name == 'p':
                text = _clean_text_content(element.get_text())
                if text:
                    content_blocks.append({'type': 'paragraph', 'text': text})
            elif element.name in ('ul', 'ol'):
                items = [_clean_text_content(li.get_text()) for li in element.find_all('li', recursive=False) if _clean_text_content(li.get_text())]
                if items:
                    content_blocks.append({'type': 'list', 'list_type': element.name, 'items': items})
            elif element.name == 'pre':  # Often contains code
                code_text = element.get_text()  # Keep original spacing
                if self.include_code and code_text.strip():
                    # Attempt to find language from class attribute if present
                    lang_class = element.get('class', [])
                    lang = "code"  # default
                    for cls in lang_class:
                        if cls.startswith('language-'):
                            lang = cls.replace('language-', '')
                            break
                    if lang == "code":  # if not found in class, use heuristic
                        lang = _detect_code_language_heuristic(code_text)
                    content_blocks.append({'type': 'code', 'language': lang, 'code': code_text})
            elif element.name == 'table':
                headers = []
                rows_data = []
                # Extract headers (th)
                for th in element.select('thead tr th, table > tr:first-child > th'):
                    headers.append(_clean_text_content(th.get_text()))
                # Extract rows (tr) and cells (td)
                for row_element in element.select('tbody tr, table > tr'):
                    # Avoid re-processing header row if it was caught by th selector
                    if row_element.find('th') and headers:
                        if all(_clean_text_content(th.get_text()) in headers for th in row_element.find_all('th')):
                            continue 
                    
                    cells = [_clean_text_content(td.get_text()) for td in row_element.find_all(['td', 'th'])]
                    if cells:
                        rows_data.append(cells)
                if not headers and rows_data:  # If no <th>, use first row as header
                    headers = rows_data.pop(0)

                if rows_data:  # Only add table if it has data rows
                    content_blocks.append({'type': 'table', 'headers': headers, 'rows': rows_data})

        if self.include_images:
            for img_tag in soup.find_all('img'):
                src = img_tag.get('src')
                alt = _clean_text_content(img_tag.get('alt', ''))
                if src:
                    img_url = urljoin(base_url, src)
                    content_blocks.append({'type': 'image', 'url': img_url, 'alt_text': alt})
        return content_blocks

    def _initialize_xml_output(self):
        self.output_xml_parts = [f'<source type="web_crawl" base_url="{escape_xml(self.start_url)}">']

    def _add_page_to_xml_output(self, page_data: Dict):
        page_xml_parts = [f'<page url="{escape_xml(page_data["url"])}">']
        page_xml_parts.append(f'<title>{escape_xml(page_data.get("title", "N/A"))}</title>')

        if page_data.get('meta'):
            meta_xml_parts = ['<meta>']
            for key, value in page_data['meta'].items():
                meta_xml_parts.append(f'<meta_item name="{escape_xml(key)}">{escape_xml(str(value))}</meta_item>')
            meta_xml_parts.append('</meta>')
            page_xml_parts.append("".join(meta_xml_parts))

        content_xml_parts = ['<content>']
        for block in page_data.get('content_blocks', []):
            block_type = block.get('type')
            if block_type == 'paragraph':
                content_xml_parts.append(f'<paragraph>{escape_xml(block.get("text", ""))}</paragraph>')
            elif block_type == 'heading':
                content_xml_parts.append(f'<heading level="{block.get("level", 0)}">{escape_xml(block.get("text", ""))}</heading>')
            elif block_type == 'list':
                list_items_xml = "".join([f'<item>{escape_xml(item)}</item>' for item in block.get("items", [])])
                content_xml_parts.append(f'<list type="{block.get("list_type", "ul")}">{list_items_xml}</list>')
            elif block_type == 'code':
                # For code, do not escape_xml the content to preserve syntax
                content_xml_parts.append(f'<code language="{escape_xml(block.get("language", "unknown"))}">{block.get("code", "")}</code>')
            elif block_type == 'image':
                content_xml_parts.append(f'<image src="{escape_xml(block.get("url", ""))}" alt_text="{escape_xml(block.get("alt_text", ""))}" />')
            elif block_type == 'table':
                table_parts = ['<table>']
                if block.get('headers'):
                    header_row = "".join([f'<th>{escape_xml(h)}</th>' for h in block['headers']])
                    table_parts.append(f'<thead><tr>{header_row}</tr></thead>')
                
                body_rows = []
                for row_data in block.get('rows', []):
                    cell_row = "".join([f'<td>{escape_xml(cell)}</td>' for cell in row_data])
                    body_rows.append(f'<tr>{cell_row}</tr>')
                if body_rows:
                    table_parts.append(f'<tbody>{"".join(body_rows)}</tbody>')
                table_parts.append('</table>')
                content_xml_parts.append("".join(table_parts))
            elif block_type == 'error':
                content_xml_parts.append(f'<error_in_page>{escape_xml(block.get("text", "Unknown page error"))}</error_in_page>')

        content_xml_parts.append('</content>')
        page_xml_parts.append("".join(content_xml_parts))
        page_xml_parts.append('</page>')
        self.output_xml_parts.append("\n".join(page_xml_parts))
    
    async def _process_pdf_content_from_bytes(self, pdf_bytes: bytes, url: str) -> Optional[Dict]:
        self.console.print(f"  [cyan]Extracting text from PDF:[/cyan] {url}")
        try:
            pdf_file_obj = io.BytesIO(pdf_bytes)
            pdf_reader = PdfReader(pdf_file_obj)
            if not pdf_reader.pages:
                self.console.print(f"  [yellow]Warning: PDF has no pages or is encrypted: {url}[/yellow]")
                return None
            
            text_list = []
            for i, page_obj in enumerate(pdf_reader.pages):
                try:
                    page_text = page_obj.extract_text()
                    if page_text:
                        text_list.append(page_text)
                except Exception as page_e:
                    self.console.print(f"  [yellow]Warning: Could not extract text from page {i+1} of {url}: {page_e}[/yellow]")
            
            if not text_list:
                self.console.print(f"  [yellow]Warning: No text extracted from PDF: {url}[/yellow]")
                return None
            
            full_text = "\n\n--- Page Break ---\n\n".join(text_list)
            return {
                'url': url,
                'title': f"PDF: {os.path.basename(urlparse(url).path)}",
                'meta': {},
                'content_blocks': [{'type': 'paragraph', 'text': full_text}]
            }
        except Exception as e:
            self.console.print(f"[bold red]Error reading PDF content for {url}: {e}[/bold red]")
            return {
                'url': url,
                'title': f"Error processing PDF: {url}",
                'meta': {},
                'content_blocks': [{'type': 'error', 'text': f"Failed to process PDF content: {e}"}]
            }

    async def _worker(self, queue: asyncio.Queue):
        while True:
            try:
                url, depth = await queue.get()

                if self.pages_crawled >= self.max_pages:
                    queue.task_done()
                    continue
                
                # Perform async robots.txt check here
                if not await self._can_fetch_robots(url):
                    self.console.print(f"  [dim]Skipping (robots.txt): {url}[/dim]")
                    self.visited_urls.add(url)
                    queue.task_done()
                    continue

                # _should_crawl_url is synchronous and checks other conditions
                if not self._should_crawl_url(url):
                    self.visited_urls.add(url)
                    queue.task_done()
                    continue

                self.visited_urls.add(url)
                await asyncio.sleep(self.delay)

                self.console.print(f"[cyan]Crawling (Depth {depth}):[/cyan] {url}")
                
                content_bytes, error_msg, content_type_header = await self._fetch_url_content(url)

                page_data_dict = None
                if error_msg:
                    self.console.print(f"  [yellow]Failed to fetch {url}: {error_msg}[/yellow]")
                    self.failed_urls.append((url, error_msg))
                elif content_bytes and content_type_header:
                    if 'application/pdf' in content_type_header.lower() and self.include_pdfs:
                        page_data_dict = await self._process_pdf_content_from_bytes(content_bytes, url)
                    elif 'text/html' in content_type_header.lower():
                        try:
                            # Attempt to decode HTML content
                            html_text_content = content_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            try:
                                html_text_content = content_bytes.decode('latin-1')
                            except UnicodeDecodeError as ude:
                                self.console.print(f"  [yellow]Failed to decode HTML for {url}: {ude}[/yellow]")
                                self.failed_urls.append((url, f"Unicode decode error: {ude}"))
                                html_text_content = None
                        if html_text_content:
                            page_data_dict = self._process_html_to_structured_data(html_text_content, url)
                    else:
                        self.console.print(f"  [dim]Skipping non-HTML/PDF content ({content_type_header}): {url}[/dim]")
                
                if page_data_dict:
                    self._add_page_to_xml_output(page_data_dict)
                    self.pages_crawled += 1
                    if self.rich_progress and self.progress_task_id is not None:
                        self.rich_progress.update(self.progress_task_id, advance=1, description=f"Crawled {self.pages_crawled}/{self.max_pages} pages")

                # Add new links to queue if depth and page count allow
                if page_data_dict and depth < self.max_depth and 'text/html' in (content_type_header or ""):
                    if 'html_text_content' in locals() and html_text_content:
                        soup_for_links = BeautifulSoup(html_text_content, 'lxml')
                        new_links = self._extract_page_links(soup_for_links, url)
                        for link_to_add in new_links:
                            if self._should_crawl_url(link_to_add) and self.pages_crawled < self.max_pages:
                                await queue.put((link_to_add, depth + 1))
                queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                current_url_in_worker = url if 'url' in locals() else "unknown"
                self.console.print(f"[bold red]Unexpected error in worker for URL {current_url_in_worker}: {type(e).__name__} - {e}[/bold red]")
                import traceback
                self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
                if 'queue' in locals() and hasattr(queue, 'task_done'):
                     queue.task_done()

    async def crawl(self, rich_progress_bar) -> str:
        self.rich_progress = rich_progress_bar
        self._initialize_xml_output()
        await self._init_session()

        queue: asyncio.Queue[Tuple[str, int]] = asyncio.Queue()
        await queue.put((self.start_url, 0))

        if self.rich_progress:
            self.progress_task_id = self.rich_progress.add_task(
                f"[cyan]Crawling {self.start_url}...", 
                total=self.max_pages, 
                completed=0
            )
        
        worker_tasks = []
        for i in range(self.concurrency):
            task = asyncio.create_task(self._worker(queue), name=f"Worker-{i+1}")
            worker_tasks.append(task)

        try:
            await queue.join()
        except KeyboardInterrupt:
            self.console.print("\n[bold yellow]Crawl interrupted by user. Finalizing...[/bold yellow]")
        finally:
            # Cancel all worker tasks
            for task in worker_tasks:
                task.cancel()
            # Wait for all tasks to complete their cancellation
            await asyncio.gather(*worker_tasks, return_exceptions=True)
            
            await self._close_session()

        self.output_xml_parts.append('</source>')
        
        if self.rich_progress and self.progress_task_id is not None:
            self.rich_progress.update(self.progress_task_id, completed=self.pages_crawled, description="Crawl finished")

        self.console.print(f"\n[green]Crawl complete.[/green] Pages crawled: {self.pages_crawled}. Failed URLs: {len(self.failed_urls)}")
        if self.failed_urls:
            self.console.print(f"[yellow]Failed URLs ({len(self.failed_urls)}):[/yellow]")
            for failed_url, reason in self.failed_urls[:5]:
                self.console.print(f"  - {failed_url} : {reason}")
            if len(self.failed_urls) > 5:
                self.console.print(f"  ... and {len(self.failed_urls) - 5} more (check verbose logs if enabled).")
        
        return "\n".join(self.output_xml_parts)


async def process_web_crawl(base_url: str, cli_args: object, console: Console, progress_bar) -> str:
    """
    Processes web crawling using the new DocCrawler.
    This function will replace crawl_and_extract_text once DocCrawler is ported.
    
    Args:
        base_url: The URL to start crawling from
        cli_args: Namespace object with CLI arguments for crawler configuration
        console: Rich Console object for output
        progress_bar: Rich Progress bar for tracking progress
        
    Returns:
        XML string with crawled content
    """
    console.print(f"\n[bold green]Initiating web crawl for:[/bold green] [bright_yellow]{base_url}[/bright_yellow]")
    
    # Create and run the DocCrawler
    crawler = DocCrawler(start_url=base_url, cli_args=cli_args, console_obj=console)
    
    try:
        xml_string_output = await crawler.crawl(rich_progress_bar=progress_bar)
        return xml_string_output
    except Exception as e:
        console.print(f"[bold red]Error during web crawl for {base_url}: {e}[/bold red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return f'<source type="web_crawl" base_url="{escape_xml(base_url)}"><error>Crawl failed: {escape_xml(str(e))}</error></source>'


def process_doi_or_pmid(identifier):
    """
    Attempts to fetch a paper PDF via Sci-Hub using DOI or PMID, wrapped in XML.
    Note: Sci-Hub access can be unreliable.
    """
    # Use a more reliable Sci-Hub domain if known, otherwise fallback
    sci_hub_domains = ['https://sci-hub.se/', 'https://sci-hub.st/', 'https://sci-hub.ru/'] # Add more mirrors if needed
    pdf_filename = f"temp_{identifier.replace('/', '-')}.pdf"
    pdf_text = None

    for base_url in sci_hub_domains:
        print(f"Attempting Sci-Hub domain: {base_url} for identifier: {identifier}")
        headers = { # Headers might help avoid blocks
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
        }
        payload = {'request': identifier}

        try:
            # Initial request to Sci-Hub page
            response = requests.post(base_url, headers=headers, data=payload, timeout=60)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the PDF link/embed (Sci-Hub structure varies)
            pdf_url = None
            # Try common patterns: iframe#pdf, button onclick location.href, direct links
            pdf_frame = soup.find('iframe', id='pdf')
            if pdf_frame and pdf_frame.get('src'):
                pdf_url = urljoin(base_url, pdf_frame['src'])
            else:
                # Look for buttons or links directing to the PDF
                pdf_button = soup.find('button', onclick=lambda x: x and 'location.href=' in x)
                if pdf_button:
                    match = re.search(r"location\.href='(//.*?)'", pdf_button['onclick'])
                    if match:
                         # Need to add scheme if missing (often //...)
                        pdf_url_part = match.group(1)
                        if pdf_url_part.startswith("//"):
                            pdf_url = "https:" + pdf_url_part
                        else:
                             pdf_url = urljoin(base_url, pdf_url_part)

            if not pdf_url:
                 print(f"  Could not find PDF link on page from {base_url}")
                 continue # Try next domain

            print(f"  Found potential PDF URL: {pdf_url}")
            # Ensure URL has scheme for requests
            if pdf_url.startswith("//"):
                pdf_url = "https:" + pdf_url
            elif not pdf_url.startswith("http"):
                 pdf_url = urljoin(base_url, pdf_url)


            print(f"  Downloading PDF from: {pdf_url}")
            # Download the PDF file
            pdf_response = requests.get(pdf_url, headers=headers, timeout=120) # Longer timeout for PDF download
            pdf_response.raise_for_status()

            # Check content type again
            if 'application/pdf' not in pdf_response.headers.get('Content-Type', '').lower():
                 print(f"  [bold yellow]Warning:[/bold yellow] Downloaded content is not PDF from {pdf_url}, trying next domain.")
                 continue


            with open(pdf_filename, 'wb') as f:
                f.write(pdf_response.content)

            print("  Extracting text from PDF...")
            with open(pdf_filename, 'rb') as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                text_list = []
                for page in range(len(pdf_reader.pages)):
                    page_text = pdf_reader.pages[page].extract_text()
                    if page_text:
                        text_list.append(page_text)
                pdf_text = " ".join(text_list)

            print(f"Identifier {identifier} processed successfully via {base_url}.")
            break # Success, exit the loop

        except requests.exceptions.Timeout:
             print(f"  Timeout connecting to {base_url} or downloading PDF.")
             continue # Try next domain
        except requests.RequestException as e:
            print(f"  Error with {base_url}: {e}")
            continue # Try next domain
        except Exception as e: # Catch other errors (PDF parsing, etc.)
             print(f"  Error processing identifier {identifier} with {base_url}: {e}")
             continue # Try next domain
        finally:
             # Clean up temp file even if loop continues
             if os.path.exists(pdf_filename):
                 os.remove(pdf_filename)

    # After trying all domains
    if pdf_text is not None:
        # Use XML structure for success
        formatted_text = f'<source type="sci-hub" identifier="{escape_xml(identifier)}">\n'
        formatted_text += pdf_text # Append raw extracted text
        formatted_text += '\n</source>'
        return formatted_text
    else:
        print(f"[bold red]Failed to process identifier {identifier} using all Sci-Hub domains tried.[/bold red]")
        # Use XML structure for error
        error_text = f'<source type="sci-hub" identifier="{escape_xml(identifier)}">\n'
        error_text += f'<error>Could not retrieve or process PDF via Sci-Hub.</error>\n'
        error_text += '</source>'
        return error_text


def process_github_pull_request(pull_request_url):
    """
    Processes a GitHub Pull Request, including details, diff, comments, and associated repo content, wrapped in XML.
    """
    if TOKEN == 'default_token_here':
         print("[bold red]Error:[/bold red] GitHub Token not set. Cannot process GitHub Pull Request.")
         return f'<source type="github_pull_request" url="{escape_xml(pull_request_url)}"><error>GitHub Token not configured.</error></source>'

    url_parts = pull_request_url.split("/")
    if len(url_parts) < 7 or url_parts[-2] != 'pull':
        print(f"[bold red]Invalid GitHub Pull Request URL: {pull_request_url}[/bold red]")
        return f'<source type="github_pull_request" url="{escape_xml(pull_request_url)}"><error>Invalid URL format.</error></source>'

    repo_owner = url_parts[3]
    repo_name = url_parts[4]
    pull_request_number = url_parts[-1]

    api_base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pull_request_number}"
    repo_url_for_content = f"https://github.com/{repo_owner}/{repo_name}" # Base repo URL

    try:
        print(f"Fetching PR data for: {pull_request_url}")
        response = requests.get(api_base_url, headers=headers)
        response.raise_for_status()
        pull_request_data = response.json()

        # Start XML structure
        formatted_text_list = [f'<source type="github_pull_request" url="{escape_xml(pull_request_url)}">']
        formatted_text_list.append(f'<title>{escape_xml(pull_request_data.get("title", "N/A"))}</title>') # Use .get for safety
        formatted_text_list.append('<description>')
        formatted_text_list.append(pull_request_data.get('body', "") or "") # Append raw body, handle None
        formatted_text_list.append('</description>')
        details = (
            f"User: {pull_request_data.get('user', {}).get('login', 'N/A')}, "
            f"State: {pull_request_data.get('state', 'N/A')}, "
            f"Commits: {pull_request_data.get('commits', 'N/A')}, "
            f"Base: {pull_request_data.get('base', {}).get('label', 'N/A')}, "
            f"Head: {pull_request_data.get('head', {}).get('label', 'N/A')}"
        )
        formatted_text_list.append(f'<details>{escape_xml(details)}</details>')

        # Fetch and add the diff
        diff_url = pull_request_data.get("diff_url")
        if diff_url:
            print("Fetching PR diff...")
            diff_response = requests.get(diff_url, headers=headers)
            diff_response.raise_for_status()
            pull_request_diff = diff_response.text
            formatted_text_list.append('\n<diff>')
            formatted_text_list.append(pull_request_diff) # Append raw diff
            formatted_text_list.append('</diff>')
        else:
             formatted_text_list.append('\n<diff><error>Could not retrieve diff URL.</error></diff>')


        # Fetch and add comments (PR comments + review comments)
        all_comments_data = []
        comments_url = pull_request_data.get("comments_url")
        review_comments_url = pull_request_data.get("review_comments_url")

        if comments_url:
            print("Fetching PR comments...")
            comments_response = requests.get(comments_url, headers=headers)
            if comments_response.ok:
                all_comments_data.extend(comments_response.json())
            else:
                 print(f"[bold yellow]Warning:[/bold yellow] Could not fetch PR comments: {comments_response.status_code}")

        if review_comments_url:
             print("Fetching PR review comments...")
             review_comments_response = requests.get(review_comments_url, headers=headers)
             if review_comments_response.ok:
                 all_comments_data.extend(review_comments_response.json())
             else:
                 print(f"[bold yellow]Warning:[/bold yellow] Could not fetch review comments: {review_comments_response.status_code}")


        if all_comments_data:
            formatted_text_list.append('\n<comments>')
             # Optional: Sort comments by creation date or position
            all_comments_data.sort(key=lambda c: c.get("created_at", ""))
            for comment in all_comments_data:
                 author = comment.get('user', {}).get('login', 'N/A')
                 body = comment.get('body', '') or "" # Handle None
                 # Add context if available (e.g., path, line for review comments)
                 path = comment.get('path')
                 line = comment.get('line') or comment.get('original_line')
                 context = f' path="{escape_xml(path)}"' if path else ''
                 context += f' line="{line}"' if line else ''
                 formatted_text_list.append(f'<comment author="{escape_xml(author)}"{context}>')
                 formatted_text_list.append(body) # Append raw comment body
                 formatted_text_list.append('</comment>')
            formatted_text_list.append('</comments>')

        # Add repository content (will include its own <source> tag)
        print(f"Fetching associated repository content from: {repo_url_for_content}")
        # Use the base branch if available, otherwise default branch content
        base_branch_ref = pull_request_data.get('base', {}).get('ref')
        repo_url_with_ref = f"{repo_url_for_content}/tree/{base_branch_ref}" if base_branch_ref else repo_url_for_content
        repo_content = process_github_repo(repo_url_with_ref) # process_github_repo returns XML string

        formatted_text_list.append('\n<!-- Associated Repository Content -->') # XML Comment
        formatted_text_list.append(repo_content) # Append the XML output directly

        formatted_text_list.append('\n</source>') # Close main PR source tag

        print(f"Pull request {pull_request_number} and repository content processed successfully.")
        return "\n".join(formatted_text_list)

    except requests.RequestException as e:
        print(f"[bold red]Error fetching GitHub PR data for {pull_request_url}: {e}[/bold red]")
        return f'<source type="github_pull_request" url="{escape_xml(pull_request_url)}"><error>Failed to fetch PR data: {escape_xml(str(e))}</error></source>'
    except Exception as e: # Catch other potential errors
         print(f"[bold red]Unexpected error processing GitHub PR {pull_request_url}: {e}[/bold red]")
         return f'<source type="github_pull_request" url="{escape_xml(pull_request_url)}"><error>Unexpected error: {escape_xml(str(e))}</error></source>'


def process_github_issue(issue_url):
    """
    Processes a GitHub Issue, including details, comments, and associated repo content, wrapped in XML.
    """
    if TOKEN == 'default_token_here':
         print("[bold red]Error:[/bold red] GitHub Token not set. Cannot process GitHub Issue.")
         return f'<source type="github_issue" url="{escape_xml(issue_url)}"><error>GitHub Token not configured.</error></source>'

    url_parts = issue_url.split("/")
    if len(url_parts) < 7 or url_parts[-2] != 'issues':
        print(f"[bold red]Invalid GitHub Issue URL: {issue_url}[/bold red]")
        return f'<source type="github_issue" url="{escape_xml(issue_url)}"><error>Invalid URL format.</error></source>'

    repo_owner = url_parts[3]
    repo_name = url_parts[4]
    issue_number = url_parts[-1]

    api_base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}"
    repo_url_for_content = f"https://github.com/{repo_owner}/{repo_name}"

    try:
        print(f"Fetching issue data for: {issue_url}")
        response = requests.get(api_base_url, headers=headers)
        response.raise_for_status()
        issue_data = response.json()

        # Start XML structure
        formatted_text_list = [f'<source type="github_issue" url="{escape_xml(issue_url)}">']
        formatted_text_list.append(f'<title>{escape_xml(issue_data.get("title", "N/A"))}</title>')
        formatted_text_list.append('<description>')
        formatted_text_list.append(issue_data.get('body', "") or "") # Append raw body, handle None
        formatted_text_list.append('</description>')
        details = (
             f"User: {issue_data.get('user', {}).get('login', 'N/A')}, "
             f"State: {issue_data.get('state', 'N/A')}, "
             f"Number: {issue_data.get('number', 'N/A')}"
         )
        formatted_text_list.append(f'<details>{escape_xml(details)}</details>')


        # Fetch and add comments
        comments_data = []
        comments_url = issue_data.get("comments_url")
        if comments_url:
            print("Fetching issue comments...")
            comments_response = requests.get(comments_url, headers=headers)
            if comments_response.ok:
                 comments_data = comments_response.json()
            else:
                 print(f"[bold yellow]Warning:[/bold yellow] Could not fetch issue comments: {comments_response.status_code}")


        if comments_data:
            formatted_text_list.append('\n<comments>')
            # Optional: Sort comments by creation date
            comments_data.sort(key=lambda c: c.get("created_at", ""))
            for comment in comments_data:
                author = comment.get('user', {}).get('login', 'N/A')
                body = comment.get('body', '') or "" # Handle None
                formatted_text_list.append(f'<comment author="{escape_xml(author)}">')
                formatted_text_list.append(body) # Append raw comment body
                formatted_text_list.append('</comment>')
            formatted_text_list.append('</comments>')

        # Add repository content (will include its own <source> tag)
        print(f"Fetching associated repository content from: {repo_url_for_content}")
        # Fetch default branch content for issues
        repo_content = process_github_repo(repo_url_for_content) # process_github_repo returns XML string

        formatted_text_list.append('\n<!-- Associated Repository Content -->') # XML Comment
        formatted_text_list.append(repo_content) # Append the XML output directly

        formatted_text_list.append('\n</source>') # Close main issue source tag

        print(f"Issue {issue_number} and repository content processed successfully.")
        return "\n".join(formatted_text_list)

    except requests.RequestException as e:
        print(f"[bold red]Error fetching GitHub issue data for {issue_url}: {e}[/bold red]")
        return f'<source type="github_issue" url="{escape_xml(issue_url)}"><error>Failed to fetch issue data: {escape_xml(str(e))}</error></source>'
    except Exception as e: # Catch other potential errors
         print(f"[bold red]Unexpected error processing GitHub issue {issue_url}: {e}[/bold red]")
         return f'<source type="github_issue" url="{escape_xml(issue_url)}"><error>Unexpected error: {escape_xml(str(e))}</error></source>'


def combine_xml_outputs(outputs):
    """
    Combines multiple XML outputs into one cohesive XML document
    under a <onefilellm_output> root tag.
    """
    if not outputs:
        return None
    
    # If only one output, wrap it in onefilellm_output for consistency
    # instead of returning it as-is
    
    # Create a wrapper for multiple sources
    combined = ['<onefilellm_output>']
    
    # Add each source
    for output in outputs:
        # Remove any XML declaration if present (rare but possible)
        output = re.sub(r'<\?xml[^>]+\?>', '', output).strip()
        combined.append(output)
    
    # Close the wrapper
    combined.append('</onefilellm_output>')
    
    return '\n'.join(combined)

async def process_input(input_path, args, progress=None, task=None):
    """
    Process a single input path and return the XML output.
    Extracted from main() for reuse with multiple inputs.
    """
    console = Console()
    urls_list_file = "processed_urls.txt"
    
    try:
        if task:
            progress.update(task, description=f"[bright_blue]Processing {input_path}...")
        
        console.print(f"\n[bold bright_green]Processing:[/bold bright_green] [bold bright_yellow]{input_path}[/bold bright_yellow]\n")
        
        # Input type detection logic
        if "github.com" in input_path:
            if "/pull/" in input_path:
                result = process_github_pull_request(input_path)
            elif "/issues/" in input_path:
                result = process_github_issue(input_path)
            else: # Assume repository URL
                result = process_github_repo(input_path)
        elif urlparse(input_path).scheme in ["http", "https"]:
            if "youtube.com" in input_path or "youtu.be" in input_path:
                result = fetch_youtube_transcript(input_path)
            elif "arxiv.org/abs/" in input_path:
                result = process_arxiv_pdf(input_path)
            elif input_path.lower().endswith(('.pdf')): # Direct PDF link
                # Simplified: wrap direct PDF processing if needed, or treat as web crawl
                print("[bold yellow]Direct PDF URL detected - treating as single-page crawl.[/bold yellow]")
                crawl_result = crawl_and_extract_text(input_path, max_depth=0, include_pdfs=True, ignore_epubs=True)
                result = crawl_result['content']
                if crawl_result['processed_urls']:
                    with open(urls_list_file, 'w', encoding='utf-8') as urls_file:
                        urls_file.write('\n'.join(crawl_result['processed_urls']))
            elif input_path.lower().endswith(('.xls', '.xlsx')): # Direct Excel file link
                console.print(f"Processing Excel file from URL: {input_path}")
                try:
                    filename = os.path.basename(urlparse(input_path).path)
                    base_filename = os.path.splitext(filename)[0]
                    
                    # Get markdown tables for each sheet
                    result_parts = [f'<source type="web_excel" url="{escape_xml(input_path)}">']
                    
                    try:
                        markdown_tables = excel_to_markdown_from_url(input_path)
                        for sheet_name, markdown in markdown_tables.items():
                            virtual_name = f"{base_filename}_{sheet_name}.md"
                            result_parts.append(f'<file path="{escape_xml(virtual_name)}">')
                            result_parts.append(markdown)
                            result_parts.append('</file>')
                    except Exception as e:
                        result_parts.append(f'<e>Failed to process Excel file from URL: {escape_xml(str(e))}</e>')
                    
                    result_parts.append('</source>')
                    result = '\n'.join(result_parts)
                except Exception as e:
                    console.print(f"[bold red]Error processing Excel URL {input_path}: {e}[/bold red]")
                    result = f'<source type="web_excel" url="{escape_xml(input_path)}"><e>Failed to process Excel file: {escape_xml(str(e))}</e></source>'
            # Process URL directly if it ends with a recognized file extension
            elif any(input_path.lower().endswith(ext) for ext in [ext for ext in ['.txt', '.md', '.rst', '.tex', '.html', '.htm', '.css', '.js', '.ts', '.py', '.java', '.c', '.cpp', '.h', '.hpp', '.cs', '.rb', '.php', '.swift', '.kt', '.scala', '.rs', '.lua', '.pl', '.sh', '.bash', '.zsh', '.ps1', '.sql', '.groovy', '.dart', '.json', '.yaml', '.yml', '.xml', '.toml', '.ini', '.cfg', '.conf', '.properties', '.csv', '.tsv', '.proto', '.graphql', '.tf', '.tfvars', '.hcl'] if is_allowed_filetype(f"test{ext}")] if ext != '.pdf'):
                console.print(f"Processing direct file URL: {input_path}")
                file_content = _download_and_read_file(input_path)
                filename = os.path.basename(urlparse(input_path).path)
                result = (f'<source type="web_file" url="{escape_xml(input_path)}">\n'
                         f'<file path="{escape_xml(filename)}">\n'
                         f'{file_content}\n'
                         f'</file>\n'
                         f'</source>')
            else: # Assume general web URL for crawling
                # Use the new async DocCrawler
                result = await process_web_crawl(input_path, args, console, progress)
                # Note: The new crawler doesn't return processed_urls separately,
                # they're included in the XML output if needed
        # Basic check for DOI (starts with 10.) or PMID (all digits)
        elif (input_path.startswith("10.") and "/" in input_path) or input_path.isdigit():
            result = process_doi_or_pmid(input_path)
        elif os.path.isdir(input_path): # Check if it's a local directory
            result = process_local_folder(input_path)
        elif os.path.isfile(input_path): # Handle single local file
            if input_path.lower().endswith('.pdf'): # Case-insensitive check
                console.print(f"Processing single local PDF file: {input_path}") # Use console for consistency
                pdf_content_text = _process_pdf_content_from_path(input_path)
                # Structure for a single local PDF file
                result = (f'<source type="local_file" path="{escape_xml(input_path)}">\n'
                         f'<file path="{escape_xml(os.path.basename(input_path))}">\n' # Wrapping content in <file>
                         f'{pdf_content_text}\n' # Raw PDF text or error message
                         f'</file>\n'
                         f'</source>')
            elif input_path.lower().endswith(('.xls', '.xlsx')): # Case-insensitive check for Excel files
                console.print(f"Processing single local Excel file: {input_path}")
                try:
                    filename = os.path.basename(input_path)
                    base_filename = os.path.splitext(filename)[0]
                    
                    # Get markdown tables for each sheet
                    result_parts = [f'<source type="local_file" path="{escape_xml(input_path)}">']
                    
                    try:
                        markdown_tables = excel_to_markdown(input_path)
                        for sheet_name, markdown in markdown_tables.items():
                            virtual_name = f"{base_filename}_{sheet_name}.md"
                            result_parts.append(f'<file path="{escape_xml(virtual_name)}">')
                            result_parts.append(markdown)
                            result_parts.append('</file>')
                    except Exception as e:
                        result_parts.append(f'<e>Failed to process Excel file: {escape_xml(str(e))}</e>')
                    
                    result_parts.append('</source>')
                    result = '\n'.join(result_parts)
                except Exception as e:
                    console.print(f"[bold red]Error processing Excel file {input_path}: {e}[/bold red]")
                    result = f'<source type="local_file" path="{escape_xml(input_path)}"><e>Failed to process Excel file: {escape_xml(str(e))}</e></source>'
            else:
                # Existing logic for other single files
                console.print(f"Processing single local file: {input_path}") # Use console
                relative_path = os.path.basename(input_path)
                file_content = safe_file_read(input_path)
                result = (f'<source type="local_file" path="{escape_xml(input_path)}">\n'
                         f'<file path="{escape_xml(relative_path)}">\n'
                         f'{file_content}\n'
                         f'</file>\n'
                         f'</source>')
        else: # Input not recognized
            raise ValueError(f"Input path or URL type not recognized: {input_path}")
            
        return result
        
    except Exception as e:
        console.print(f"\n[bold red]Error processing {input_path}:[/bold red] {str(e)}")
        # Return an error-wrapped source instead of raising
        return f'<source type="error" path="{escape_xml(input_path)}">\n<e>Failed to process: {escape_xml(str(e))}</e>\n</source>'


def create_argument_parser():
    """Create and return the argument parser with all CLI options."""
    parser = argparse.ArgumentParser(
        description="OneFileLLM - Content Aggregator for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a GitHub repository
  python onefilellm.py https://github.com/user/repo
  
  # Process multiple inputs
  python onefilellm.py file1.txt https://example.com file2.pdf
  
  # Use advanced web crawler with custom settings
  python onefilellm.py https://example.com --crawl-max-depth 5 --crawl-max-pages 200
  
  # Process from clipboard
  python onefilellm.py --clipboard
  
  # Process from stdin
  cat file.txt | python onefilellm.py -
"""
    )
    
    # Positional arguments
    parser.add_argument('inputs', nargs='*', help='Input paths, URLs, or aliases to process')
    
    # Input source options
    parser.add_argument('-c', '--clipboard', action='store_true',
                        help='Process text from clipboard')
    parser.add_argument('-f', '--format', choices=['text', 'markdown', 'json', 'html', 'yaml', 'doculing', 'markitdown'],
                        help='Override format detection for text input')
    
    # Alias management
    parser.add_argument('--add-alias', nargs=2, metavar=('ALIAS_NAME', 'SOURCE'),
                        help='Create an alias for a source')
    parser.add_argument('--alias-from-clipboard', metavar='ALIAS_NAME',
                        help='Create alias from clipboard content (one source per line)')
    
    # Web crawler options
    crawler_group = parser.add_argument_group('Web Crawler Options')
    crawler_group.add_argument('--crawl-max-depth', type=int, default=3,
                               help='Maximum crawl depth (default: 3)')
    crawler_group.add_argument('--crawl-max-pages', type=int, default=1000,
                               help='Maximum pages to crawl (default: 1000)')
    crawler_group.add_argument('--crawl-user-agent', default='OneFileLLMCrawler/1.1',
                               help='User agent for web requests (default: OneFileLLMCrawler/1.1)')
    crawler_group.add_argument('--crawl-delay', type=float, default=0.25,
                               help='Delay between requests in seconds (default: 0.25)')
    crawler_group.add_argument('--crawl-include-pattern',
                               help='Regex pattern for URLs to include')
    crawler_group.add_argument('--crawl-exclude-pattern',
                               help='Regex pattern for URLs to exclude')
    crawler_group.add_argument('--crawl-timeout', type=int, default=20,
                               help='Request timeout in seconds (default: 20)')
    crawler_group.add_argument('--crawl-include-images', action='store_true',
                               help='Include image URLs in output')
    crawler_group.add_argument('--crawl-no-include-code', action='store_false', dest='crawl_include_code',
                               default=True, help='Exclude code blocks from output')
    crawler_group.add_argument('--crawl-no-extract-headings', action='store_false', dest='crawl_extract_headings',
                               default=True, help='Exclude heading extraction')
    crawler_group.add_argument('--crawl-follow-links', action='store_true',
                               help='Follow links to external domains')
    crawler_group.add_argument('--crawl-no-clean-html', action='store_false', dest='crawl_clean_html',
                               default=True, help='Disable readability cleaning')
    crawler_group.add_argument('--crawl-no-strip-js', action='store_false', dest='crawl_strip_js',
                               default=True, help='Keep JavaScript code')
    crawler_group.add_argument('--crawl-no-strip-css', action='store_false', dest='crawl_strip_css',
                               default=True, help='Keep CSS styles')
    crawler_group.add_argument('--crawl-no-strip-comments', action='store_false', dest='crawl_strip_comments',
                               default=True, help='Keep HTML comments')
    crawler_group.add_argument('--crawl-respect-robots', action='store_true', dest='crawl_respect_robots',
                               default=False, help='Respect robots.txt (default: ignore for backward compatibility)')
    crawler_group.add_argument('--crawl-concurrency', type=int, default=3,
                               help='Number of concurrent requests (default: 3)')
    crawler_group.add_argument('--crawl-restrict-path', action='store_true',
                               help='Restrict crawl to paths under start URL')
    crawler_group.add_argument('--crawl-no-include-pdfs', action='store_false', dest='crawl_include_pdfs',
                               default=True, help='Skip PDF files')
    crawler_group.add_argument('--crawl-no-ignore-epubs', action='store_false', dest='crawl_ignore_epubs',
                               default=True, help='Include EPUB files')
    
    return parser


async def main():
    console = Console()
    
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # --- Handle alias management commands ---
    if args.add_alias:
        alias_name, source = args.add_alias
        handle_add_alias(alias_name, source, console)
        return
    
    if args.alias_from_clipboard:
        handle_alias_from_clipboard(args.alias_from_clipboard, console)
        return
    
    # --- Handle stream input modes ---
    is_stream_input_mode = False
    stream_source_dict = {}
    stream_content_to_process = None
    user_format_override = args.format
    
    # Check for stdin input ('-' in inputs)
    if '-' in args.inputs:
        is_stream_input_mode = True
        stream_source_dict = {'type': 'stdin'}
        # Remove '-' from inputs list
        args.inputs = [inp for inp in args.inputs if inp != '-']
    
    # Check for clipboard input
    elif args.clipboard:
        is_stream_input_mode = True
        stream_source_dict = {'type': 'clipboard'}

    # Process stream input if specified

    if is_stream_input_mode:
        if stream_source_dict['type'] == 'stdin':
            if not sys.stdin.isatty():
                console.print("[bright_blue]Reading from standard input...[/bright_blue]")
                stream_content_to_process = read_from_stdin()
                if stream_content_to_process is None or not stream_content_to_process.strip():
                    console.print("[bold red]Error: No input received from standard input or input is empty.[/bold red]")
                    return
            else:
                console.print("[bold red]Error: '-' specified but no data piped to stdin.[/bold red]")
                console.print("To use standard input, pipe data like: `your_command | python onefilellm.py -`")
                return
        elif stream_source_dict['type'] == 'clipboard':
            console.print("[bright_blue]Reading from clipboard...[/bright_blue]")
            stream_content_to_process = read_from_clipboard()
            if stream_content_to_process is None or not stream_content_to_process.strip():
                console.print("[bold red]Error: Clipboard is empty, does not contain text, or could not be accessed.[/bold red]")
                if sys.platform.startswith('linux'):
                    console.print("[yellow]On Linux, you might need to install 'xclip' or 'xsel': `sudo apt install xclip`[/yellow]")
                return
    
    # --- Main Processing Logic Dispatch ---
    if is_stream_input_mode:
        # We are in stream processing mode
        if stream_content_to_process: # Ensure we have content
            xml_output_for_stream = process_text_stream(
                stream_content_to_process, 
                stream_source_dict, 
                console, # Pass the console object
                user_format_override
            )

            if xml_output_for_stream:
                # For a single stream input, it forms the entire <onefilellm_output>
                final_combined_output = f"<onefilellm_output>\n{xml_output_for_stream}\n</onefilellm_output>"

                # Use existing output mechanisms
                output_file = "output.xml" 
                processed_file = "compressed_output.txt" 

                console.print(f"\nWriting output to {output_file}...")
                with open(output_file, "w", encoding="utf-8") as file:
                    file.write(final_combined_output)
                console.print("Output written successfully.")

                uncompressed_token_count = get_token_count(final_combined_output)
                console.print(f"\n[bright_green]Content Token Count (approx):[/bright_green] [bold bright_cyan]{uncompressed_token_count}[/bold bright_cyan]")

                if ENABLE_COMPRESSION_AND_NLTK:
                    # ... (existing compression logic using output_file and processed_file) ...
                    console.print("\n[bright_magenta]Compression and NLTK processing enabled.[/bright_magenta]")
                    print(f"Preprocessing text and writing compressed output to {processed_file}...")
                    preprocess_text(output_file, processed_file) # Pass correct output_file
                    print("Compressed file written.")
                    compressed_text_content = safe_file_read(processed_file)
                    compressed_token_count = get_token_count(compressed_text_content)
                    console.print(f"[bright_green]Compressed Token Count (approx):[/bright_green] [bold bright_cyan]{compressed_token_count}[/bold bright_cyan]")

                try:
                    pyperclip.copy(final_combined_output)
                    console.print(f"\n[bright_white]The contents of [bold bright_blue]{output_file}[/bold bright_blue] have been copied to the clipboard.[/bright_white]")
                except Exception as clip_err:
                    console.print(f"\n[bold yellow]Warning:[/bold yellow] Could not copy to clipboard: {clip_err}")
            else:
                console.print("[bold red]Error: Text stream processing failed to generate output.[/bold red]")
        # else: stream_content_to_process was None or empty, error already printed.
        return # Exit after stream processing

    # --- ELSE: Fall through to existing file/URL/alias processing logic ---

    # Updated intro text to reflect XML output
    intro_text = Text("\nProcesses Inputs and Wraps Content in XML:\n", style="dodger_blue1")
    input_types = [
        ("• Local folder path", "bright_white"),
        ("• GitHub repository URL", "bright_white"),
        ("• GitHub pull request URL (PR + Repo)", "bright_white"),
        ("• GitHub issue URL (Issue + Repo)", "bright_white"),
        ("• Documentation URL (Web Crawl)", "bright_white"),
        ("• YouTube video URL (Transcript)", "bright_white"),
        ("• ArXiv Paper URL (PDF Text)", "bright_white"),
        ("• DOI or PMID (via Sci-Hub, best effort)", "bright_white"),
        ("• Text from stdin (e.g., `cat file.txt | onefilellm -`)", "light_sky_blue1"), # New
        ("• Text from clipboard (e.g., `onefilellm --clipboard`)", "light_sky_blue1"), # New
    ]

    for input_type, color in input_types:
        intro_text.append(f"\n{input_type}", style=color)
    intro_text.append("\n\nOutput is saved to file and copied to clipboard.", style="dim")
    intro_text.append("\nContent within XML tags remains unescaped for readability.", style="dim")
    intro_text.append("\nMultiple inputs can be provided as command line arguments.", style="bright_green")

    intro_panel = Panel(
        intro_text,
        expand=False,
        border_style="bold",
        title="[bright_white]onefilellm - Content Aggregator[/bright_white]",
        title_align="center",
        padding=(1, 1),
    )
    console.print(intro_panel)

    # --- Determine Input Paths (resolve aliases) ---
    final_input_sources = []
    if args.inputs:
        for arg_string in args.inputs:
            final_input_sources.extend(resolve_single_input_source(arg_string, console))
    
    if not final_input_sources and not is_stream_input_mode:
        # No inputs provided - show intro panel and prompt for input
        user_input_string = Prompt.ask("\n[bold dodger_blue1]Enter the path, URL, or alias[/bold dodger_blue1]", console=console).strip()
        if user_input_string:
            final_input_sources.extend(resolve_single_input_source(user_input_string, console))
    
    # For minimal changes later, assign to input_paths
    input_paths = final_input_sources
    # --- End Determine Input Paths ---

    if not input_paths:
        console.print("[yellow]No valid input sources provided. Exiting.[/yellow]")
        return

    # Define output filenames
    output_file = "output.xml" # Changed extension to reflect content
    processed_file = "compressed_output.txt" # Keep as txt for compressed
    
    # List to collect individual outputs
    outputs = []

    with Progress(
        TextColumn("[bold bright_blue]{task.description}"),
        BarColumn(bar_width=None),
        TimeRemainingColumn(),
        console=console,
        transient=True # Clear progress on exit
    ) as progress:

        task = progress.add_task("[bright_blue]Processing...", total=None) # Indeterminate task

        try:
            # Process each input path
            for input_path in input_paths:
                result = await process_input(input_path, args, progress, task)
                if result:
                    outputs.append(result)
                    console.print(f"[green]Successfully processed: {input_path}[/green]")
                else:
                    console.print(f"[yellow]No output generated for: {input_path}[/yellow]")
            
            # Combine all outputs into one final output
            if not outputs:
                raise RuntimeError("No inputs were successfully processed.")
                
            final_output = combine_xml_outputs(outputs)

            # --- Output Generation ---
            if final_output is None:
                 raise RuntimeError("Processing failed to produce any output.")

            # Write the main (uncompressed) XML output
            print(f"\nWriting output to {output_file}...")
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(final_output)
            print("Output written successfully.")

            # Get token count for the main output (strips XML tags)
            uncompressed_token_count = get_token_count(final_output)
            console.print(f"\n[bright_green]Content Token Count (approx):[/bright_green] [bold bright_cyan]{uncompressed_token_count}[/bold bright_cyan]")

            # --- Conditional Compression Block ---
            if ENABLE_COMPRESSION_AND_NLTK:
                console.print("\n[bright_magenta]Compression and NLTK processing enabled.[/bright_magenta]")
                print(f"Preprocessing text and writing compressed output to {processed_file}...")
                # Process the text (input is the XML file, output is compressed txt)
                preprocess_text(output_file, processed_file)
                print("Compressed file written.")

                # Get token count for the compressed file (should be plain text)
                compressed_text = safe_file_read(processed_file)
                compressed_token_count = get_token_count(compressed_text) # Pass compressed text directly
                console.print(f"[bright_green]Compressed Token Count (approx):[/bright_green] [bold bright_cyan]{compressed_token_count}[/bold bright_cyan]")
                console.print(f"\n[bold bright_blue]{output_file}[/bold bright_blue] (main XML) and [bold bright_yellow]{processed_file}[/bold bright_yellow] (compressed text) created.")
            else:
                 console.print(f"\n[bold bright_blue]{output_file}[/bold bright_blue] (main XML) has been created.")
            # --- End Conditional Compression Block ---


            # Copy the main XML output to clipboard
            try:
                 pyperclip.copy(final_output)
                 console.print(f"\n[bright_white]The contents of [bold bright_blue]{output_file}[/bold bright_blue] have been copied to the clipboard.[/bright_white]")
            except Exception as clip_err: # Catch potential pyperclip errors
                 console.print(f"\n[bold yellow]Warning:[/bold yellow] Could not copy to clipboard: {clip_err}")


        except Exception as e:
            console.print(f"\n[bold red]An error occurred during processing:[/bold red]")
            console.print_exception(show_locals=False) # Print traceback
            # Optionally write the partial output if it exists
            if outputs:
                 try:
                     error_filename = "error_output.xml"
                     partial_output = combine_xml_outputs(outputs)
                     with open(error_filename, "w", encoding="utf-8") as err_file:
                         err_file.write(partial_output)
                     console.print(f"[yellow]Partial output written to {error_filename}[/yellow]")
                 except Exception as write_err:
                     console.print(f"[red]Could not write partial output: {write_err}[/red]")

        finally:
             progress.stop_task(task)
             progress.refresh() # Ensure progress bar clears


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    asyncio.run(main())
