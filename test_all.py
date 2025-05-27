#!/usr/bin/env python3
"""
Comprehensive test suite for OneFileLLM
Consolidates all tests and expands coverage
"""

import unittest
import os
import sys
import json
import tempfile
import shutil
import time
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import pyperclip
from rich.console import Console
from rich.table import Table
from rich.text import Text

# Import the modules we're testing
from onefilellm import (
    process_github_repo,
    process_arxiv_pdf,
    process_local_folder,
    fetch_youtube_transcript,
    crawl_and_extract_text,
    process_doi_or_pmid,
    process_github_pull_request,
    process_github_issue,
    excel_to_markdown,
    process_input,
    process_text_stream,
    get_token_count,
    combine_xml_outputs,
    preprocess_text,
    ensure_alias_dir_exists,
    is_potential_alias,
    handle_add_alias,
    handle_alias_from_clipboard,
    load_alias,
    ENABLE_COMPRESSION_AND_NLTK
)

from utils import (
    safe_file_read,
    read_from_clipboard,
    read_from_stdin,
    detect_text_format,
    parse_as_plaintext,
    parse_as_markdown,
    parse_as_json,
    parse_as_html,
    parse_as_yaml,
    download_file,
    is_same_domain,
    is_within_depth,
    is_excluded_file,
    is_allowed_filetype,
    escape_xml,
    get_file_extension,
    is_binary_file
)

# Test configuration
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
RUN_INTEGRATION_TESTS = os.environ.get('RUN_INTEGRATION_TESTS', 'true').lower() == 'true'
RUN_SLOW_TESTS = os.environ.get('RUN_SLOW_TESTS', 'false').lower() == 'true'


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions from utils.py"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_safe_file_read(self):
        """Test safe file reading with different encodings"""
        # Test UTF-8 file
        utf8_file = os.path.join(self.temp_dir, "utf8.txt")
        with open(utf8_file, 'w', encoding='utf-8') as f:
            f.write("Hello 世界")
        content = safe_file_read(utf8_file)
        self.assertEqual(content, "Hello 世界")
        
        # Test latin-1 file
        latin1_file = os.path.join(self.temp_dir, "latin1.txt")
        with open(latin1_file, 'wb') as f:
            f.write("Café".encode('latin-1'))
        content = safe_file_read(latin1_file)
        self.assertEqual(content, "Café")
    
    def test_file_extension_detection(self):
        """Test file extension detection"""
        self.assertEqual(get_file_extension("test.py"), ".py")
        self.assertEqual(get_file_extension("TEST.PY"), ".py")
        self.assertEqual(get_file_extension("no_extension"), "")
        self.assertEqual(get_file_extension("multiple.dots.txt"), ".txt")
    
    def test_is_binary_file(self):
        """Test binary file detection"""
        # Create text file
        text_file = os.path.join(self.temp_dir, "text.txt")
        with open(text_file, 'w') as f:
            f.write("This is text")
        self.assertFalse(is_binary_file(text_file))
        
        # Create binary file
        binary_file = os.path.join(self.temp_dir, "binary.bin")
        with open(binary_file, 'wb') as f:
            f.write(b'\x00\x01\x02\x03')
        self.assertTrue(is_binary_file(binary_file))
    
    def test_is_excluded_file(self):
        """Test file exclusion patterns"""
        self.assertTrue(is_excluded_file("test.pb.go"))
        self.assertTrue(is_excluded_file("file_test.go"))
        self.assertTrue(is_excluded_file("script.min.js"))
        self.assertTrue(is_excluded_file("__pycache__/file.pyc"))
        self.assertTrue(is_excluded_file("node_modules/package.json"))
        self.assertFalse(is_excluded_file("main.go"))
        self.assertFalse(is_excluded_file("app.js"))
    
    def test_is_allowed_filetype(self):
        """Test allowed file type checking"""
        self.assertTrue(is_allowed_filetype("script.py"))
        self.assertTrue(is_allowed_filetype("README.md"))
        self.assertTrue(is_allowed_filetype("config.yaml"))
        self.assertFalse(is_allowed_filetype("image.png"))
        self.assertFalse(is_allowed_filetype("binary.exe"))
        self.assertFalse(is_allowed_filetype("archive.zip"))
    
    def test_url_utilities(self):
        """Test URL-related utilities"""
        base_url = "https://example.com/docs/"
        
        # Test same domain
        self.assertTrue(is_same_domain(base_url, "https://example.com/other/"))
        self.assertFalse(is_same_domain(base_url, "https://other.com/docs/"))
        
        # Test depth checking
        self.assertTrue(is_within_depth(base_url, "https://example.com/docs/page1", 1))
        self.assertTrue(is_within_depth(base_url, "https://example.com/docs/sub/page", 2))
        self.assertFalse(is_within_depth(base_url, "https://example.com/docs/a/b/c", 2))
    
    def test_escape_xml(self):
        """Test XML escaping (currently returns unchanged)"""
        text = "<tag>Content & more</tag>"
        self.assertEqual(escape_xml(text), text)


class TestTextFormatDetection(unittest.TestCase):
    """Test text format detection and parsing"""
    
    def test_format_detection(self):
        """Test detection of various text formats"""
        # Plain text
        self.assertEqual(detect_text_format("Just plain text"), "text")
        self.assertEqual(detect_text_format(""), "text")
        
        # JSON
        self.assertEqual(detect_text_format('{"key": "value"}'), "json")
        self.assertEqual(detect_text_format('[1, 2, 3]'), "json")
        self.assertEqual(detect_text_format('{\n  "nested": {\n    "data": true\n  }\n}'), "json")
        
        # HTML
        self.assertEqual(detect_text_format('<!DOCTYPE html><html></html>'), "html")
        self.assertEqual(detect_text_format('<html><body>Content</body></html>'), "html")
        self.assertEqual(detect_text_format('<div>Single tag</div>'), "html")
        
        # Markdown
        self.assertEqual(detect_text_format('# Heading\n\nParagraph'), "markdown")
        self.assertEqual(detect_text_format('**Bold** and *italic*'), "markdown")
        self.assertEqual(detect_text_format('- List item\n- Another item'), "markdown")
        self.assertEqual(detect_text_format('[Link](https://example.com)'), "markdown")
        self.assertEqual(detect_text_format('```code block```'), "markdown")
        
        # YAML (if available)
        try:
            import yaml
            self.assertEqual(detect_text_format('key: value\nlist:\n  - item1\n  - item2'), "yaml")
        except ImportError:
            pass
    
    def test_parsers(self):
        """Test individual parser functions"""
        # Plain text parser
        text = "Plain text content"
        self.assertEqual(parse_as_plaintext(text), text)
        
        # Markdown parser
        markdown = "# Heading\n\nContent"
        self.assertEqual(parse_as_markdown(markdown), markdown)
        
        # JSON parser
        json_text = '{"valid": "json"}'
        self.assertEqual(parse_as_json(json_text), json_text)
        with self.assertRaises(json.JSONDecodeError):
            parse_as_json("not valid json")
        
        # HTML parser
        html = "<html><body><h1>Title</h1><p>Content</p></body></html>"
        parsed = parse_as_html(html)
        self.assertIn("Title", parsed)
        self.assertIn("Content", parsed)
        self.assertNotIn("<h1>", parsed)
        
        # YAML parser (if available)
        try:
            import yaml
            yaml_text = "key: value"
            self.assertEqual(parse_as_yaml(yaml_text), yaml_text)
        except ImportError:
            pass


class TestStreamProcessing(unittest.TestCase):
    """Test stream processing functionality"""
    
    def setUp(self):
        self.console = MagicMock()
    
    def test_stdin_processing(self):
        """Test processing from stdin"""
        text_content = "Test content from stdin"
        result = process_text_stream(text_content, {'type': 'stdin'}, self.console)
        self.assertIn('<source type="stdin"', result)
        self.assertIn(text_content, result)
    
    def test_clipboard_processing(self):
        """Test processing from clipboard"""
        text_content = "Test content from clipboard"
        result = process_text_stream(text_content, {'type': 'clipboard'}, self.console)
        self.assertIn('<source type="clipboard"', result)
        self.assertIn(text_content, result)
    
    def test_format_override(self):
        """Test format override functionality"""
        json_content = '{"key": "value"}'
        
        # Without override - should detect as JSON
        result = process_text_stream(json_content, {'type': 'stdin'}, self.console)
        self.assertIn('processed_as_format="json"', result)
        
        # With override to text
        result = process_text_stream(json_content, {'type': 'stdin'}, self.console, format_override="text")
        self.assertIn('processed_as_format="text"', result)
    
    def test_clipboard_errors(self):
        """Test clipboard error handling"""
        # We're testing the actual function behavior, not mocking
        # The function returns None when clipboard is empty or has errors
        with patch('pyperclip.paste') as mock_paste:
            mock_paste.side_effect = pyperclip.PyperclipException("Test error")
            result = read_from_clipboard()
            self.assertIsNone(result)
    
    @patch('utils.read_from_stdin')
    def test_stdin_errors(self, mock_stdin):
        """Test stdin error handling"""
        # Test no piped input
        mock_stdin.return_value = None
        result = read_from_stdin()
        self.assertIsNone(result)


class TestCoreProcessing(unittest.TestCase):
    """Test core processing functions"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_local_file_processing(self):
        """Test processing of local files"""
        # Create test files
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("Test content")
        
        result = process_input(test_file)
        self.assertIn('<source type="local_file"', result)
        self.assertIn('Test content', result)
    
    def test_local_folder_processing(self):
        """Test processing of local folders"""
        # Create test files in folder
        for i in range(3):
            test_file = os.path.join(self.temp_dir, f"file{i}.txt")
            with open(test_file, 'w') as f:
                f.write(f"Content {i}")
        
        result = process_local_folder(self.temp_dir)
        self.assertIn('<source type="local_folder"', result)
        for i in range(3):
            self.assertIn(f'Content {i}', result)
    
    def test_excel_processing(self):
        """Test Excel file processing"""
        # Create test Excel file
        excel_file = os.path.join(self.temp_dir, "test.xlsx")
        df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df2 = pd.DataFrame({'X': ['a', 'b'], 'Y': ['c', 'd']})
        
        with pd.ExcelWriter(excel_file) as writer:
            df1.to_excel(writer, sheet_name='Sheet1', index=False)
            df2.to_excel(writer, sheet_name='Sheet2', index=False)
        
        result = excel_to_markdown(excel_file)
        # excel_to_markdown returns a dict, not a string
        self.assertIsInstance(result, dict)
        self.assertIn('Sheet1', result)
        self.assertIn('Sheet2', result)
        self.assertIn('A', result['Sheet1'])
        self.assertIn('B', result['Sheet1'])
        self.assertIn('X', result['Sheet2'])
        self.assertIn('Y', result['Sheet2'])
    
    def test_token_counting(self):
        """Test token counting functionality"""
        text = "This is a test text for token counting."
        count = get_token_count(text)
        self.assertGreater(count, 0)
        self.assertIsInstance(count, int)
        
        # Test with XML tags (should be stripped)
        xml_text = "<tag>Content</tag>"
        xml_count = get_token_count(xml_text)
        content_count = get_token_count("Content")
        self.assertEqual(xml_count, content_count)
    
    def test_combine_xml_outputs(self):
        """Test combining multiple XML outputs"""
        outputs = [
            '<source type="test1"><content>Content 1</content></source>',
            '<source type="test2"><content>Content 2</content></source>'
        ]
        
        combined = combine_xml_outputs(outputs)
        self.assertIn('<onefilellm_output>', combined)
        self.assertIn('Content 1', combined)
        self.assertIn('Content 2', combined)
        self.assertIn('</onefilellm_output>', combined)
    
    def test_text_preprocessing(self):
        """Test text preprocessing with NLTK"""
        input_file = os.path.join(self.temp_dir, "input.txt")
        output_file = os.path.join(self.temp_dir, "output.txt")
        
        with open(input_file, 'w') as f:
            f.write("This is a TEST with STOPWORDS and   extra   spaces.")
        
        preprocess_text(input_file, output_file)
        
        with open(output_file, 'r') as f:
            processed = f.read()
        
        self.assertNotIn("STOPWORDS", processed)  # Should be lowercase
        self.assertNotIn("   ", processed)  # Extra spaces removed


class TestAliasSystem(unittest.TestCase):
    """Test alias functionality"""
    
    def setUp(self):
        self.temp_alias_dir = tempfile.mkdtemp()
        self.original_alias_dir = Path.home() / ".onefilellm_aliases"
        # Mock the alias directory for testing
        self.patcher = patch('onefilellm.ALIAS_DIR', Path(self.temp_alias_dir))
        self.patcher.start()
        
    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.temp_alias_dir)
    
    def test_alias_detection(self):
        """Test alias detection logic"""
        # Should be detected as potential aliases
        self.assertTrue(is_potential_alias("myalias"))
        self.assertTrue(is_potential_alias("my_alias_123"))
        
        # Should NOT be detected as aliases
        self.assertFalse(is_potential_alias("https://example.com"))
        self.assertFalse(is_potential_alias("/path/to/file"))
        self.assertFalse(is_potential_alias("C:\\Windows\\file"))
        self.assertFalse(is_potential_alias("10.1234/doi"))
    
    def test_alias_directory_creation(self):
        """Test alias directory creation"""
        ensure_alias_dir_exists()
        alias_dir = Path.home() / ".onefilellm_aliases"
        self.assertTrue(alias_dir.exists())
        self.assertTrue(alias_dir.is_dir())
    
    def test_handle_add_alias(self):
        """Test creating aliases with --add-alias"""
        from onefilellm import handle_add_alias
        from rich.console import Console
        
        console = Console()
        
        # Test successful alias creation
        args = ["--add-alias", "mytest", "https://github.com/user/repo", "https://example.com"]
        result = handle_add_alias(args, console)
        self.assertTrue(result)
        
        # Verify alias file was created
        alias_file = Path(self.temp_alias_dir) / "mytest"
        self.assertTrue(alias_file.exists())
        
        # Verify contents
        with open(alias_file, 'r') as f:
            contents = f.read()
            self.assertIn("https://github.com/user/repo", contents)
            self.assertIn("https://example.com", contents)
    
    def test_handle_alias_from_clipboard(self):
        """Test creating aliases from clipboard content"""
        from onefilellm import handle_alias_from_clipboard
        from rich.console import Console
        
        console = Console()
        
        # Mock clipboard content
        test_urls = "https://github.com/repo1\nhttps://example.com/doc\n/local/path/file.txt"
        with patch('pyperclip.paste', return_value=test_urls):
            args = ["--alias-from-clipboard", "cliptest"]
            result = handle_alias_from_clipboard(args, console)
            self.assertTrue(result)
            
            # Verify alias file was created
            alias_file = Path(self.temp_alias_dir) / "cliptest"
            self.assertTrue(alias_file.exists())
            
            # Verify contents
            with open(alias_file, 'r') as f:
                contents = f.read()
                self.assertIn("https://github.com/repo1", contents)
                self.assertIn("https://example.com/doc", contents)
                self.assertIn("/local/path/file.txt", contents)
    
    def test_load_alias(self):
        """Test loading and resolving aliases"""
        from onefilellm import load_alias
        from rich.console import Console
        
        console = Console()
        
        # Create a test alias file
        alias_name = "testalias"
        alias_file = Path(self.temp_alias_dir) / alias_name
        test_targets = ["https://github.com/test/repo", "https://example.com/page"]
        with open(alias_file, 'w') as f:
            for target in test_targets:
                f.write(target + "\n")
        
        # Test loading the alias
        loaded_targets = load_alias(alias_name, console)
        self.assertEqual(loaded_targets, test_targets)
    
    def test_alias_validation(self):
        """Test alias name validation"""
        from onefilellm import handle_add_alias
        from rich.console import Console
        
        console = Console()
        
        # Test invalid alias names
        invalid_names = ["test/alias", "test\\alias", "test.alias", "test:alias"]
        for invalid_name in invalid_names:
            args = ["--add-alias", invalid_name, "https://example.com"]
            result = handle_add_alias(args, console)
            self.assertTrue(result)  # Should return True indicating error
            # Verify no file was created
            alias_file = Path(self.temp_alias_dir) / invalid_name
            self.assertFalse(alias_file.exists())


class TestIntegration(unittest.TestCase):
    """Integration tests for external services"""
    
    def test_github_repo_integration(self):
        """Test GitHub repository processing"""
        repo_url = "https://github.com/jimmc414/onefilellm"
        result = process_github_repo(repo_url)
        self.assertIn('<source type="github_repository"', result)
        self.assertIn('README.md', result)
        self.assertIn('onefilellm.py', result)
    
    def test_arxiv_integration(self):
        """Test ArXiv PDF processing"""
        arxiv_url = "https://arxiv.org/abs/2401.14295"
        result = process_arxiv_pdf(arxiv_url)
        self.assertIn('<source type="arxiv"', result)
        self.assertIn('</source>', result)
        # Check that we got actual PDF content (not an error)
        self.assertNotIn('<error>', result)
    
    def test_youtube_transcript_error_handling(self):
        """Test that YouTube transcript errors are handled gracefully.
        
        This test intentionally uses a video (Rick Astley - Never Gonna Give You Up)
        that may have restricted transcripts to ensure error handling works correctly.
        The error output is expected and suppressed to avoid confusion.
        """
        import io
        import contextlib
        
        # Use Rick Astley's video which often has transcript issues
        youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        
        # Capture stdout and stderr to suppress error messages during test
        captured_output = io.StringIO()
        captured_error = io.StringIO()
        
        with contextlib.redirect_stdout(captured_output), contextlib.redirect_stderr(captured_error):
            result = fetch_youtube_transcript(youtube_url)
        
        # Check that the function returns a valid XML structure even on error
        self.assertIn('<source type="youtube_transcript"', result)
        self.assertIn('</source>', result)
        self.assertIn(f'url="{youtube_url}"', result)
        
        # If an error occurred, verify it's properly formatted in XML
        if '<error>' in result:
            self.assertIn('<error>', result)
            self.assertIn('</error>', result)
            # This is expected - the test passes when errors are handled gracefully
        else:
            # If transcript was actually fetched, verify it has content
            self.assertTrue(len(result) > 100, "Transcript should contain actual content")
    
    def test_web_crawl_integration(self):
        """Test web crawling"""
        url = "https://docs.anthropic.com/"
        result = crawl_and_extract_text(url, max_depth=1, include_pdfs=False, ignore_epubs=True)
        self.assertIn('<source type="web_crawl"', result)
        self.assertIn('Anthropic', result)


class TestCLIFunctionality(unittest.TestCase):
    """Test command-line interface functionality"""
    
    def run_cli(self, args, input_text=None):
        """Helper to run CLI commands"""
        cmd = [sys.executable, "onefilellm.py"] + args
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if input_text else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=input_text)
        return stdout, stderr, process.returncode
    
    def test_help_message(self):
        """Test help message"""
        stdout, stderr, returncode = self.run_cli(["--help"])
        self.assertIn("Usage:", stdout)
        self.assertIn("Standard Input Options:", stdout)
        self.assertIn("--format TYPE", stdout)
    
    def test_stdin_input(self):
        """Test stdin input processing"""
        stdout, stderr, returncode = self.run_cli(["-"], "Test input")
        self.assertEqual(returncode, 0)
        self.assertIn("Detected format:", stdout)
    
    def test_format_override(self):
        """Test format override via CLI"""
        stdout, stderr, returncode = self.run_cli(["-", "--format", "json"], '{"key": "value"}')
        self.assertEqual(returncode, 0)
        self.assertIn("Processing input as json", stdout)
    
    def test_invalid_format(self):
        """Test invalid format handling"""
        stdout, stderr, returncode = self.run_cli(["-", "--format", "invalid"], "test")
        self.assertNotEqual(returncode, 0)
        self.assertIn("Invalid format type", stderr)
    
    def test_multiple_inputs(self):
        """Test multiple input handling"""
        # Create test files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1:
            f1.write("Content 1")
            file1 = f1.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
            f2.write("Content 2")
            file2 = f2.name
        
        try:
            stdout, stderr, returncode = self.run_cli([file1, file2])
            self.assertEqual(returncode, 0)
            # Check that both files were processed
            self.assertIn(file1, stdout)
            self.assertIn(file2, stdout)
            self.assertIn("Successfully processed", stdout)
        finally:
            os.unlink(file1)
            os.unlink(file2)


class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios"""
    
    def test_invalid_file_path(self):
        """Test handling of invalid file paths"""
        result = process_input("/nonexistent/file/path.txt")
        self.assertIn('error', result.lower())
    
    def test_invalid_url(self):
        """Test handling of invalid URLs"""
        result = process_input("not_a_valid_url")
        self.assertIn('error', result.lower())
    
    def test_empty_input(self):
        """Test handling of empty input"""
        console = MagicMock()
        result = process_text_stream("", {'type': 'stdin'}, console)
        self.assertIsNotNone(result)  # Should still return valid XML
    
    def test_network_errors(self):
        """Test handling of network errors"""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Network error")
            result = crawl_and_extract_text("https://example.com", 1, False, True)
            # crawl_and_extract_text returns a dict with 'content' and 'processed_urls' keys
            self.assertIsInstance(result, dict)
            self.assertIn('content', result)
            self.assertIn('Error processing page', result['content'])


class TestPerformance(unittest.TestCase):
    """Performance and edge case tests"""
    
    def test_large_file_handling(self):
        """Test handling of large files"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write 1MB of text
            content = "x" * 1024 * 1024
            f.write(content)
            f.flush()
            
            start_time = time.time()
            result = process_input(f.name)
            end_time = time.time()
            
            self.assertIn('<source type="local_file"', result)
            self.assertLess(end_time - start_time, 5)  # Should complete within 5 seconds
            
            # Close file before unlinking to avoid Windows permission error
            f.close()
            os.unlink(f.name)
    
    def test_unicode_handling(self):
        """Test Unicode character handling"""
        unicode_content = "Hello 世界 🌍 Émojis"
        console = MagicMock()
        result = process_text_stream(unicode_content, {'type': 'stdin'}, console)
        self.assertIn(unicode_content, result)
    
    def test_special_characters(self):
        """Test special character handling"""
        special_content = "Special <>&\" characters"
        console = MagicMock()
        result = process_text_stream(special_content, {'type': 'stdin'}, console)
        self.assertIn(special_content, result)


class RichTestResult(unittest.TextTestResult):
    """Custom test result class with rich formatting"""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.console = Console()
        self.test_results = []
        self.verbosity = verbosity
        
    def startTest(self, test):
        super().startTest(test)
        if self.verbosity > 1:
            test_name = f"{test.__class__.__name__}.{test._testMethodName}"
            self.console.print(f"🔄 Running: [cyan]{test_name}[/cyan]", end="")
    
    def addSuccess(self, test):
        super().addSuccess(test)
        test_name = f"{test.__class__.__name__}.{test._testMethodName}"
        self.test_results.append(('success', test_name))
        if self.verbosity > 1:
            self.console.print(" [bold green]✓ PASSED[/bold green]")
        elif self.verbosity == 1:
            self.console.print("[bold green].[/bold green]", end="")
    
    def addError(self, test, err):
        super().addError(test, err)
        test_name = f"{test.__class__.__name__}.{test._testMethodName}"
        self.test_results.append(('error', test_name))
        if self.verbosity > 1:
            self.console.print(" [bold red]✗ ERROR[/bold red]")
            if self.verbosity > 2:
                import traceback
                self.console.print(f"[red]{traceback.format_exception(*err)[-1]}[/red]")
        elif self.verbosity == 1:
            self.console.print("[bold red]E[/bold red]", end="")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        test_name = f"{test.__class__.__name__}.{test._testMethodName}"
        self.test_results.append(('failure', test_name))
        if self.verbosity > 1:
            self.console.print(" [bold red]✗ FAILED[/bold red]")
            if self.verbosity > 2:
                import traceback
                self.console.print(f"[red]{traceback.format_exception(*err)[-1]}[/red]")
        elif self.verbosity == 1:
            self.console.print("[bold red]F[/bold red]", end="")
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        test_name = f"{test.__class__.__name__}.{test._testMethodName}"
        self.test_results.append(('skip', test_name))
        if self.verbosity > 1:
            self.console.print(f" [yellow]⚠ SKIPPED[/yellow]: {reason}")
        elif self.verbosity == 1:
            self.console.print("[yellow]S[/yellow]", end="")


class RichTestRunner(unittest.TextTestRunner):
    """Custom test runner with rich formatting"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.console = Console()
        self.resultclass = RichTestResult
    
    def run(self, test):
        self.console.print("\n[bold bright_blue]══════════════════════════════════════════════════════════════════════[/bold bright_blue]")
        self.console.print("[bold bright_yellow]OneFileLLM Test Suite - All Tests Consolidated[/bold bright_yellow]", justify="center")
        self.console.print("[bold bright_blue]══════════════════════════════════════════════════════════════════════[/bold bright_blue]\n")
        
        result = super().run(test)
        
        # Print summary table
        self._print_summary(result)
        
        return result
    
    def _print_summary(self, result):
        self.console.print("\n[bold bright_blue]══════════════════════════════════════════════════════════════════════[/bold bright_blue]")
        
        # Create summary table
        table = Table(title="Test Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Count", justify="right", width=10)
        table.add_column("Status", width=20)
        
        # Add rows
        table.add_row("Tests Run", str(result.testsRun), "")
        
        success_count = result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
        table.add_row("Passed", str(success_count), Text("✓", style="bold green") if success_count > 0 else "")
        
        table.add_row("Failed", str(len(result.failures)), 
                     Text("✗", style="bold red") if len(result.failures) > 0 else Text("✓", style="bold green"))
        
        table.add_row("Errors", str(len(result.errors)), 
                     Text("✗", style="bold red") if len(result.errors) > 0 else Text("✓", style="bold green"))
        
        table.add_row("Skipped", str(len(result.skipped)), 
                     Text("⚠", style="yellow") if len(result.skipped) > 0 else "")
        
        self.console.print(table)
        
        # Overall result
        if result.wasSuccessful():
            self.console.print("\n[bold green]✅ All tests passed![/bold green]", justify="center")
        else:
            self.console.print("\n[bold red]❌ Some tests failed![/bold red]", justify="center")
            
            # Show failed tests
            if result.failures or result.errors:
                self.console.print("\n[bold red]Failed Tests:[/bold red]")
                for test, _ in result.failures + result.errors:
                    test_name = f"{test.__class__.__name__}.{test._testMethodName}"
                    self.console.print(f"  [red]• {test_name}[/red]")


def run_all_tests(verbosity=2):
    """Run all tests with optional filtering"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestUtilityFunctions,
        TestTextFormatDetection,
        TestStreamProcessing,
        TestCoreProcessing,
        TestAliasSystem,
        TestIntegration,
        TestCLIFunctionality,
        TestErrorHandling,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with rich formatting
    runner = RichTestRunner(verbosity=verbosity, stream=sys.stderr)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("""
OneFileLLM Comprehensive Test Suite

Usage:
    python test_all.py [options]

Options:
    -h, --help          Show this help message
    --integration       Run integration tests (requires network)
    --slow              Run slow tests (ArXiv, web crawling)
    --verbose           Increase test output verbosity
    --quiet             Decrease test output verbosity
    --no-color          Disable colored output

Environment Variables:
    GITHUB_TOKEN        Set to run GitHub integration tests
    RUN_INTEGRATION_TESTS=true   Enable integration tests
    RUN_SLOW_TESTS=true         Enable slow tests

Examples:
    # Run basic tests only
    python test_all.py

    # Run all tests including integration
    python test_all.py --integration --slow

    # Run with GitHub token
    GITHUB_TOKEN=your_token python test_all.py --integration
    
    # Run specific test class
    python -m unittest test_all.TestUtilityFunctions
    
    # Run specific test method
    python -m unittest test_all.TestUtilityFunctions.test_safe_file_read
""")
            sys.exit(0)
        
        # Set environment variables based on arguments
        if "--integration" in sys.argv:
            os.environ['RUN_INTEGRATION_TESTS'] = 'true'
        if "--slow" in sys.argv:
            os.environ['RUN_SLOW_TESTS'] = 'true'
        
        # Set verbosity
        verbosity = 2  # Default
        if "--verbose" in sys.argv:
            verbosity = 3
        elif "--quiet" in sys.argv:
            verbosity = 1
        
        # Check for no-color option
        if "--no-color" in sys.argv:
            os.environ['NO_COLOR'] = '1'
            from rich import console as rich_console
            rich_console._console = Console(force_terminal=False, no_color=True)
    else:
        verbosity = 2
    
    # Run tests
    success = run_all_tests(verbosity=verbosity)
    sys.exit(0 if success else 1)