#!/usr/bin/env python3
"""
Test suite for the text stream input processing features in onefilellm.py.
This tests the new functionality:
1. Reading from stdin
2. Reading from clipboard
3. Format detection and parsing
4. Format override
5. Processing and XML output
"""

import unittest
import sys
import json
import io
import pyperclip
from unittest.mock import patch, MagicMock
from rich.console import Console

# Import the functions we want to test
from onefilellm import (
    read_from_clipboard,
    read_from_stdin,
    detect_text_format,
    parse_as_plaintext,
    parse_as_markdown,
    parse_as_json,
    parse_as_html,
    parse_as_yaml,
    parse_as_doculing,
    parse_as_markitdown,
    get_parser_for_format,
    process_text_stream,
)

class TestStreamProcessing(unittest.TestCase):
    """Test the text stream processing functionality."""

    def setUp(self):
        """Set up test environment."""
        self.console = Console(file=io.StringIO())  # Create a console that writes to StringIO
        self.test_text = "This is a simple test."
        self.test_markdown = "# Heading\n\nThis is **markdown**."
        self.test_json = '{"key": "value", "array": [1, 2, 3]}'
        self.test_html = "<html><body><h1>Hello</h1><p>World</p></body></html>"
        self.test_yaml = "key: value\nlist:\n  - item1\n  - item2"
        
    def test_read_from_clipboard(self):
        """Test reading content from the clipboard."""
        # Mock pyperclip.paste to return our test text
        with patch('pyperclip.paste', return_value=self.test_text):
            result = read_from_clipboard()
            self.assertEqual(result, self.test_text)
            
        # Test empty clipboard
        with patch('pyperclip.paste', return_value=""):
            result = read_from_clipboard()
            self.assertIsNone(result)
            
        # Test PyperclipException
        with patch('pyperclip.paste', side_effect=pyperclip.PyperclipException('Test error')):
            result = read_from_clipboard()
            self.assertIsNone(result)
    
    def test_read_from_stdin(self):
        """Test reading content from stdin."""
        # Mock sys.stdin.read to return our test text
        with patch('sys.stdin.isatty', return_value=False), \
             patch('sys.stdin.read', return_value=self.test_text):
            result = read_from_stdin()
            self.assertEqual(result, self.test_text)
            
        # Test tty connection (interactive terminal)
        with patch('sys.stdin.isatty', return_value=True):
            result = read_from_stdin()
            self.assertIsNone(result)
            
        # Test empty stdin
        with patch('sys.stdin.isatty', return_value=False), \
             patch('sys.stdin.read', return_value=""):
            result = read_from_stdin()
            self.assertIsNone(result)
            
        # Test exception
        with patch('sys.stdin.isatty', return_value=False), \
             patch('sys.stdin.read', side_effect=Exception('Test error')):
            result = read_from_stdin()
            self.assertIsNone(result)
    
    def test_detect_text_format(self):
        """Test format detection for different text inputs."""
        # Test empty or whitespace-only string
        self.assertEqual(detect_text_format(""), "text")
        self.assertEqual(detect_text_format("   \n   "), "text")
        
        # Test plain text
        self.assertEqual(detect_text_format(self.test_text), "text")
        
        # Test Markdown
        self.assertEqual(detect_text_format(self.test_markdown), "markdown")
        self.assertEqual(detect_text_format("- Item 1\n- Item 2"), "markdown")
        self.assertEqual(detect_text_format("[Link](https://example.com)"), "markdown")
        
        # Test JSON
        self.assertEqual(detect_text_format(self.test_json), "json")
        self.assertEqual(detect_text_format('{"simple": true}'), "json")
        self.assertEqual(detect_text_format('[1, 2, 3]'), "json")
        
        # Test HTML
        self.assertEqual(detect_text_format(self.test_html), "html")
        self.assertEqual(detect_text_format("<div>Simple div</div>"), "html")
        
        # Test YAML (if available)
        try:
            import yaml
            if yaml:
                self.assertEqual(detect_text_format(self.test_yaml), "yaml")
                self.assertEqual(detect_text_format("simple: yaml"), "text")  # Not enough indicators
        except ImportError:
            print("YAML module not available for testing")
    
    def test_parsers(self):
        """Test the individual parser functions."""
        # Test plaintext parser (passthrough)
        self.assertEqual(parse_as_plaintext(self.test_text), self.test_text)
        
        # Test markdown parser (passthrough for V1)
        self.assertEqual(parse_as_markdown(self.test_markdown), self.test_markdown)
        
        # Test JSON parser (validates but returns original)
        self.assertEqual(parse_as_json(self.test_json), self.test_json)
        with self.assertRaises(json.JSONDecodeError):
            parse_as_json("not valid json")
            
        # Test HTML parser (extracts text)
        html_text = parse_as_html(self.test_html)
        self.assertIn("Hello", html_text)
        self.assertIn("World", html_text)
        self.assertNotIn("<", html_text)  # Tags should be stripped
        
        # Test YAML parser (if available)
        try:
            import yaml
            if yaml:
                self.assertEqual(parse_as_yaml(self.test_yaml), self.test_yaml)
                with self.assertRaises(yaml.YAMLError):
                    parse_as_yaml("not: valid: yaml")
        except ImportError:
            print("YAML module not available for testing")
    
    def test_get_parser_for_format(self):
        """Test the parser selector function."""
        self.assertEqual(get_parser_for_format("text"), parse_as_plaintext)
        self.assertEqual(get_parser_for_format("markdown"), parse_as_markdown)
        self.assertEqual(get_parser_for_format("json"), parse_as_json)
        self.assertEqual(get_parser_for_format("html"), parse_as_html)
        self.assertEqual(get_parser_for_format("yaml"), parse_as_yaml)
        self.assertEqual(get_parser_for_format("doculing"), parse_as_doculing)
        self.assertEqual(get_parser_for_format("markitdown"), parse_as_markitdown)
        
        # Test unknown format (defaults to plaintext)
        self.assertEqual(get_parser_for_format("unknown"), parse_as_plaintext)
    
    def test_process_text_stream(self):
        """Test the orchestrator function."""
        # Test with plaintext
        result = process_text_stream(self.test_text, {'type': 'stdin'}, self.console)
        self.assertIsInstance(result, str)
        self.assertIn('<source type="stdin" processed_as_format="text">', result)
        self.assertIn(f'<content>{self.test_text}</content>', result)
        
        # Test with format override
        result = process_text_stream(self.test_text, {'type': 'clipboard'}, self.console, format_override="markdown")
        self.assertIsInstance(result, str)
        self.assertIn('<source type="clipboard" processed_as_format="markdown">', result)
        self.assertIn(f'<content>{self.test_text}</content>', result)
        
        # Test with JSON
        result = process_text_stream(self.test_json, {'type': 'stdin'}, self.console)
        self.assertIsInstance(result, str)
        self.assertIn('<source type="stdin" processed_as_format="json">', result)
        self.assertIn(f'<content>{self.test_json}</content>', result)
        
        # Test with invalid JSON and format override
        result = process_text_stream("not valid json", {'type': 'stdin'}, self.console, format_override="json")
        self.assertIsNone(result)  # Should return None for invalid JSON
        
        # Test with HTML
        result = process_text_stream(self.test_html, {'type': 'clipboard'}, self.console)
        self.assertIsInstance(result, str)
        self.assertIn('<source type="clipboard" processed_as_format="html">', result)
        self.assertNotIn("<html>", result)  # HTML should be processed to extract text
        
        # Test with YAML (if available)
        try:
            import yaml
            if yaml:
                result = process_text_stream(self.test_yaml, {'type': 'stdin'}, self.console)
                self.assertIsInstance(result, str)
                self.assertIn('<source type="stdin" processed_as_format="yaml">', result)
                self.assertIn('key: value', result)
                
                # Test with invalid YAML and format override
                result = process_text_stream("not: valid: yaml", {'type': 'stdin'}, self.console, format_override="yaml")
                self.assertIsNone(result)  # Should return None for invalid YAML
        except ImportError:
            print("YAML module not available for testing")

if __name__ == "__main__":
    print("Running Text Stream Processing Tests")
    unittest.main()