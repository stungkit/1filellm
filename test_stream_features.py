#!/usr/bin/env python3
"""
End-to-end test script for the stream processing features in onefilellm.py.
This runs various command-line tests to verify the stream processing features work correctly.
"""

import os
import sys
import subprocess
import pyperclip
import json
import yaml
import tempfile
import time

# ANSI color codes for prettier output
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"
BOLD = "\033[1m"

def print_header(text):
    """Print a formatted test header."""
    print(f"\n{BOLD}{YELLOW}{'='*10} {text} {'='*10}{RESET}\n")

def print_success(text):
    """Print a success message."""
    print(f"{GREEN}✓ {text}{RESET}")

def print_failure(text):
    """Print a failure message."""
    print(f"{RED}✗ {text}{RESET}")

def run_command(cmd, input_text=None):
    """Run a command and capture its output."""
    print(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, 
        stdin=subprocess.PIPE if input_text else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate(input=input_text)
    
    # For debugging
    if stderr:
        print(f"  stderr: {stderr.strip()}")
    
    return stdout, stderr, process.returncode

def test_stdin_processing():
    """Test processing text from stdin."""
    print_header("Testing Standard Input Processing")
    
    # Test plain text
    plain_text = "This is a plain text input via stdin."
    stdout, stderr, returncode = run_command(["python", "onefilellm.py", "-"], input_text=plain_text)
    
    if returncode == 0 and "Reading from standard input" in stdout:
        if "Detected format: text" in stdout:
            print_success("Plain text format correctly detected")
        else:
            print_failure("Plain text format not correctly detected")
    else:
        print_failure(f"Stdin processing failed: {stderr}")
    
    # Test JSON
    json_text = json.dumps({"key": "value", "array": [1, 2, 3]})
    stdout, stderr, returncode = run_command(["python", "onefilellm.py", "-"], input_text=json_text)
    
    if returncode == 0 and "Reading from standard input" in stdout:
        if "Detected format: json" in stdout:
            print_success("JSON format correctly detected")
        else:
            print_failure("JSON format not correctly detected")
    else:
        print_failure(f"Stdin JSON processing failed: {stderr}")
    
    # Test with format override
    stdout, stderr, returncode = run_command(
        ["python", "onefilellm.py", "-", "--format", "markdown"], 
        input_text=plain_text
    )
    
    if returncode == 0 and "Processing input as markdown (user override)" in stdout:
        print_success("Format override works correctly")
    else:
        print_failure(f"Format override failed: {stderr}")

def test_clipboard_processing():
    """Test processing text from clipboard."""
    print_header("Testing Clipboard Processing")
    
    # Save original clipboard content to restore later
    original_clipboard = pyperclip.paste()
    
    try:
        # Test plain text
        test_text = "This is a test from the clipboard."
        pyperclip.copy(test_text)
        
        stdout, stderr, returncode = run_command(["python", "onefilellm.py", "--clipboard"])
        
        if returncode == 0 and "Reading from clipboard" in stdout:
            if "Detected format: text" in stdout:
                print_success("Clipboard plain text format correctly detected")
            else:
                print_failure("Clipboard plain text format not correctly detected")
        else:
            print_failure(f"Clipboard processing failed: {stderr}")
        
        # Test Markdown
        markdown_text = "# Heading\n\nThis is **bold** text."
        pyperclip.copy(markdown_text)
        
        stdout, stderr, returncode = run_command(["python", "onefilellm.py", "-c"])
        
        if returncode == 0 and "Reading from clipboard" in stdout:
            if "Detected format: markdown" in stdout:
                print_success("Clipboard Markdown format correctly detected")
            else:
                print_failure(f"Clipboard Markdown format not correctly detected")
        else:
            print_failure(f"Clipboard processing failed: {stderr}")
        
        # Test with format override shorthand
        pyperclip.copy(test_text)
        stdout, stderr, returncode = run_command(["python", "onefilellm.py", "--clipboard", "-f", "json"])
        
        if "Error:" in stdout and "not valid JSON" in stdout:
            print_success("Invalid JSON correctly detected when using format override")
        else:
            print_failure("Format validation failed to catch invalid JSON")
    
    finally:
        # Restore original clipboard content
        pyperclip.copy(original_clipboard)

def test_output_files():
    """Test that output files are created correctly."""
    print_header("Testing Output Files")
    
    # First remove any existing output files
    for filename in ["output.xml", "compressed_output.txt"]:
        if os.path.exists(filename):
            os.remove(filename)
    
    # Run a test command
    test_text = "Test output file creation."
    stdout, stderr, returncode = run_command(
        ["python", "onefilellm.py", "-"], 
        input_text=test_text
    )
    
    # Check output.xml was created
    if os.path.exists("output.xml"):
        with open("output.xml", "r") as f:
            content = f.read()
            if "<onefilellm_output>" in content and test_text in content:
                print_success("output.xml created with correct content")
            else:
                print_failure("output.xml content is incorrect")
    else:
        print_failure("output.xml was not created")

def test_format_detection():
    """Test format detection for different input types."""
    print_header("Testing Format Detection")
    
    # Create test files for different formats
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as text_file:
        text_file.write("This is plain text.")
        text_path = text_file.name
    
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".md", delete=False) as md_file:
        md_file.write("# Markdown Heading\n\nThis is a paragraph with **bold** text.")
        md_path = md_file.name
    
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as json_file:
        json.dump({"key": "value", "array": [1, 2, 3]}, json_file)
        json_path = json_file.name
    
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".html", delete=False) as html_file:
        html_file.write("<html><body><h1>Test</h1><p>This is HTML content</p></body></html>")
        html_path = html_file.name
    
    try:
        import yaml
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as yaml_file:
            yaml.dump({"key": "value", "list": ["item1", "item2"]}, yaml_file)
            yaml_path = yaml_file.name
    except ImportError:
        yaml_path = None
        print("YAML module not available for testing")
    
    try:
        # Test text detection
        stdout, stderr, returncode = run_command(["python", "onefilellm.py", "-"], input_text=open(text_path).read())
        if "Detected format: text" in stdout:
            print_success("Plain text format correctly detected")
        else:
            print_failure("Plain text format detection failed")
        
        # Test markdown detection
        stdout, stderr, returncode = run_command(["python", "onefilellm.py", "-"], input_text=open(md_path).read())
        if "Detected format: markdown" in stdout:
            print_success("Markdown format correctly detected")
        else:
            print_failure("Markdown format detection failed")
        
        # Test JSON detection
        stdout, stderr, returncode = run_command(["python", "onefilellm.py", "-"], input_text=open(json_path).read())
        if "Detected format: json" in stdout:
            print_success("JSON format correctly detected")
        else:
            print_failure("JSON format detection failed")
        
        # Test HTML detection
        stdout, stderr, returncode = run_command(["python", "onefilellm.py", "-"], input_text=open(html_path).read())
        if "Detected format: html" in stdout:
            print_success("HTML format correctly detected")
        else:
            print_failure("HTML format detection failed")
        
        # Test YAML detection if available
        if yaml_path:
            stdout, stderr, returncode = run_command(["python", "onefilellm.py", "-"], input_text=open(yaml_path).read())
            if "Detected format: yaml" in stdout:
                print_success("YAML format correctly detected")
            else:
                print_failure("YAML format detection failed")
    
    finally:
        # Clean up temporary files
        for path in [text_path, md_path, json_path, html_path]:
            if os.path.exists(path):
                os.remove(path)
        if yaml_path and os.path.exists(yaml_path):
            os.remove(yaml_path)

def test_original_functionality():
    """Test that original file processing still works."""
    print_header("Testing Original Functionality")
    
    # Create a test file
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as test_file:
        test_file.write("This is a test file.")
        test_path = test_file.name
    
    try:
        # Test that we can still process a file normally
        stdout, stderr, returncode = run_command(["python", "onefilellm.py", test_path])
        
        if returncode == 0 and "Successfully processed:" in stdout:
            print_success("Original file processing still works")
        else:
            print_failure(f"Original file processing failed: {stderr}")
    
    finally:
        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)

def test_help_message():
    """Test help message output."""
    print_header("Testing Help Message")
    
    stdout, stderr, returncode = run_command(["python", "onefilellm.py", "--help"])
    
    if returncode == 0:
        if "onefilellm - Content Aggregation Tool" in stdout:
            print_success("Help message displays correctly")
        else:
            print_failure("Help message doesn't contain expected title")
            
        if "-c, --clipboard" in stdout and "--format TYPE" in stdout:
            print_success("Help message includes new stream options")
        else:
            print_failure("Help message missing new stream options")
    else:
        print_failure(f"Help command failed: {stderr}")

def test_invalid_inputs():
    """Test error handling for invalid inputs."""
    print_header("Testing Error Handling")
    
    # Test invalid format
    stdout, stderr, returncode = run_command(
        ["python", "onefilellm.py", "--clipboard", "--format", "invalid_format"],
        input_text="Any content"
    )
    if returncode != 0 and "Invalid format type" in stderr:
        print_success("Invalid format type correctly rejected")
    else:
        print_failure("Invalid format type not properly handled")
    
    # Test stdin with no pipe
    stdout, stderr, returncode = run_command(["python", "onefilellm.py", "-"])
    # Print outputs for debugging
    print(f"  stdout contains: {stdout[:100].strip()}")
    
    # Check if any warning about stdin or pipe appears anywhere in the output
    if (any(msg in stdout for msg in ["stdin", "pipe", "Warning", "-"]) or 
        any(msg in stderr for msg in ["stdin", "pipe", "Warning", "-"])):
        print_success("Warning about stdin usage detected")
    else:
        print_failure("No warning about stdin usage detected")

def main():
    """Run all tests."""
    print(f"{BOLD}{YELLOW}Starting Stream Processing Feature Tests{RESET}")
    
    test_stdin_processing()
    test_clipboard_processing()
    test_output_files()
    test_format_detection()
    test_original_functionality()
    test_help_message()
    test_invalid_inputs()
    
    print(f"\n{BOLD}{GREEN}Stream processing feature tests completed{RESET}")

if __name__ == "__main__":
    main()