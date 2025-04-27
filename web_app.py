from flask import Flask, request, render_template_string, send_file
import os
import sys

# Import functions from onefilellm.py.
# Ensure onefilellm.py is accessible in the same directory.
# --- Updated Imports ---
from onefilellm import (
    determine_and_process, # Use the centralized processing function
    get_token_count,
    preprocess_text,
    safe_file_read
)
# --- End Updated Imports ---
from pathlib import Path
import pyperclip
from urllib.parse import urlparse # Keep urlparse if needed elsewhere, or remove if not

app = Flask(__name__)

# Simple HTML template using inline rendering for demonstration.
template = """
<!DOCTYPE html>
<html>
<head>
    <title>1FileLLM Web Interface</title>
    <style>
    body { font-family: sans-serif; margin: 2em; }
    input[type="text"] { width: 80%; padding: 0.5em; }
    .output-container { margin-top: 2em; }
    .file-links { margin-top: 1em; }
    pre { background: #f8f8f8; padding: 1em; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <h1>1FileLLM Web Interface</h1>
    <form method="POST" action="/">
        <p>Enter a URL, path, DOI, or PMID:</p>
        <input type="text" name="input_path" required placeholder="e.g. https://github.com/jimmc414/1filellm or /path/to/local/folder"/>
        <button type="submit">Process</button>
    </form>

    {% if output %}
    <div class="output-container">
        <h2>Processed Output</h2>
        <pre>{{ output }}</pre> <!-- Output is now XML -->

        <h3>Token Counts</h3>
        <p>Uncompressed Tokens (XML): {{ uncompressed_token_count }}<br>
        Compressed Tokens (Text): {{ compressed_token_count }}</p>

        <div class="file-links">
            <a href="/download?filename=uncompressed_output.xml">Download Uncompressed XML</a> |
            <a href="/download?filename=compressed_output.txt">Download Compressed Text</a>
        </div>
    </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_path = request.form.get("input_path", "").strip()

        # Prepare filenames (Updated)
        output_file_xml = "uncompressed_output.xml" # Changed extension
        processed_file_txt = "compressed_output.txt" # Kept as txt
        # urls_list_file is handled internally by determine_and_process if needed,
        # but the web app doesn't expose it directly anymore.

        try:
            # Call the centralized processing function
            # Pass None for console and progress as they are CLI specific
            # determine_and_process returns the XML string (or an error XML string)
            final_output_xml = determine_and_process(input_path, console=None, progress=None)

            # Write the uncompressed XML output
            with open(output_file_xml, "w", encoding="utf-8") as file:
                file.write(final_output_xml)

            # Process the compressed output (reading from XML, writing to TXT)
            # Note: preprocess_text handles reading XML/text and writing text
            preprocess_text(output_file_xml, processed_file_txt)

            # Read compressed text for token count
            compressed_text = safe_file_read(processed_file_txt)
            compressed_token_count = get_token_count(compressed_text) # Count tokens in plain text

            # Get token count for the uncompressed XML (get_token_count strips tags)
            uncompressed_token_count = get_token_count(final_output_xml)

            # Copy the uncompressed XML to clipboard
            pyperclip.copy(final_output_xml)

            # Render the template with the XML output and token counts
            return render_template_string(template,
                                          output=final_output_xml, # Pass the XML string
                                          uncompressed_token_count=uncompressed_token_count,
                                          compressed_token_count=compressed_token_count)
        except Exception as e:
            # Render template with error message if determine_and_process or subsequent steps fail
            # determine_and_process should return an <error> tag on failure,
            # but catch other potential exceptions here.
            error_message = f'<error source="{input_path}">\n  <message>Web app error: {str(e)}</message>\n</error>'
            return render_template_string(template, output=error_message)

    # For GET requests
    return render_template_string(template)


@app.route("/download")
def download():
    filename = request.args.get("filename")
    # Check against expected filenames for security
    allowed_files = ["uncompressed_output.xml", "compressed_output.txt"]
    if filename in allowed_files and os.path.exists(filename):
        return send_file(filename, as_attachment=True)
    return "File not found or not accessible", 404

if __name__ == "__main__":
    # Run the app in debug mode for local development
    app.run(debug=True, host="0.0.0.0", port=5000)

