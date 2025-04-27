import unittest
import os
import tempfile
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
# Assuming onefilellm.py is in the same directory or accessible via PYTHONPATH
from onefilellm import (
    process_github_repo,
    process_arxiv_pdf,
    process_local_folder,
    fetch_youtube_transcript,
    crawl_and_extract_text,
    process_doi_or_pmid,
    process_github_pull_request,
    process_github_issue
)

class TestDataAggregation(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for any potential test artifacts if needed
        self.temp_dir = tempfile.mkdtemp()
        # You might want to create dummy files here if tests need them

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_github_repo(self):
        print("\nTesting GitHub repository processing...")
        repo_url = "https://github.com/jimmc414/onefilellm"
        repo_content = process_github_repo(repo_url)
        self.assertIsInstance(repo_content, str)
        self.assertGreater(len(repo_content), 0)
        # Check for the correct root source tag
        self.assertIn('<source type="github_repository"', repo_content)
        # Check if file tags are present (more specific check)
        self.assertIn('<file path="', repo_content)
        print("GitHub repository processing test passed.")

    def test_arxiv_pdf(self):
        print("\nTesting arXiv PDF processing...")
        arxiv_url = "https://arxiv.org/abs/2401.14295"
        arxiv_content = process_arxiv_pdf(arxiv_url)
        self.assertIsInstance(arxiv_content, str)
        self.assertGreater(len(arxiv_content), 0)
        # Corrected the expected source type
        self.assertIn('<source type="arxiv" url="https://arxiv.org/abs/2401.14295">', arxiv_content)
        # Optionally check for some content indicator
        self.assertIn("Demystifying Chains, Trees, and Graphs", arxiv_content)
        print("arXiv PDF processing test passed.")

    def test_local_folder(self):
        print("\nTesting local folder processing...")
        # Use the directory containing this test file as the test target
        local_path = os.path.dirname(os.path.abspath(__file__))
        local_content = process_local_folder(local_path)
        self.assertIsInstance(local_content, str)
        self.assertGreater(len(local_content), 0)
        # Corrected the expected source type
        self.assertIn('<source type="local_folder"', local_content)
        # Check if file tags are present
        self.assertIn('<file path="', local_content)
        print("Local folder processing test passed.")

    def test_youtube_transcript(self):
        print("\nTesting YouTube transcript fetching...")
        video_url = "https://www.youtube.com/watch?v=KZ_NlnmPQYk"
        transcript = fetch_youtube_transcript(video_url)
        self.assertIsInstance(transcript, str)
        self.assertGreater(len(transcript), 0)
        # Check for the correct source tag
        self.assertIn('<source type="youtube_transcript"', transcript)
        # Check for a common word instead of assuming "LLM" is present
        self.assertTrue("the" in transcript.lower() or "a" in transcript.lower(), "Transcript content seems empty or very unusual.")
        print("YouTube transcript fetching test passed.")

    def test_webpage_crawl(self):
        print("\nTesting webpage crawling and text extraction...")
        webpage_url = "https://llm.datasette.io/en/stable/"
        max_depth = 1
        include_pdfs = False # Keep False to speed up test
        ignore_epubs = True
        crawl_result = crawl_and_extract_text(webpage_url, max_depth, include_pdfs, ignore_epubs)
        self.assertIsInstance(crawl_result, dict)
        self.assertIn('content', crawl_result)
        self.assertIn('processed_urls', crawl_result)
        self.assertGreater(len(crawl_result['content']), 0)
        self.assertGreater(len(crawl_result['processed_urls']), 0)
        # Corrected the expected source type
        self.assertIn('<source type="web_crawl"', crawl_result['content'])
        # Check for page tags
        self.assertIn('<page url="', crawl_result['content'])
        print("Webpage crawling and text extraction test passed.")

    def test_process_doi(self):
        print("\nTesting DOI processing...")
        doi = "10.1053/j.ajkd.2017.08.002"
        doi_content = process_doi_or_pmid(doi)
        self.assertIsInstance(doi_content, str)
        self.assertGreater(len(doi_content), 0)
        # Corrected the expected source type and check identifier attribute
        self.assertIn('<source type="sci-hub" identifier="10.1053/j.ajkd.2017.08.002">', doi_content)
        # Check for some content
        self.assertIn("Oxalate Nephropathy", doi_content)
        print("DOI processing test passed.")

    def test_process_pmid(self):
        print("\nTesting PMID processing...")
        pmid = "29203127" # This PMID corresponds to the same paper as the DOI above
        pmid_content = process_doi_or_pmid(pmid)
        self.assertIsInstance(pmid_content, str)
        self.assertGreater(len(pmid_content), 0)
        # Corrected the expected source type and check identifier attribute
        self.assertIn('<source type="sci-hub" identifier="29203127">', pmid_content)
         # Check for some content
        self.assertIn("Oxalate Nephropathy", pmid_content)
        print("PMID processing test passed.")

    def test_process_github_pull_request(self):
        print("\nTesting GitHub pull request processing...")
        pull_request_url = "https://github.com/dear-github/dear-github/pull/102"
        pull_request_content = process_github_pull_request(pull_request_url)
        self.assertIsInstance(pull_request_content, str)
        self.assertGreater(len(pull_request_content), 0)
        # Check for the correct source tag
        self.assertIn('<source type="github_pull_request"', pull_request_content)
        # Check for specific structural elements instead of <pull_request_info>
        self.assertIn('<title>', pull_request_content)
        self.assertIn('<description>', pull_request_content)
        self.assertIn('<details>', pull_request_content)
        self.assertIn('<diff>', pull_request_content)
        # Check for the embedded repository content tag
        self.assertIn('<source type="github_repository"', pull_request_content)
        print("GitHub pull request processing test passed.")

    def test_process_github_issue(self):
        print("\nTesting GitHub issue processing...")
        issue_url = "https://github.com/isaacs/github/issues/1191"
        issue_content = process_github_issue(issue_url)
        self.assertIsInstance(issue_content, str)
        self.assertGreater(len(issue_content), 0)
         # Check for the correct source tag
        self.assertIn('<source type="github_issue"', issue_content)
        # Check for specific structural elements instead of <issue_info>
        self.assertIn('<title>', issue_content)
        self.assertIn('<description>', issue_content)
        self.assertIn('<details>', issue_content)
        # Check for the embedded repository content tag
        self.assertIn('<source type="github_repository"', issue_content)
        print("GitHub issue processing test passed.")


# --- New Test Class for Main Functionality ---

class TestMainFunctionality(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for output files
        self.temp_dir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.temp_dir, "output_aggregate.xml")
        self.compressed_file = os.path.join(self.temp_dir, "compressed_aggregate.txt")
        self.urls_file = os.path.join(self.temp_dir, "processed_urls.txt")
        # Store original CWD and change to temp dir for predictable output paths
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        # Create a dummy local file for testing local file input
        self.dummy_file_path = os.path.join(self.temp_dir, "dummy_test_file.txt")
        with open(self.dummy_file_path, "w") as f:
            f.write("This is a dummy test file.")
        # Create a dummy local folder
        self.dummy_folder_path = os.path.join(self.temp_dir, "dummy_test_folder")
        os.makedirs(self.dummy_folder_path, exist_ok=True)
        with open(os.path.join(self.dummy_folder_path, "file_in_folder.txt"), "w") as f:
            f.write("Content inside folder.")


    def tearDown(self):
        # Change back to original CWD
        os.chdir(self.original_cwd)
        # Clean up the temporary directory and its contents
        shutil.rmtree(self.temp_dir)

    def run_script(self, args):
        """Helper method to run the onefilellm.py script."""
        # Construct the command
        script_path = os.path.join(self.original_cwd, "onefilellm.py") # Assuming script is in original CWD
        command = [sys.executable, script_path] + args
        print(f"\nRunning command: {' '.join(command)}")
        # Run the command from the temporary directory context
        result = subprocess.run(command, capture_output=True, text=True, cwd=self.temp_dir)
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        return result

    def test_main_single_local_file_input(self):
        print("\nTesting main() with single local file input...")
        result = self.run_script([self.dummy_file_path])
        self.assertEqual(result.returncode, 0, f"Script failed with stderr: {result.stderr}")
        self.assertTrue(os.path.exists(self.output_file))

        # Verify XML structure
        tree = ET.parse(self.output_file)
        root = tree.getroot()
        self.assertEqual(root.tag, "onefilellm_aggregate")
        source_tags = root.findall('source[@type="local_file"]')
        self.assertEqual(len(source_tags), 1)
        file_tags = source_tags[0].findall('file')
        self.assertEqual(len(file_tags), 1)
        self.assertIn("dummy_test_file.txt", file_tags[0].get("path"))
        self.assertIn("This is a dummy test file.", file_tags[0].text)
        print("Single local file input test passed.")

    def test_main_multiple_inputs_success(self):
        print("\nTesting main() with multiple successful inputs (local file, local folder)...")
        # Using local inputs to avoid network dependency in basic tests
        result = self.run_script([self.dummy_file_path, self.dummy_folder_path])
        self.assertEqual(result.returncode, 0, f"Script failed with stderr: {result.stderr}")
        self.assertTrue(os.path.exists(self.output_file))

        # Verify XML structure
        tree = ET.parse(self.output_file)
        root = tree.getroot()
        self.assertEqual(root.tag, "onefilellm_aggregate")
        # Check for one local_file source and one local_folder source
        local_file_sources = root.findall('source[@type="local_file"]')
        local_folder_sources = root.findall('source[@type="local_folder"]')
        self.assertEqual(len(local_file_sources), 1)
        self.assertEqual(len(local_folder_sources), 1)
        # Check content within folder source
        folder_file_tags = local_folder_sources[0].findall('file')
        self.assertEqual(len(folder_file_tags), 1)
        self.assertIn("file_in_folder.txt", folder_file_tags[0].get("path"))
        self.assertIn("Content inside folder.", folder_file_tags[0].text)
        print("Multiple successful inputs test passed.")

    def test_main_input_error_handling(self):
        print("\nTesting main() with mixed valid and invalid inputs...")
        invalid_path = os.path.join(self.temp_dir, "non_existent_file.xyz")
        result = self.run_script([self.dummy_file_path, invalid_path])
        # Script should still exit successfully (return code 0) but contain an error tag
        self.assertEqual(result.returncode, 0, f"Script failed with stderr: {result.stderr}")
        self.assertTrue(os.path.exists(self.output_file))

        # Verify XML structure
        tree = ET.parse(self.output_file)
        root = tree.getroot()
        self.assertEqual(root.tag, "onefilellm_aggregate")
        source_tags = root.findall('source[@type="local_file"]')
        error_tags = root.findall('error')
        self.assertEqual(len(source_tags), 1, "Should have one successful source tag")
        self.assertEqual(len(error_tags), 1, "Should have one error tag")
        self.assertEqual(error_tags[0].get("source"), invalid_path)
        self.assertIn("Input path or URL type not recognized", error_tags[0].find('message').text)
        print("Mixed valid/invalid input test passed.")

    def test_main_no_arguments(self):
        print("\nTesting main() with no arguments...")
        result = self.run_script([])
        # Should exit gracefully (code 0) and print help text
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage: onefilellm.py [-h] INPUT [INPUT ...]", result.stdout) # Check for usage string
        self.assertIn("Processes multiple inputs", result.stdout) # Check for description
        # Output file should NOT be created
        self.assertFalse(os.path.exists(self.output_file))
        print("No arguments test passed.")

    # Optional: Add more tests using real network URLs, but mark them appropriately
    # as they might be slow or fail due to network issues.
    # Example:
    # @unittest.skipIf("CI" in os.environ, "Skipping network tests in CI")
    # def test_main_network_inputs(self):
    #     print("\nTesting main() with network inputs (GitHub repo)...")
    #     repo_url = "https://github.com/jimmc414/onefilellm" # A relatively small repo
    #     result = self.run_script([repo_url])
    #     self.assertEqual(result.returncode, 0, f"Script failed with stderr: {result.stderr}")
    #     self.assertTrue(os.path.exists(self.output_file))
    #     tree = ET.parse(self.output_file)
    #     root = tree.getroot()
    #     self.assertEqual(root.tag, "onefilellm_aggregate")
    #     source_tags = root.findall('source[@type="github_repository"]')
    #     self.assertEqual(len(source_tags), 1)
    #     self.assertGreater(len(source_tags[0].findall('file')), 0) # Check for some files
    #     print("Network input test passed.")


if __name__ == "__main__":
    unittest.main()
