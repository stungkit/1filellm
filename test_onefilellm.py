import unittest
import os
import tempfile
import shutil
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
        # Check for some expected content
        self.assertIn("LLM", transcript) # Assuming 'LLM' appears in this specific transcript
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

if __name__ == "__main__":
    unittest.main()
