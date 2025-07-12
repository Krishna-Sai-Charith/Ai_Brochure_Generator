# ------------------------------ #
# 📄 Imports & Setup
# ------------------------------ #

import os
import json
import requests
import ollama
from bs4 import BeautifulSoup
from typing import List
from IPython.display import Markdown, display

# Set the model you're using (LLaMA3.2 or any other available in your environment)
MODEL = "llama3.2"

# Optional headers for polite scraping
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

# ------------------------------ #
# 🌐 Web Scraping Class
# ------------------------------ #

class Website:
    """
    Represents a scraped website with text and links.
    Removes non-essential elements like script/style/img/input.
    """

    def __init__(self, url: str):
        self.url = url
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Title and cleaned body text
        self.title = soup.title.string.strip() if soup.title else "No title found"
        if soup.body:
            for tag in soup.body(["script", "style", "img", "input"]):
                tag.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""

        # Extract non-empty <a href> links
        links = [a.get('href') for a in soup.find_all('a')]
        self.links = [link for link in links if link]

    def get_contents(self) -> str:
        """Returns the page title and cleaned content as a string."""
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n"


# ------------------------------ #
# 🔗 Link Extraction Helpers
# ------------------------------ #

LINK_SYSTEM_PROMPT = """You are provided with a list of links found on a webpage.
You are able to decide which of the links would be most relevant to include in a brochure about the company,
such as links to an About page, or a Company page, or Careers/Jobs pages.
Respond only in JSON like this:
{
  "links": [
    {"type": "about page", "url": "https://example.com/about"},
    {"type": "careers page", "url": "https://example.com/careers"}
  ]
}"""

def get_links_user_prompt(website: Website) -> str:
    """
    Generates a user prompt with all extracted links from the website,
    asking the LLM to pick brochure-relevant links.
    """
    prompt = f"Here is the list of links on the website of {website.url}. " \
             f"Please decide which of these are relevant for a brochure " \
             f"(About, Company, Careers). Ignore email/terms/privacy links.\n\n"
    prompt += "\n".join(website.links)
    return prompt


def get_relevant_links(url: str) -> List[str]:
    """
    Calls the LLM to extract brochure-relevant links from the website.
    """
    website = Website(url)

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a JSON API. Respond only with a JSON array of URLs. No explanation."},
            {"role": "user", "content": get_links_user_prompt(website)}
        ]
    )

    content = response['message']['content'].strip()
    if content.startswith("```json"):
        content = content[7:].strip("` \n")
    elif content.startswith("```"):
        content = content[3:].strip("` \n")

    try:
        parsed = json.loads(content)
        return [item.get("url") or list(item.values())[0] for item in parsed.get("links", [])]
    except Exception as e:
        print("❌ Error parsing JSON:", e)
        return []

# ------------------------------ #
# 📥 Collect Website Details
# ------------------------------ #

def get_all_website_details(url: str) -> str:
    """
    Collects and concatenates cleaned contents of the main page and brochure-relevant subpages.
    """
    result = f"Landing page:\n{Website(url).get_contents()}"
    links = get_relevant_links(url)

    if not links:
        result += "\n\nNo valid subpage links found.\n"
        return result

    for link in links:
        if not link.startswith("http"):
            link = requests.compat.urljoin(url, link)
        try:
            result += f"\n\nURL: {link}\n{Website(link).get_contents()}"
        except Exception as e:
            result += f"\n❌ Could not fetch content from {link}: {e}\n"

    return result


# ------------------------------ #
# 🧠 Brochure Prompt Builders
# ------------------------------ #

BROCHURE_SYSTEM_PROMPT = """You are an assistant that analyzes the contents of several relevant pages from a company website and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown. Include details of company culture, customers and careers/jobs if available."""

def get_brochure_user_prompt(company_name: str, url: str) -> str:
    """
    Prepares the full user prompt with all gathered website content,
    truncated to 5000 characters if needed.
    """
    prompt = f"You are looking at a company called: {company_name}\n\n"
    prompt += "Here are the contents of its landing page and other relevant pages. " \
              "Use this information to build a short brochure of the company in markdown.\n\n"
    prompt += get_all_website_details(url)
    return prompt[:5000]  # Truncate to fit model limits


# ------------------------------ #
# 📤 Brochure Generation Functions
# ------------------------------ #

def create_brochure(company_name: str, url: str):
    """
    Calls the LLM to generate and display the company brochure in one go.
    """
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": BROCHURE_SYSTEM_PROMPT},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
        ]
    )
    display(Markdown(response['message']['content']))


def stream_brochure(company_name: str, url: str) -> str:
    """
    Streams and displays the brochure content live while it's being generated.
    """
    stream = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": BROCHURE_SYSTEM_PROMPT},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
        ],
        stream=True
    )

    response = ""
    display_handle = display(Markdown(""), display_id=True)

    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        response += content
        cleaned = response.replace("```", "").replace("markdown", "")
        display_handle.update(Markdown(cleaned))

    return response


# ------------------------------ #
# ✅ Example Usage
# ------------------------------ #

# Create a brochure for Hugging Face
# create_brochure("HuggingFace", "https://huggingface.co")
# OR
# stream_brochure("HuggingFace", "https://huggingface.co")
