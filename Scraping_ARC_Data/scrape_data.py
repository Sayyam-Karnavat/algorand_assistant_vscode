import requests
from bs4 import BeautifulSoup
import time
import os
import json
from urllib.parse import urljoin
import re

# Headers to mimic a browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
}

def clean_text(text):
    """Clean extracted text by removing excessive whitespace and newlines."""
    return re.sub(r'\s+', ' ', text.strip())

def scrape_page(url, data_list):
    """Scrape content from a single page, append to data_list, and return the next URL."""
    try:
        # Send GET request
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the main content div
        content_div = soup.find('div', class_='sl-markdown-content')
        if not content_div:
            print(f"No content found at {url}")
            return None
        
        # Extract all text from the content div
        content_text = clean_text(content_div.get_text(separator=' '))
        
        # Get page title for reference
        title = soup.find('h1').get_text(strip=True) if soup.find('h1') else 'Untitled'
        
        # Append to data list
        try:
            data_list.append({"title": title, "content": content_text})
        except UnicodeEncodeError as e:
            print(f"Encoding error at {url}: {e}")
            # Fallback: Replace problematic characters
            data_list.append({
                "title": title.encode('utf-8', errors='replace').decode('utf-8'),
                "content": content_text.encode('utf-8', errors='replace').decode('utf-8')
            })
        
        print(f"Scraped: {title} ({url})")
        
        # Find the next link
        next_link = soup.find('a', rel='next')
        if next_link and 'href' in next_link.attrs:
            next_url = urljoin(url, next_link['href'])
            # Check if the next URL contains '/arc-standards/'
            if '/arc-standards/' in next_url:
                return next_url
        return None
    
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

def main():
    # Starting URL
    start_url = 'https://dev.algorand.co/arc-standards/arc-0000/'
    output_file = 'arc_standards.json'
    
    # Initialize data list
    data_list = []
    
    current_url = start_url
    while current_url:
        # Scrape current page and append to data_list
        next_url = scrape_page(current_url, data_list)
        
        # Move to next URL if available
        current_url = next_url
        
        # Add delay to avoid overwhelming the server
        if current_url:
            time.sleep(1)  # 1-second delay between requests
    
    # Save data to JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        print(f"Data saved to {output_file}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")

if __name__ == '__main__':
    main()