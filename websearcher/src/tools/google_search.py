





import os
import json
import requests
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
import time
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
import re
import string
from typing import Optional, Tuple, List, Dict, Union
import aiohttp
import asyncio
import ssl

from ..utils.download_utils import download_to_tempfile


_DOWNLOAD_DEFAULTS = {
    "large_file_threshold": 20 * 1024 * 1024,
    "segment_size": 5 * 1024 * 1024,
    "max_retries": 3
}


def set_download_defaults(
    large_file_threshold_mb: Optional[int] = None,
    segment_size_mb: Optional[int] = None,
    max_retries: Optional[int] = None
) -> None:
    if isinstance(large_file_threshold_mb, int) and large_file_threshold_mb > 0:
        _DOWNLOAD_DEFAULTS["large_file_threshold"] = large_file_threshold_mb * 1024 * 1024
    if isinstance(segment_size_mb, int) and segment_size_mb > 0:
        _DOWNLOAD_DEFAULTS["segment_size"] = segment_size_mb * 1024 * 1024
    if isinstance(max_retries, int) and max_retries > 0:
        _DOWNLOAD_DEFAULTS["max_retries"] = max_retries

          
try:
    from config.proxy_config import get_proxy_config, should_use_proxy
    PROXY_CONFIG_AVAILABLE = True
except ImportError:
    PROXY_CONFIG_AVAILABLE = False
    def get_proxy_config():
        return None
    def should_use_proxy(url):
        return False

                   
_LINK_PATTERN = re.compile(r"\(https?:.*?\)|\[https?:.*?\]")
_WHITESPACE_PATTERN = re.compile(r'\s+')

                                  
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
    'Connection': 'keep-alive',
}

          
def _get_proxy_config():

    try:
        import sys
        import os
                    
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from config.proxy_config import get_proxy_config, should_use_proxy
        return get_proxy_config, should_use_proxy
    except ImportError:
        return None, None

_proxy_config_func, _should_use_proxy_func = _get_proxy_config()

                    
_session = None


def _get_session():

    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(HEADERS)
    return _session


                           
ERROR_INDICATORS = frozenset([
    'limit exceeded', 'error fetching', 'account balance not enough',
    'invalid bearer token', 'http error occurred', 'connection error',
    'request timed out', 'unexpected error', 'please turn on javascript',
    'enable javascript', 'port=443', 'please enable cookies',
    'access denied', '403 forbidden', '404 not found', 'captcha',
])


def has_error_indicators(content: str) -> bool:

    if not content or len(content) < 50:
        return True

    content_lower = content.lower()
    word_count = len(content.split())

    if word_count < 64:
        for indicator in ERROR_INDICATORS:
            if indicator in content_lower:
                return True

    return False


def _is_wikipedia_url(url: str) -> bool:
    lower = url.lower()
    return "wikipedia.org/wiki/" in lower or "wikipedia.org/w/" in lower


def _get_wikipedia_fallbacks(url: str) -> List[str]:
    if "action=render" in url or "printable=yes" in url:
        return []
    base = url.split("#")[0]
    if "?" in base:
        base = base.split("?")[0]
    return [f"{base}?action=render", f"{base}?printable=yes"]


def _fetch_wikipedia_fallback(url: str, proxies: Optional[dict], keep_links: bool) -> Optional[str]:
    if not _is_wikipedia_url(url):
        return None
    for fb in _get_wikipedia_fallbacks(url):
        try:
            response = requests.get(fb, headers=HEADERS, proxies=proxies, timeout=30)
            html = _decode_response(response)
            if has_error_indicators(html):
                continue
            text = _extract_text_from_html(html, fb, keep_links)
            if text and len(text) > 500:
                return text
        except Exception:
            continue
    return None


async def _fetch_wikipedia_fallback_async(url: str, session: aiohttp.ClientSession, proxy: Optional[str], keep_links: bool) -> Optional[str]:
    if not _is_wikipedia_url(url):
        return None
    for fb in _get_wikipedia_fallbacks(url):
        try:
            async with session.get(fb, proxy=proxy, timeout=aiohttp.ClientTimeout(total=30)) as response:
                content_type = response.headers.get('content-type', '').lower()
                if 'charset' in content_type:
                    charset = content_type.split('charset=')[-1].split(';')[0].strip()
                    html = await response.text(encoding=charset)
                else:
                    content = await response.read()
                    sample = content[:10000]
                    try:
                        import chardet
                        detected = chardet.detect(sample)
                        encoding = detected.get('encoding') or 'utf-8'
                    except Exception:
                        encoding = 'utf-8'
                    html = content.decode(encoding, errors='replace')
            if has_error_indicators(html):
                continue
            text = _extract_text_from_html(html, fb, keep_links)
            if text and len(text) > 500:
                return text
        except Exception:
            continue
    return None


def _try_crawl4ai_sync(url: str, snippet: Optional[str]) -> Optional[Tuple[str, str]]:
    try:
        from .crawl4ai_client import crawl_url_simple, get_crawl4ai_client
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(crawl_url_simple(url, timeout=60))
            if result[0] and not result[0].startswith("Error:") and len(result[0]) > 500:
                if snippet:
                    success, context = extract_snippet_with_context(result[0], snippet)
                    return (context, result[1]) if success else (result[0][:50000], result[1])
                return (result[0][:50000], result[1])

            print(f"⚠️ Crawl4AI failed for {url}: {result[0]}")
            try:
                print(f"[Crawl4AI] JS retry for {url}")
                client = get_crawl4ai_client()
                js_result = loop.run_until_complete(
                    client.crawl_url(url, wait_for="body", timeout=90)
                )
                if js_result.get("success"):
                    text = js_result.get("markdown") or js_result.get("content") or ""
                    if text and len(text) > 500 and not has_error_indicators(text):
                        if snippet:
                            success, context = extract_snippet_with_context(text, snippet)
                            return (context, text) if success else (text[:50000], text)
                        return (text[:50000], text)
            except Exception:
                pass
        finally:
            loop.close()
    except Exception:
        pass
    return None


async def _try_crawl4ai_async(url: str, snippet: Optional[str]) -> Optional[Tuple[str, str]]:
    try:
        from .crawl4ai_client import crawl_url_simple, get_crawl4ai_client
        result = await crawl_url_simple(url, timeout=60)
        if result[0] and not result[0].startswith("Error:") and len(result[0]) > 500:
            if snippet:
                success, context = extract_snippet_with_context(result[0], snippet)
                return (context, result[1]) if success else (result[0][:50000], result[1])
            return (result[0][:50000], result[1])

        if result[0] and result[0].startswith("Error:"):
            print(f"⚠️ Crawl4AI failed for {url}: {result[0]}")

        try:
            print(f"[Crawl4AI] JS retry for {url}")
            client = get_crawl4ai_client()
            js_result = await client.crawl_url(url, wait_for="body", timeout=90)
            if js_result.get("success"):
                text_js = js_result.get("markdown") or js_result.get("content") or ""
                if text_js and len(text_js) > 500 and not has_error_indicators(text_js):
                    if snippet:
                        success, context = extract_snippet_with_context(text_js, snippet)
                        return (context, text_js) if success else (text_js[:50000], text_js)
                    return (text_js[:50000], text_js)
        except Exception:
            pass
    except Exception as e:
        print(f"⚠️ Crawl4AI exception for {url}: {str(e)}")
    return None


def remove_punctuation(text: str) -> str:

    return text.translate(str.maketrans("", "", string.punctuation))


def f1_score(true_set: set, pred_set: set) -> float:

    intersection = len(true_set & pred_set)
    if not intersection:
        return 0.0
    precision = intersection / len(pred_set) if pred_set else 0
    recall = intersection / len(true_set) if true_set else 0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def detect_tables(content: str) -> List[Tuple[int, int]]:









    table_regions = []
    lines = content.split('\n')
    
                    
    if '<table' in content.lower():
                                           
        table_pattern = re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE)
        for match in table_pattern.finditer(content):
            start_pos = match.start()
            end_pos = match.end()
                   
            start_line = content[:start_pos].count('\n')
            end_line = content[:end_pos].count('\n')
            table_regions.append((start_line, end_line))
    
                        
    markdown_table_start = None
    for i, line in enumerate(lines):
                                        
        if '|' in line and line.count('|') >= 3:
            if markdown_table_start is None:
                markdown_table_start = i
        elif markdown_table_start is not None:
                  
            if i - markdown_table_start >= 2:                   
                table_regions.append((markdown_table_start, i - 1))
            markdown_table_start = None
    
                                    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
                                    
        fields = line.split()
        if len(fields) >= 3:        
                          
            consecutive_similar = 1
            for j in range(i + 1, min(i + 20, len(lines))):           
                next_line = lines[j].strip()
                if not next_line:
                    break
                next_fields = next_line.split()
                                          
                if abs(len(next_fields) - len(fields)) <= 1:
                    consecutive_similar += 1
                else:
                    break
            
                               
            if consecutive_similar >= 3:
                table_regions.append((i, i + consecutive_similar - 1))
                i += consecutive_similar
                continue
        
        i += 1
    
    return table_regions


def merge_candidate_regions(
    candidates: List[Tuple[float, int, int]],                                  
    table_regions: List[Tuple[int, int]],                          
    full_text: str,
    max_distance: int = 2000
) -> List[Tuple[int, int]]:






    if not candidates and not table_regions:
        return []
    
                  
    candidate_positions = []
    for f1, start_pos, end_pos in candidates:
        candidate_positions.append((start_pos, end_pos))
    
                  
    table_positions = []
    lines = full_text.split('\n')
    for start_line, end_line in table_regions:
                   
        start_pos = sum(len(lines[i]) + 1 for i in range(start_line))
        end_pos = sum(len(lines[i]) + 1 for i in range(end_line + 1))
        table_positions.append((start_pos, end_pos))
    
            
    all_regions = candidate_positions + table_positions
    if not all_regions:
        return []
    
             
    all_regions.sort(key=lambda x: x[0])
    
                
    merged = []
    current_start, current_end = all_regions[0]
    
    for start, end in all_regions[1:]:
                     
        if start - current_end <= max_distance:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    
    merged.append((current_start, current_end))
    return merged


def extract_snippet_with_context(
    full_text: str, 
    snippet: str, 
    context_chars: int = 4000,
    num_candidates: int = 3           
) -> Tuple[bool, str]:









    try:
        full_text = full_text[:150000]                                       

        snippet_lower = remove_punctuation(snippet.lower())
        snippet_words = set(snippet_lower.split())

                     
        candidates = []
        
                                   
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(full_text)
        except ImportError:
            sentences = re.split(r'(?<=[.!?])\s+', full_text)

        for sentence in sentences:
            key_sentence = remove_punctuation(sentence.lower())
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > 0.15:               
                             
                sentence_start = full_text.find(sentence)
                if sentence_start != -1:
                    sentence_end = sentence_start + len(sentence)
                    candidates.append((f1, sentence_start, sentence_end))
        
                               
        candidates.sort(reverse=True, key=lambda x: x[0])
        top_candidates = candidates[:num_candidates]
        
                  
        table_regions = detect_tables(full_text)
        
                    
        merged_regions = merge_candidate_regions(
            top_candidates, 
            table_regions, 
            full_text,
            max_distance=2000
        )
        
                       
        if merged_regions:
                                  
            min_start = min(r[0] for r in merged_regions)
            max_end = max(r[1] for r in merged_regions)
            
                     
            start_index = max(0, min_start - context_chars)
            end_index = min(len(full_text), max_end + context_chars)
            
            return True, full_text[start_index:end_index]
        elif top_candidates:
                             
            best_f1, best_start, best_end = top_candidates[0]
            start_index = max(0, best_start - context_chars)
            end_index = min(len(full_text), best_end + context_chars)
            return True, full_text[start_index:end_index]
        else:
                           
            return False, full_text[:context_chars * 3]

    except Exception as e:
        return False, f"Failed to extract context: {str(e)}"


                    
_ssl_context = None


def _get_ssl_context():

    global _ssl_context
    if _ssl_context is None:
        _ssl_context = ssl.create_default_context()
        _ssl_context.check_hostname = False
        _ssl_context.verify_mode = ssl.CERT_NONE
    return _ssl_context


class RateLimiter:

    def __init__(self, rate_limit: int, time_window: int = 60):
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.tokens = rate_limit
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            self.tokens = min(
                self.rate_limit,
                self.tokens + (time_passed * self.rate_limit / self.time_window)
            )
            self.last_update = now

            if self.tokens <= 0:
                await asyncio.sleep(1)
                self.tokens = 1

            self.tokens -= 1
            return True


jina_rate_limiter = RateLimiter(rate_limit=100)


def extract_pdf_text(url: str) -> str:

    try:
        temp_path, meta = download_to_tempfile(
            url,
            suffix=".pdf",
            timeout=30,
            max_retries=_DOWNLOAD_DEFAULTS["max_retries"],
            segment_size=_DOWNLOAD_DEFAULTS["segment_size"],
            large_file_threshold=_DOWNLOAD_DEFAULTS["large_file_threshold"]
        )
        if not meta.get("success") or not temp_path:
            return f"Error: Unable to retrieve PDF ({meta.get('error', 'download failed')})"

        with pdfplumber.open(temp_path) as pdf:
            text_parts = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n".join(text_parts)

    except requests.exceptions.Timeout:
        return "Error: PDF request timed out"
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"
    finally:
        try:
            if 'temp_path' in locals() and temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


async def extract_pdf_text_async(url: str, session: aiohttp.ClientSession) -> str:

    try:
        temp_path, meta = await asyncio.to_thread(
            download_to_tempfile,
            url,
            ".pdf",
            30,
            _DOWNLOAD_DEFAULTS["max_retries"],
            _DOWNLOAD_DEFAULTS["segment_size"],
            _DOWNLOAD_DEFAULTS["large_file_threshold"]
        )
        if not meta.get("success") or not temp_path:
            return f"Error: Unable to retrieve PDF ({meta.get('error', 'download failed')})"

        with pdfplumber.open(temp_path) as pdf:
            text_parts = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n".join(text_parts)

    except asyncio.TimeoutError:
        return "Error: PDF request timed out"
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"
    finally:
        try:
            if 'temp_path' in locals() and temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


def _decode_response(response) -> str:

    content_type = response.headers.get('content-type', '').lower()
    if 'charset' in content_type:
        charset = content_type.split('charset=')[-1].split(';')[0].strip()
        return response.content.decode(charset, errors='replace')

    content = response.content
    sample = content[:10000]                         
    try:
        import chardet
        detected = chardet.detect(sample)
        encoding = detected.get('encoding') or 'utf-8'
    except Exception:
        encoding = 'utf-8'

    return content.decode(encoding, errors='replace')


def _extract_text_from_html(html: str, url: str = '', keep_links: bool = False) -> str:

    try:
        soup = BeautifulSoup(html, 'lxml')
    except Exception:
        soup = BeautifulSoup(html, 'html.parser')

                              
    for element in soup.find_all(['script', 'style', 'meta', 'link', 'noscript', 'header', 'footer', 'nav']):
        element.decompose()

    if keep_links:
        text_parts = []
        body = soup.body if soup.body else soup
        for element in body.descendants:
            if isinstance(element, str) and element.strip():
                cleaned = _WHITESPACE_PATTERN.sub(' ', element.strip())
                if cleaned:
                    text_parts.append(cleaned)
            elif getattr(element, 'name', None) == 'a' and element.get('href'):
                href = element.get('href', '')
                link_text = element.get_text(strip=True)
                if href and link_text:
                    if href.startswith('/'):
                        base_url = '/'.join(url.split('/')[:3])
                        href = base_url + href
                    text_parts.append(f"[{link_text}]({href})")
        return ' '.join(text_parts)
    else:
        return soup.get_text(separator=' ', strip=True)

def extract_text_from_url(
    url: str,
    use_jina: bool = False,
    jina_api_key: Optional[str] = None,
    snippet: Optional[str] = None,
    keep_links: bool = False,
    use_webparser: bool = True,
    webparser_url: str = "http://localhost:8000",
    use_crawl4ai: bool = True        
) -> Tuple[str, str]:






    try:
                      
        proxies = None
        if should_use_proxy(url):
            proxies = get_proxy_config()
            if proxies:
                print(f"🔥 Using proxy for {url}")
        
        if use_jina and jina_api_key:
            jina_headers = {
                'Authorization': f'Bearer {jina_api_key}',
                'X-Return-Format': 'markdown',
            }
            response = requests.get(f'https://r.jina.ai/{url}', headers=jina_headers, proxies=proxies, timeout=30)
            text = response.text
            if not keep_links:
                text = _LINK_PATTERN.sub("", text)
            text = text.replace('---', '-').replace('===', '=')
            text = _WHITESPACE_PATTERN.sub(' ', text)
        elif 'pdf' in url.lower():
            text = extract_pdf_text(url)
        else:
            response = requests.get(url, headers=HEADERS, proxies=proxies, timeout=30)
            html = _decode_response(response)

            if has_error_indicators(html):
                                                                     
                if use_webparser:
                    try:
                        from .webparser_client import get_webparser_client
                        client = get_webparser_client(webparser_url)
                        if client.enabled:
                            result = client.parse_url(url, timeout=120)
                            if result.get("success"):
                                text = result.get("content", "")
                                if text and not has_error_indicators(text):
                                                                          
                                    if snippet:
                                        success, context = extract_snippet_with_context(text, snippet)
                                        return (context, text) if success else (text[:50000], text)
                                    else:
                                        return (text[:50000], text)
                    except Exception:
                        pass                     
                
                                                  
                crawl_result = _try_crawl4ai_sync(url, snippet)
                if crawl_result:
                    return crawl_result

                # Proxy fallback: retry without proxy if proxy returned error indicators
                if proxies:
                    try:
                        response = requests.get(url, headers=HEADERS, proxies=None, timeout=30)
                        html = _decode_response(response)
                        if not has_error_indicators(html):
                            text = _extract_text_from_html(html, url, keep_links)
                            if snippet:
                                success, context = extract_snippet_with_context(text, snippet)
                                return (context, text) if success else (text[:50000], text)
                            else:
                                return (text[:50000], text)
                    except Exception:
                        pass
                
                wiki_text = _fetch_wikipedia_fallback(url, proxies, keep_links)
                if wiki_text:
                    if snippet:
                        success, context = extract_snippet_with_context(wiki_text, snippet)
                        return (context, wiki_text) if success else (wiki_text[:50000], wiki_text)
                    else:
                        return (wiki_text[:50000], wiki_text)

                return "Error: Page contains error indicators", ""

            text = _extract_text_from_html(html, url, keep_links)
            if use_crawl4ai and (has_error_indicators(text) or len(text) < 800):
                crawl_result = _try_crawl4ai_sync(url, snippet)
                if crawl_result:
                    return crawl_result
            if _is_wikipedia_url(url) and len(text) < 800:
                wiki_text = _fetch_wikipedia_fallback(url, proxies, keep_links)
                if wiki_text:
                    text = wiki_text

                                 
        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            return (context, text) if success else (text[:50000], text)
        else:
            return (text[:50000], text)

    except Exception as e:
        error_msg = f"Error fetching {url}: {str(e)}"
        return error_msg, error_msg


def fetch_page_content(
    urls: List[str],
    max_workers: int = 32,
    use_jina: bool = False,
    jina_api_key: Optional[str] = None,
    snippets: Optional[Dict[str, str]] = None,
    show_progress: bool = False,
    keep_links: bool = False,
    use_webparser: bool = True,
    webparser_url: str = "http://localhost:8000",
    use_crawl4ai: bool = True        
) -> Dict[str, Tuple[str, str]]:

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                extract_text_from_url,
                url, use_jina, jina_api_key,
                snippets.get(url) if snippets else None,
                keep_links, use_webparser, webparser_url, use_crawl4ai
            ): url for url in urls
        }

        completed = concurrent.futures.as_completed(futures)
        if show_progress:
            from tqdm import tqdm
            completed = tqdm(completed, desc="Fetching URLs", total=len(urls))

        for future in completed:
            url = futures[future]
            try:
                results[url] = future.result()
            except Exception as e:
                results[url] = (f"Error fetching {url}: {e}", "")

    return results


def extract_relevant_info(search_results):

    useful_info = []
    if 'webPages' in search_results and 'value' in search_results['webPages']:
        for id, result in enumerate(search_results['webPages']['value']):
            info = {
                'id': id + 1,
                'title': result.get('name', ''),
                'link': result.get('link', ''),
                'date': result.get('date', '').split('T')[0],
                'snippet': result.get('snippet', ''),
                'context': ''
            }
            useful_info.append(info)
    return useful_info


async def extract_text_from_url_async(
    url: str,
    session: aiohttp.ClientSession,
    use_jina: bool = False,
    jina_api_key: Optional[str] = None,
    snippet: Optional[str] = None,
    keep_links: bool = False,
    use_webparser: bool = True,
    webparser_url: str = "http://localhost:8000",
    use_crawl4ai: bool = True        
) -> Tuple[str, str]:






    try:
                      
        proxy = None
        if should_use_proxy(url):
            proxy_config = get_proxy_config()
            if proxy_config:
                                   
                proxy = proxy_config.get('https') or proxy_config.get('http')
                if proxy:
                    print(f"🔥 Using proxy for {url}")
        
                                   
        if use_jina and jina_api_key:
            await jina_rate_limiter.acquire()
            jina_headers = {
                'Authorization': f'Bearer {jina_api_key}',
                'X-Return-Format': 'markdown',
            }
            async with session.get(
                f'https://r.jina.ai/{url}',
                headers=jina_headers,
                ssl=False,
                proxy=proxy,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                text = await response.text()
                if not keep_links:
                    text = _LINK_PATTERN.sub("", text)
                text = text.replace('---', '-').replace('===', '=')
                text = _WHITESPACE_PATTERN.sub(' ', text)

        elif 'pdf' in url.lower():
            text = await extract_pdf_text_async(url, session)

        else:
                                          
            if use_crawl4ai:
                crawl_result = await _try_crawl4ai_async(url, snippet)
                if crawl_result:
                    return crawl_result
            
                                       
            async with session.get(url, proxy=proxy, timeout=aiohttp.ClientTimeout(total=30)) as response:
                                
                content_type = response.headers.get('content-type', '').lower()
                if 'charset' in content_type:
                    charset = content_type.split('charset=')[-1].split(';')[0].strip()
                    html = await response.text(encoding=charset)
                else:
                    content = await response.read()
                    sample = content[:10000]
                    try:
                        import chardet
                        detected = chardet.detect(sample)
                        encoding = detected.get('encoding') or 'utf-8'
                    except Exception:
                        encoding = 'utf-8'
                    html = content.decode(encoding, errors='replace')

                text = _extract_text_from_html(html, url, keep_links)
                
                               
                if has_error_indicators(html) or len(text) < 1000:
                                          
                    if use_webparser:
                        try:
                            from .webparser_client import get_webparser_client
                            client = get_webparser_client(webparser_url)
                            if client.enabled:
                                result = client.parse_url(url, timeout=120)
                                if result.get("success"):
                                    webparser_text = result.get("content", "")
                                    if webparser_text and len(webparser_text) > len(text):
                                        text = webparser_text
                        except Exception:
                            pass
                    
                    # Proxy fallback: retry without proxy if proxy returned error indicators
                    if proxy and (has_error_indicators(html) or len(text) < 500):
                        try:
                            async with session.get(url, proxy=None, timeout=aiohttp.ClientTimeout(total=30)) as response:
                                content_type = response.headers.get('content-type', '').lower()
                                if 'charset' in content_type:
                                    charset = content_type.split('charset=')[-1].split(';')[0].strip()
                                    html_np = await response.text(encoding=charset)
                                else:
                                    content = await response.read()
                                    sample = content[:10000]
                                    try:
                                        import chardet
                                        detected = chardet.detect(sample)
                                        encoding = detected.get('encoding') or 'utf-8'
                                    except Exception:
                                        encoding = 'utf-8'
                                    html_np = content.decode(encoding, errors='replace')
                                text_np = _extract_text_from_html(html_np, url, keep_links)
                                if not has_error_indicators(html_np) and len(text_np) >= 500:
                                    text = text_np
                        except Exception:
                            pass
                    
                                 
                    if use_crawl4ai and (has_error_indicators(html) or len(text) < 800):
                        crawl_result = await _try_crawl4ai_async(url, snippet)
                        if crawl_result:
                            return crawl_result

                    if _is_wikipedia_url(url) and (has_error_indicators(html) or len(text) < 800):
                        wiki_text = await _fetch_wikipedia_fallback_async(url, session, proxy, keep_links)
                        if wiki_text:
                            text = wiki_text

                    if has_error_indicators(html) and len(text) < 500:
                        return "Error: Page contains error indicators or requires JavaScript", ""

                                
        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            return (context, text) if success else (text[:50000], text)
        else:
            return (text[:50000], text)

    except asyncio.TimeoutError:
        return f"Error: Request to {url} timed out", ""
    except aiohttp.ClientError as e:
                                 
        if not proxy and not should_use_proxy(url):
            print(f"⚠️ Connection failed, retrying with proxy: {url}")
            try:
                proxy_config = get_proxy_config()
                if proxy_config:
                    proxy = proxy_config.get('https') or proxy_config.get('http')
                    if proxy:
                                   
                        async with session.get(url, proxy=proxy, timeout=aiohttp.ClientTimeout(total=30)) as response:
                            content_type = response.headers.get('content-type', '').lower()
                            if 'charset' in content_type:
                                charset = content_type.split('charset=')[-1].split(';')[0].strip()
                                html = await response.text(encoding=charset)
                            else:
                                content = await response.read()
                                sample = content[:10000]
                                try:
                                    import chardet
                                    detected = chardet.detect(sample)
                                    encoding = detected.get('encoding') or 'utf-8'
                                except Exception:
                                    encoding = 'utf-8'
                                html = content.decode(encoding, errors='replace')
                            
                            text = _extract_text_from_html(html, url, keep_links)
                            
                            if snippet:
                                success, context = extract_snippet_with_context(text, snippet)
                                return (context, text) if success else (text[:50000], text)
                            else:
                                return (text[:50000], text)
            except Exception as retry_error:
                return f"Error fetching {url} (even with proxy): {str(retry_error)}", ""
        
        return f"Error fetching {url}: {str(e)}", ""
    except Exception as e:
        return f"Error: {str(e)}", ""


async def fetch_page_content_async(
    urls: Union[str, List[str]],
    use_jina: bool = False,
    jina_api_key: Optional[str] = None,
    snippets: Optional[Dict[str, str]] = None,
    show_progress: bool = False,
    keep_links: bool = False,
    max_concurrent: int = 32,
    use_webparser: bool = True,
    webparser_url: str = "http://localhost:8000",
    use_crawl4ai: bool = True        
) -> Dict[str, Tuple[str, str]]:



    if isinstance(urls, str):
        urls = [urls]

    ssl_ctx = _get_ssl_context()
    connector = aiohttp.TCPConnector(limit=max_concurrent, ssl=ssl_ctx)
    timeout = aiohttp.ClientTimeout(total=180)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=HEADERS) as session:
        tasks = [
            extract_text_from_url_async(
                url, session, use_jina, jina_api_key,
                snippets.get(url) if snippets else None,
                keep_links, use_webparser, webparser_url, use_crawl4ai
            ) for url in urls
        ]

        if show_progress:
            from tqdm import tqdm
            results = []
            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching"):
                results.append(await task)
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)

                           
        final_results = {}
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                final_results[url] = (f"Error: {str(result)}", "")
            else:
                final_results[url] = result

        return final_results

def google_serper_search(query: str, api_key: str, timeout: int = 15, num_results: int = 20) -> List[Dict]:








    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "num": min(num_results, 100)            
    })
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }

    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=timeout)
            response.raise_for_status()
            return extract_relevant_info_serper(response.json())
        except Timeout:
            if attempt == 2:
                print(f"Serper search timed out for: {query}")
                return []
            time.sleep(1)
        except Exception as e:
            if attempt == 2:
                print(f"Serper search error: {e}")
                return []
            time.sleep(1)

    return []


def extract_relevant_info_serper(search_results: dict) -> List[Dict]:

    useful_info = []
    if 'organic' in search_results:
        for i, result in enumerate(search_results['organic']):
            url = result.get('link', '')
            try:
                from urllib.parse import urlparse
                site_name = urlparse(url).netloc
            except Exception:
                site_name = ''

            info = {
                'id': i + 1,
                'title': result.get('title', ''),
                'url': url,
                'site_name': site_name,
                'date': result.get('date', ''),
                'snippet': result.get('snippet', ''),
            }
            useful_info.append(info)
    return useful_info


class SerperQuotaExceededError(Exception):

    pass


async def google_serper_search_async(
    query: str,
    api_key: str,
    timeout: int = 30,
    num_results: int = 20                     
) -> List[Dict]:










    url = "https://google.serper.dev/search"
                             
    payload = json.dumps({
        "q": query,
        "num": min(num_results, 100)                           
    })
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }

    ssl_ctx = _get_ssl_context()
    client_timeout = aiohttp.ClientTimeout(total=timeout)

    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        for attempt in range(5):
            try:
                async with session.post(url, headers=headers, data=payload, ssl=ssl_ctx) as response:
                                                             
                    if response.status in [400, 401, 403]:
                        try:
                            error_data = await response.json()
                            error_message = str(error_data).lower()
                            
                                            
                            quota_keywords = [
                                'quota', 'limit', 'exceeded', 'insufficient', 
                                'balance', 'credit', 'usage limit'
                            ]
                                                 
                            key_error_keywords = [
                                'invalid', 'unauthorized', 'forbidden', 
                                'bad request', 'api key', 'authentication',
                                'invalid key', 'key not found'
                            ]
                            
                                                           
                            if (any(keyword in error_message for keyword in quota_keywords) or
                                any(keyword in error_message for keyword in key_error_keywords) or
                                response.status == 400):                          
                                raise SerperQuotaExceededError(
                                    f"Serper API error (status {response.status}): {error_data}. Switching to backup API key..."
                                )
                        except (json.JSONDecodeError, aiohttp.ContentTypeError):
                                                  
                            if response.status in [401, 403]:
                                                              
                                raise SerperQuotaExceededError(
                                    f"Serper API authentication error (status {response.status}). Switching to backup API key..."
                                )
                            elif response.status == 400:
                                                     
                                raise SerperQuotaExceededError(
                                    f"Serper API bad request (status 400), possibly invalid key. Switching to backup API key..."
                                )
                    
                    response.raise_for_status()
                    search_results = await response.json()
                    return extract_relevant_info_serper(search_results)

            except SerperQuotaExceededError:
                                
                raise

            except asyncio.TimeoutError:
                if attempt == 4:
                    print(f"Serper search timed out after 5 attempts: {query}")
                    return []
                await asyncio.sleep(1 * (attempt + 1))

            except aiohttp.ClientError as e:
                                                                    
                                                         
                if hasattr(e, 'status') and e.status in [400, 401, 403]:
                                                   
                    raise SerperQuotaExceededError(
                        f"Serper API client error (status {e.status}): {str(e)}. Switching to backup API key..."
                    )
                
                if attempt == 4:
                    print(f"Serper search error after 5 attempts: {e}")
                    return []
                await asyncio.sleep(1 * (attempt + 1))

            except Exception as e:
                if attempt == 4:
                    print(f"Unexpected error in Serper search: {e}")
                    return []
                await asyncio.sleep(1)

    return []


def get_openai_function_web_search() -> dict:

    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using Google. Returns titles, URLs, and snippets. Use specific queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string. Be specific for better results."
                    }
                },
                "required": ["query"]
            }
        }
    }


def get_openai_function_browse_pages() -> dict:

    return {
        "type": "function",
        "function": {
            "name": "browse_pages",
            "description": "Fetch content from web pages or PDFs. Returns extracted text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "URLs to fetch content from"
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional query for relevance-based extraction"
                    },
                    "force_refresh": {
                        "type": "boolean",
                        "description": "If true, bypass cache and refetch URLs"
                    }
                },
                "required": ["urls"]
            }
        }
    }
