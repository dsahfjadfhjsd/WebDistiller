





import asyncio
from typing import Optional, Dict, Tuple, List, Union
import re
import sys

                          
if sys.version_info < (3, 10):
    from typing import Union as UnionType
else:
    UnionType = Union

          
try:
    from config.proxy_config import get_proxy_config, should_use_proxy
    PROXY_CONFIG_AVAILABLE = True
except ImportError:
    PROXY_CONFIG_AVAILABLE = False
    def get_proxy_config():
        return None
    def should_use_proxy(url):
        return False

                                        
_crawl4ai_available = None
_AsyncWebCrawler = None
_BrowserConfig = None
_CrawlerRunConfig = None
_CacheMode = None


def _check_crawl4ai():

    global _crawl4ai_available, _AsyncWebCrawler, _BrowserConfig, _CrawlerRunConfig, _CacheMode
    
    if _crawl4ai_available is not None:
        return _crawl4ai_available
    
    try:
                                                                             
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8", errors="ignore")
        except Exception:
            pass

        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
        _AsyncWebCrawler = AsyncWebCrawler
        _BrowserConfig = BrowserConfig
        _CrawlerRunConfig = CrawlerRunConfig
        _CacheMode = CacheMode
        _crawl4ai_available = True
        return True
    except ImportError:
        _crawl4ai_available = False
        return False


class Crawl4AIClient:




    
    def __init__(self, headless: bool = True, verbose: bool = False):







        self.enabled = _check_crawl4ai()
        self.headless = headless
        self.verbose = verbose
        self._crawler = None
        
        if not self.enabled:
            if verbose:
                print("[WARN] crawl4ai not available. Install with: pip install crawl4ai")
    
    async def _get_crawler(self):

        if self._crawler is None and self.enabled:
                    
            proxy_server = None
            proxy_cfg = None
            if PROXY_CONFIG_AVAILABLE:
                proxy_config = get_proxy_config()
                if proxy_config:
                                                                      
                    proxy_server = proxy_config.get('https') or proxy_config.get('http')
                    if proxy_server:
                        proxy_cfg = {"server": proxy_server}
            
            browser_config = _BrowserConfig(
                headless=self.headless,
                verbose=self.verbose,
                proxy_config=proxy_cfg                              
            )
            self._crawler = _AsyncWebCrawler(config=browser_config)
            await self._crawler.__aenter__()
        return self._crawler
    
    async def close(self):

        if self._crawler is not None:
            try:
                await self._crawler.__aexit__(None, None, None)
            except Exception:
                pass
            self._crawler = None
    
    async def crawl_url(
        self,
        url: str,
        wait_for: Optional[str] = None,
        timeout: int = 60,
        extract_links: bool = False,
        remove_overlay: bool = True
    ) -> Dict:














        if not self.enabled:
            return {
                "success": False,
                "error": "crawl4ai not available",
                "content": "",
                "markdown": "",
                "links": []
            }
        
        try:
                          
            if should_use_proxy(url):
                print(f"[Crawl4AI] Using proxy for {url}")
            
            crawler = await self._get_crawler()
            
                                      
            config = _CrawlerRunConfig(
                cache_mode=_CacheMode.BYPASS,
                wait_for=wait_for,
                page_timeout=timeout * 1000,                           
                remove_overlay_elements=remove_overlay,
            )
            
                           
            result = await crawler.arun(url=url, config=config)
            
            if not result.success:
                return {
                    "success": False,
                    "error": result.error_message or "Crawl failed",
                    "content": "",
                    "markdown": "",
                    "links": []
                }
            
                             
            content = result.cleaned_html or result.html or ""
            markdown = result.markdown or ""
            
                              
            content = self._clean_text(content)
            markdown = self._clean_text(markdown)
            
                                        
            links = []
            if extract_links and result.links:
                links = [
                    {"text": link.get("text", ""), "url": link.get("href", "")}
                    for link in result.links.get("internal", [])
                ]
            
            return {
                "success": True,
                "content": content,
                "markdown": markdown,
                "links": links,
                "error": None
            }
            
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Timeout after {timeout}s",
                "content": "",
                "markdown": "",
                "links": []
            }
        except RuntimeError as e:
                                                      
            error_str = str(e)
            if "Timeout" in error_str or "timeout" in error_str.lower():
                return {
                    "success": False,
                    "error": f"Page navigation timeout after {timeout}s. The page may be slow or require JavaScript.",
                    "content": "",
                    "markdown": "",
                    "links": []
                }
            else:
                return {
                    "success": False,
                    "error": f"Crawl4AI runtime error: {error_str}",
                    "content": "",
                    "markdown": "",
                    "links": []
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Crawl4AI error: {str(e)}",
                "content": "",
                "markdown": "",
                "links": []
            }
    
    async def crawl_urls_batch(
        self,
        urls: List[str],
        wait_for: Optional[str] = None,
        timeout: int = 60,
        extract_links: bool = False,
        max_concurrent: int = 5
    ) -> Dict[str, Dict]:













        if not self.enabled:
            return {
                url: {
                    "success": False,
                    "error": "crawl4ai not available",
                    "content": "",
                    "markdown": "",
                    "links": []
                }
                for url in urls
            }
        
                                                  
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def crawl_with_semaphore(url: str):
            async with semaphore:
                return url, await self.crawl_url(
                    url, wait_for, timeout, extract_links
                )
        
                        
        tasks = [crawl_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
                         
        output = {}
        for result in results:
            if isinstance(result, Exception):
                continue
            url, data = result
            output[url] = data
        
        return output
    
    def _clean_text(self, text: str) -> str:

        if not text:
            return ""
        
                                     
        text = re.sub(r'\s+', ' ', text)
        
                             
        text = re.sub(r'Cookie\s+Policy|Accept\s+Cookies|Privacy\s+Policy', '', text, flags=re.IGNORECASE)
        
        return text.strip()


                        
_global_client = None


def get_crawl4ai_client(headless: bool = True, verbose: bool = False) -> Crawl4AIClient:

    global _global_client
    if _global_client is None:
        _global_client = Crawl4AIClient(headless=headless, verbose=verbose)
    return _global_client


async def crawl_url_simple(url: str, timeout: int = 60) -> Tuple[str, str]:




    client = get_crawl4ai_client()
    result = await client.crawl_url(url, timeout=timeout)
    
    if result["success"]:
                                                                   
        text = result["markdown"] or result["content"]
        return (text[:50000], text)
    else:
        error_msg = f"Error: {result['error']}"
        return (error_msg, error_msg)


async def crawl_urls_batch_simple(
    urls: List[str],
    timeout: int = 60,
    max_concurrent: int = 5
) -> Dict[str, Tuple[str, str]]:




    client = get_crawl4ai_client()
    results = await client.crawl_urls_batch(
        urls, timeout=timeout, max_concurrent=max_concurrent
    )
    
    output = {}
    for url, result in results.items():
        if result["success"]:
            text = result["markdown"] or result["content"]
            output[url] = (text[:50000], text)
        else:
            error_msg = f"Error: {result['error']}"
            output[url] = (error_msg, error_msg)
    
    return output
