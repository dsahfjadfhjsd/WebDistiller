






import requests
from typing import List, Dict, Union
from urllib.parse import urljoin


class WebParserClient:








    
    def __init__(self, base_url: str = "http://localhost:8000"):






        self.base_url = base_url.rstrip('/')
        self.enabled = False                                                        
    
    def set_enabled(self, enabled: bool):

        self.enabled = enabled
    
    def parse_urls(
        self, 
        urls: List[str], 
        timeout: int = 120
    ) -> List[Dict[str, Union[str, bool]]]:














        if not self.enabled:
            return [{"success": False, "error": "WebParser client is disabled"} for _ in urls]
        
        try:
            endpoint = urljoin(self.base_url, "/parse_urls")
            response = requests.post(
                endpoint, 
                json={"urls": urls}, 
                timeout=timeout
            )
            response.raise_for_status()
            
            return response.json()["results"]
        except requests.exceptions.Timeout:
            return [{"success": False, "error": "Request timeout"} for _ in urls]
        except requests.exceptions.RequestException as e:
            return [{"success": False, "error": str(e)} for _ in urls]
        except Exception as e:
            return [{"success": False, "error": f"Unexpected error: {str(e)}"} for _ in urls]
    
    def parse_url(self, url: str, timeout: int = 120) -> Dict[str, Union[str, bool]]:










        results = self.parse_urls([url], timeout)
        return results[0] if results else {"success": False, "error": "No result"}


                                                        
_webparser_client = None


def get_webparser_client(base_url: str = "http://localhost:8000") -> WebParserClient:









    global _webparser_client
    if _webparser_client is None:
        _webparser_client = WebParserClient(base_url)
    return _webparser_client


def enable_webparser(base_url: str = "http://localhost:8000"):






    client = get_webparser_client(base_url)
    client.set_enabled(True)
    print(f"WebParser client enabled at {base_url}")


def disable_webparser():

    client = get_webparser_client()
    client.set_enabled(False)
    print("WebParser client disabled")
