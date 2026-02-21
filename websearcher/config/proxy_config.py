




        
PROXY_ENABLED = True                

               
                                                         
HTTP_PROXY = "http://127.0.0.1:2222"                               
HTTPS_PROXY = "http://127.0.0.1:2222"                               

          
PROXY_URL = "http://127.0.0.1:2222"                               

             
PROXY_DOMAINS = [
    "wikipedia.org",
    "wikimedia.org",
    "github.com",
    "google.com",
    "googleapis.com",
    "gstatic.com",
    "youtube.com",
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "genius.com",
    "allmusic.com",
    "discogs.com",
    "spotify.com",
    "apple.com",
                 
]

def get_proxy_config():

    if not PROXY_ENABLED:
        return None
    
    if PROXY_URL:
        return {
            "http": PROXY_URL,
            "https": PROXY_URL
        }
    
    return {
        "http": HTTP_PROXY,
        "https": HTTPS_PROXY
    }


def should_use_proxy(url: str) -> bool:

    if not PROXY_ENABLED:
        return False
    
    for domain in PROXY_DOMAINS:
        if domain in url:
            return True
    
    return False
