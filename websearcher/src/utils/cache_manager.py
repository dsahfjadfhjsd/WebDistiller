









import json
import os
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import hashlib
from datetime import datetime
import atexit

                      
_global_cache_manager: Optional['CacheManager'] = None


class CacheManager:









    def __init__(
        self,
        cache_dir: str = './data/files/cache',
        search_engine: str = 'google',
        keep_links: bool = False,
        auto_save_threshold: int = 10                                   
    ):









        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

                                                          
        self.search_cache_path = self.cache_dir / f'{search_engine}_search_cache.json'
        if keep_links:
            self.url_cache_path = self.cache_dir / 'url_cache_with_links.json'
        else:
            self.url_cache_path = self.cache_dir / 'url_cache.json'

                              
        self.search_cache = self._load_cache(self.search_cache_path)
        self.url_cache = self._load_cache(self.url_cache_path)

                                      
        self._search_dirty = False
        self._url_dirty = False
        self._pending_saves = 0
        self._auto_save_threshold = auto_save_threshold

                    
        self.stats = {
            'search_hits': 0,
            'search_misses': 0,
            'url_hits': 0,
            'url_misses': 0
        }

                                      
        atexit.register(self._save_on_exit)

    def _save_on_exit(self):

        if self._search_dirty or self._url_dirty:
            self.save_all()

    def _load_cache(self, path: Path) -> Dict:

        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                print(f"Loaded cache from {path} ({len(cache)} entries)")
                return cache
            except Exception as e:
                print(f"Error loading cache from {path}: {e}")
                return {}
        return {}

    def _save_cache(self, cache: Dict, path: Path):
        """
        Save cache to disk safely using a temporary file and atomic rename.
        Also creates a backup of the previous version.
        """
        temp_path = path.with_suffix('.tmp')
        backup_path = path.with_suffix('.json.backup')
        
        try:
            # Write to temp file first
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            
            # If successful, backup the old file if it exists
            if path.exists():
                import shutil
                try:
                    shutil.copy2(path, backup_path)
                except Exception as e:
                    print(f"Warning: Failed to create backup: {e}")

            # Atomic rename
            os.replace(temp_path, path)
            
        except Exception as e:
            print(f"Error saving cache to {path}: {e}")
            # Clean up temp file if it exists
            if temp_path.exists():
                try:
                    os.remove(temp_path)
                except:
                    pass

    def _normalize_query(self, query: str) -> str:






        return ' '.join(query.lower().split())

    def _maybe_auto_save(self):

        self._pending_saves += 1
        if self._pending_saves >= self._auto_save_threshold:
            self.save_all()
            self._pending_saves = 0

    def get_search_results(self, query: str) -> Optional[Any]:









                                                  
        if query in self.search_cache:
            self.stats['search_hits'] += 1
            return self.search_cache[query]

        normalized = self._normalize_query(query)
        if normalized in self.search_cache:
            self.stats['search_hits'] += 1
            return self.search_cache[normalized]

        self.stats['search_misses'] += 1
        return None

    def set_search_results(self, query: str, results: Any):







                                                         
        normalized = self._normalize_query(query)
        self.search_cache[normalized] = results
                                          
        if query != normalized:
            self.search_cache[query] = results
        self._search_dirty = True
        self._maybe_auto_save()

    def get_url_content(self, url: str) -> Optional[Tuple[str, str]]:









        if url in self.url_cache:
            self.stats['url_hits'] += 1
            cached = self.url_cache[url]

                                                
            if isinstance(cached, dict):
                return (cached.get('extracted_text', ''), cached.get('full_text', ''))
            elif isinstance(cached, (list, tuple)) and len(cached) == 2:
                return (cached[0], cached[1])
            else:
                                       
                return (cached, cached)

        self.stats['url_misses'] += 1
        return None

    def set_url_content(self, url: str, extracted_text: str, full_text: str = None):








        if full_text is None:
            full_text = extracted_text

                                              
        self.url_cache[url] = [extracted_text, full_text]
        self._url_dirty = True
        self._maybe_auto_save()

    def has_search_query(self, query: str) -> bool:

        if query in self.search_cache:
            return True
        return self._normalize_query(query) in self.search_cache

    def has_url(self, url: str) -> bool:

        return url in self.url_cache

    def clear_search_cache(self):

        self.search_cache = {}
        self._save_cache(self.search_cache, self.search_cache_path)
        self._search_dirty = False
        print("Search cache cleared")

    def clear_url_cache(self):

        self.url_cache = {}
        self._save_cache(self.url_cache, self.url_cache_path)
        self._url_dirty = False
        print("URL cache cleared")

    def clear_all(self):

        self.clear_search_cache()
        self.clear_url_cache()
        print("All caches cleared")

    def save_all(self):

        if self._search_dirty:
            self._save_cache(self.search_cache, self.search_cache_path)
            self._search_dirty = False
        if self._url_dirty:
            self._save_cache(self.url_cache, self.url_cache_path)
            self._url_dirty = False

    def get_stats(self) -> Dict:

        total_searches = self.stats['search_hits'] + self.stats['search_misses']
        total_urls = self.stats['url_hits'] + self.stats['url_misses']

        search_rate = (self.stats['search_hits'] / total_searches * 100) if total_searches > 0 else 0
        url_rate = (self.stats['url_hits'] / total_urls * 100) if total_urls > 0 else 0

        return {
            'search_cache_size': len(self.search_cache),
            'url_cache_size': len(self.url_cache),
            'search_hits': self.stats['search_hits'],
            'search_misses': self.stats['search_misses'],
            'search_hit_rate': f"{search_rate:.1f}%",
            'url_hits': self.stats['url_hits'],
            'url_misses': self.stats['url_misses'],
            'url_hit_rate': f"{url_rate:.1f}%",
        }

    def print_stats(self):

        stats = self.get_stats()
        print(f"\nCache Stats: Search {stats['search_cache_size']} entries ({stats['search_hit_rate']} hit), "
              f"URL {stats['url_cache_size']} entries ({stats['url_hit_rate']} hit)")

    def export_cache_info(self, output_path: str):






        info = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_stats(),
            'search_queries': list(self.search_cache.keys()),
            'cached_urls': list(self.url_cache.keys()),
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        print(f"Cache info exported to {output_path}")

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.save_all()


def get_cache_manager(
    cache_dir: str = './data/files/cache',
    search_engine: str = 'google',
    keep_links: bool = False
) -> CacheManager:











    global _global_cache_manager

    if _global_cache_manager is None:
        _global_cache_manager = CacheManager(
            cache_dir=cache_dir,
            search_engine=search_engine,
            keep_links=keep_links
        )

    return _global_cache_manager


def reset_cache_manager():

    global _global_cache_manager
    if _global_cache_manager is not None:
        _global_cache_manager.save_all()
    _global_cache_manager = None
