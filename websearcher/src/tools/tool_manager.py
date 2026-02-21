


   

import asyncio
import json
import os
import datetime
import re
from typing import Dict, List, Any, Optional, Union, Set, Tuple

                    
from .google_search import (
    google_serper_search_async,
    fetch_page_content_async,
    has_error_indicators,
    SerperQuotaExceededError,
    set_download_defaults
)
from .file_process import get_file_processor, process_file_content
from .download_file import download_file_to_base
from .context_compressor import ContextCompressor
from ..utils.cache_manager import CacheManager
from ..utils.text_utils import extract_json_content
from ..utils.calculator import evaluate_expression, CalculatorError
from ..memory.extraction_intent import ExtractionIntent, DistilledObservation


class ToolManager:



       

    def __init__(
        self,
        serper_api_key: Optional[Union[str, List[str]]] = None,
        serper_api_keys: Optional[List[str]] = None,
        jina_api_key: Optional[str] = None,
        file_base_dir: str = "./data/files",
        aux_client: Optional[Any] = None,
        aux_model_name: str = "qwen-plus",
        enable_cache: bool = True,
        cache_dir: str = "./data/files/cache",
        python_timeout: int = 30,
        download_large_file_threshold_mb: int = 20,
        download_segment_size_mb: int = 5,
        download_max_retries: int = 3
    ):





           
                                                             
        if serper_api_keys:
            self.serper_api_keys = serper_api_keys if isinstance(serper_api_keys, list) else [serper_api_keys]
        elif serper_api_key:
                                   
            if isinstance(serper_api_key, list):
                self.serper_api_keys = serper_api_key
            else:
                self.serper_api_keys = [serper_api_key]
        else:
            raise ValueError("必须提供至少一个 Serper API key")
        
                          
        self._current_api_key_index = 0
        self.serper_api_key = self.serper_api_keys[self._current_api_key_index]        
        self.jina_api_key = jina_api_key
        self.use_jina = bool(jina_api_key)
        self.file_base_dir = file_base_dir

                                                
        self.aux_client = aux_client
        self.aux_model_name = aux_model_name

                                      
        self.cache_manager = CacheManager(
            cache_dir=cache_dir,
            search_engine='google',
            keep_links=False
        ) if enable_cache else None

        # Dedup click_link by (url, goal) to allow re-extraction for the same URL with a new goal
        # while still preventing true loops.
        self._clicked_url_goals: Dict[str, Set[str]] = {}

                                        
        self.compressor = ContextCompressor()

                                                          
        self._file_processor = None
        
                             
        self._current_question: str = ""
        self._fetch_error_log_path: Optional[str] = None
        self.python_timeout = python_timeout
        self.download_settings = {
            "large_file_threshold_mb": download_large_file_threshold_mb,
            "segment_size_mb": download_segment_size_mb,
            "max_retries": download_max_retries
        }
        set_download_defaults(
            large_file_threshold_mb=download_large_file_threshold_mb,
            segment_size_mb=download_segment_size_mb,
            max_retries=download_max_retries
        )

    @property
    def file_processor(self):
                                                   
        if self._file_processor is None:
            self._file_processor = get_file_processor(self.file_base_dir)
        return self._file_processor

    def set_file_base_dir(self, base_dir: str):
                                     
        self.file_base_dir = base_dir
        if self._file_processor:
            self._file_processor.set_base_dir(base_dir)

    def _get_fetch_error_log_path(self) -> str:
        if self._fetch_error_log_path:
            return self._fetch_error_log_path
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        log_dir = os.path.join(repo_root, "logs")
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            log_dir = os.getcwd()
        self._fetch_error_log_path = os.path.join(log_dir, "web_fetch_errors.log")
        return self._fetch_error_log_path

    def _log_fetch_error(self, tool: str, url: str, message: str, extra: Optional[Dict[str, Any]] = None):
        try:
            log_path = self._get_fetch_error_log_path()
            payload = {
                "time": datetime.datetime.now().isoformat(timespec="seconds"),
                "tool": tool,
                "url": url,
                "message": message
            }
            if extra:
                payload["extra"] = extra
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _is_successful_content(self, content: str) -> bool:
        if not content:
            return False
        if content.startswith("Error:"):
            return False
        if has_error_indicators(content) and len(content) < 800:
            return False
        return True

    def _normalize_goal_key(self, goal: str) -> str:
        g = (goal or "").strip().lower()
        g = re.sub(r"\s+", " ", g)
        return g[:300] if len(g) > 300 else g

    def _looks_like_file_url(self, url: str) -> bool:
        if not url:
            return False
        u = url.strip().lower()
        u = u.split("#", 1)[0]
        base = u.split("?", 1)[0]
        # Check for common file extensions that should be downloaded, not browsed
        return any(base.endswith(ext) for ext in (
            # Documents
            ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
            # Archives
            ".zip", ".rar", ".7z", ".gz", ".tgz", ".tar", ".bz2",
            # Data files
            ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml",
            # Text/log files (often large)
            ".txt", ".log", ".dat", ".dump",
            # Media files (binary)
            ".mp4", ".avi", ".mov", ".mp3", ".wav", ".flac",
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff",
            # Code/executables
            ".exe", ".dmg", ".deb", ".rpm", ".msi"
        ))

    async def analyze_tool_intent(
        self,
        tool_name: str,
        arguments: dict,
        reasoning_context: str = ""
    ) -> str:
                                                       
        if not self.aux_client or not reasoning_context:
            return f"Calling {tool_name} to gather information"

        try:
                        
            steps = reasoning_context.split("\n\n")
            recent_steps = "\n\n".join(steps[-10:]) if len(steps) > 10 else reasoning_context

            prompt = f"""Based on the previous reasoning, what is the intent of this tool call?

Tool: {tool_name}
Arguments: {json.dumps(arguments, ensure_ascii=False)}

Recent reasoning:
{recent_steps[-3000:]}

Provide intent in 1-2 sentences:"""

            response = await self.aux_client.chat.completions.create(
                model=self.aux_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Calling {tool_name} to gather information"
    
    async def _analyze_search_intent(
        self,
        query: str,
        reasoning_context: str
    ) -> str:
                          
        if not self.aux_client or not reasoning_context:
            return ""
        
        try:
            from ..prompts.webthinker_prompts import get_search_intent_instruction
            
                                       
            prompt = get_search_intent_instruction(reasoning_context)
            
            response = await self.aux_client.chat.completions.create(
                model=self.aux_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            intent = response.choices[0].message.content.strip()
            
                                 
                                                            
                             
            
            return intent
        except Exception as e:
            print(f"⚠️ Search intent analysis error: {e}")
            return ""
    
    async def _analyze_click_intent(
        self,
        url: str,
        reasoning_context: str
    ) -> str:
                          
        if not self.aux_client or not reasoning_context:
            return ""
        
        try:
            from ..prompts.webthinker_prompts import get_click_intent_instruction
            
                                       
            prompt = get_click_intent_instruction(reasoning_context)
            
            response = await self.aux_client.chat.completions.create(
                model=self.aux_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            intent = response.choices[0].message.content.strip()
            return intent
        except Exception as e:
            print(f"⚠️ Click intent analysis error: {e}")
            return ""

    async def analyze_tool_response(
        self,
        tool_call: dict,
        tool_call_intent: str,
        tool_response: str
    ) -> str:

        if not self.aux_client:
            return tool_response


        if len(str(tool_response)) <= 8000:
            return tool_response

        try:
            prompt = f"""Extract only the helpful information from this tool response.

Tool Call: {json.dumps(tool_call, ensure_ascii=False)}
Intent: {tool_call_intent}

Tool Response:
{tool_response[:15000]}

Output only the relevant information without explanation:"""

            response = await self.aux_client.chat.completions.create(
                model=self.aux_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4000
            )
            return response.choices[0].message.content.strip()

        except Exception:
            return tool_response[:8000] + "\n\n[Response truncated]"

    async def distill_with_intent(
        self,
        raw_content: str,
        intent: ExtractionIntent,
        question: str
    ) -> Tuple[str, DistilledObservation]:
        """Active Cognitive Distillation conditioned on ExtractionIntent.

        Uses the Manager model (G_φ) to distill raw tool output into structured
        observations per the paper's distillation operator:
            o_t = D(G_φ, r_t, i_t)

        Args:
            raw_content: Raw tool output to distill
            intent: Structured ExtractionIntent guiding distillation
            question: Original user question

        Returns:
            Tuple of (distilled_text, DistilledObservation metadata)
        """
        observation = DistilledObservation(
            original_tokens=len(raw_content.split())
        )

        if not self.aux_client:
            observation.distilled_tokens = observation.original_tokens
            observation.compression_ratio = 1.0
            return raw_content, observation

        # Small content doesn't need distillation
        if len(raw_content) <= 3000:
            observation.distilled_tokens = observation.original_tokens
            observation.compression_ratio = 1.0
            return raw_content, observation

        try:
            distillation_prompt = intent.to_distillation_prompt()

            prompt = f"""You are the Manager model performing Active Cognitive Distillation.

## Question
{question}

{distillation_prompt}

## Raw Content (first 20000 chars)
{raw_content[:20000]}

## Instructions
Distill the raw content according to the extraction intent above.
- Extract ONLY information matching the target
- Filter out noise matching the constraints
- Preserve exact values, dates, names, and numerical data

## Required Output Format
For each piece of evidence, output a SNIPPET line with source attribution:
  SNIPPET: <relevant text> | SOURCE: <url or domain>

If factual candidates are found, also list them as:
  CANDIDATE: entity={{...}} attr={{...}} value={{...}} source={{...}}

Then provide a brief SUMMARY of the distilled content.

## Distilled Output:"""

            response = await self.aux_client.chat.completions.create(
                model=self.aux_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4000
            )
            distilled = response.choices[0].message.content.strip()

            if distilled:
                observation.distilled_tokens = len(distilled.split())
                observation.compression_ratio = (
                    observation.distilled_tokens / observation.original_tokens
                    if observation.original_tokens > 0 else 1.0
                )

                # Parse CANDIDATE lines from distilled output
                import re
                candidate_pattern = re.compile(
                    r'CANDIDATE:\s*entity=\{([^}]*)\}\s*attr=\{([^}]*)\}\s*value=\{([^}]*)\}\s*source=\{([^}]*)\}'
                )
                for match in candidate_pattern.finditer(distilled):
                    observation.add_candidate(
                        entity=match.group(1).strip(),
                        attr=match.group(2).strip(),
                        value=match.group(3).strip(),
                        url="",
                        domain=match.group(4).strip(),
                        snippet=""
                    )

                # Parse SNIPPET lines: o_t = {(snippet_k, source_k)}
                snippet_pattern = re.compile(
                    r'SNIPPET:\s*(.+?)\s*\|\s*SOURCE:\s*(.+)'
                )
                for match in snippet_pattern.finditer(distilled):
                    snippet_text = match.group(1).strip()
                    source_text = match.group(2).strip()
                    observation.add_snippet(
                        url=source_text,
                        domain=source_text,
                        snippet=snippet_text
                    )

                original_size = len(raw_content)
                return (
                    f"[Distilled from {original_size} chars by Manager (G_φ) | "
                    f"intent: {intent.intent_family}]\n\n{distilled}",
                    observation
                )

        except Exception as e:
            print(f"Distillation failed: {e}, falling back to truncation")

        # Fallback: truncate
        if len(raw_content) > 20000:
            raw_content = raw_content[:20000] + f"\n\n[Content truncated from {len(raw_content)} chars]"
        observation.distilled_tokens = len(raw_content.split())
        observation.compression_ratio = (
            observation.distilled_tokens / observation.original_tokens
            if observation.original_tokens > 0 else 1.0
        )
        return raw_content, observation

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict,
        reasoning_context: str = "",
        extraction_intent: Optional[ExtractionIntent] = None
    ) -> str:
        try:
            if tool_name == "web_search":
                return await self._web_search(arguments, reasoning_context, extraction_intent)
            elif tool_name == "browse_pages":
                return await self._browse_pages(arguments, extraction_intent)
            elif tool_name == "click_link":
                return await self._click_link(arguments, reasoning_context, extraction_intent)
            elif tool_name == "execute_python_code":
                return await self._execute_python(arguments)
            elif tool_name == "process_file":
                return await self._process_file(arguments)
            elif tool_name == "solve_math":
                return await self._solve_math(arguments)
            elif tool_name == "calculator":
                return self._calculator(arguments)
            elif tool_name == "download_file":
                return await self._download_file(arguments)
            else:
                return f"Error: Unknown tool '{tool_name}'"
        except Exception as e:
            return f"Error calling tool {tool_name}: {str(e)}"

    async def _web_search(self, args: dict, reasoning_context: str = "", extraction_intent: Optional[ExtractionIntent] = None) -> str:
        query = args.get("query", "").strip()
        if not query:
            return json.dumps({"error": "Missing required parameter 'query'"})

        print(f"\n[WebSearch] New search request: {query}")

        # Use ExtractionIntent as search intent; fall back to legacy _analyze_search_intent
        search_intent = ""
        if extraction_intent:
            search_intent = extraction_intent.to_prompt_string()
            print(f"Search intent (from ExtractionIntent): {extraction_intent.target[:80]}")
        elif self.aux_client and reasoning_context:
            try:
                search_intent = await self._analyze_search_intent(
                    query=query,
                    reasoning_context=reasoning_context
                )
                if search_intent:
                    print(f"Search intent (from analysis): {search_intent}")
            except Exception as e:
                print(f"Intent analysis failed: {e}")

                     
        if self.cache_manager:
            cached = self.cache_manager.get_search_results(query)
            if cached is not None:
                print(f"[WebSearch] ✅ Cache hit for query")
                return self._format_search_results(cached, query)

                                         
        last_error = None
        for attempt in range(len(self.serper_api_keys)):
            current_key = self.serper_api_keys[self._current_api_key_index]
            try:
                print(f"[WebSearch] ▶ Using Serper API key #{self._current_api_key_index + 1} (attempt {attempt + 1}/{len(self.serper_api_keys)})")
                                             
                results = await google_serper_search_async(
                    query=query,
                    api_key=current_key,
                    timeout=30,
                    num_results=20                              
                )

                if not results:
                    print(f"[WebSearch] ⚠ No search results returned for query")
                    return json.dumps({"message": "No search results found", "results": []})

                                        
                if self.cache_manager:
                    self.cache_manager.set_search_results(query, results)

                                            
                results = self.compressor.prioritize_search_results(
                    results, 
                    query, 
                    max_results=15                     
                )

                return self._format_search_results(results, query)

            except SerperQuotaExceededError as e:
                last_error = e
                print(f"⚠️ Serper API key {self._current_api_key_index + 1} 额度耗尽，尝试切换到备用 key...")
                
                                
                self._current_api_key_index = (self._current_api_key_index + 1) % len(self.serper_api_keys)
                self.serper_api_key = self.serper_api_keys[self._current_api_key_index]             
                
                                    
                if attempt == len(self.serper_api_keys) - 1:
                    print(f"[WebSearch] ❌ All Serper API keys exhausted for query: {query}")
                    return json.dumps({
                        "error": "所有 Serper API key 额度已耗尽",
                        "message": str(last_error)
                    })
                
                             
                continue
        
                             
        print(f"[WebSearch] ❌ Search failed for query: {query}, last_error={last_error}")
        return json.dumps({
            "error": "搜索失败",
            "message": str(last_error) if last_error else "未知错误"
        })

    def _format_search_results(self, results: List[Dict], query: str = "") -> str:



           
        if not results:
            return json.dumps({"message": "No results", "results": []})

        formatted = [f"Search results for: {query}\n" if query else "Search results:\n"]
                                           
        max_display = min(len(results), 15)
        for i, r in enumerate(results[:max_display], 1):
            formatted.append(
                f"{i}. {r.get('title', 'No title')}\n"
                f"   URL: {r.get('url', 'No URL')}\n"
                f"   {r.get('snippet', 'No snippet')}\n"
            )
        
                                 
        if len(results) > max_display:
            formatted.append(f"\n(Showing top {max_display} most relevant results out of {len(results)} total)")
        
        return "\n".join(formatted)

    async def _browse_pages(self, args: dict, extraction_intent: Optional[ExtractionIntent] = None) -> str:
                                                           
        urls = args.get("urls")
        query = args.get("query", "")
        force_refresh = bool(args.get("force_refresh"))

        if not urls:
            return json.dumps({"error": "Missing required parameter 'urls'"})

        if not isinstance(urls, list):
            urls = [urls]

        urls = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
        if not urls:
            return json.dumps({"error": "No valid URLs provided"})

        # Filter out file URLs and suggest download_file instead
        file_urls = []
        page_urls = []
        for url in urls:
            if self._looks_like_file_url(url):
                file_urls.append(url)
            else:
                page_urls.append(url)
        
        if file_urls:
            file_msg = (
                f"The following URLs look like file downloads and should use `download_file` instead of `browse_pages`:\n"
                + "\n".join(f"- {url}" for url in file_urls) +
                f"\n\nUse: download_file(url='<url>', process=true) for each file."
            )
            if not page_urls:
                return file_msg
            # If there are both file and page URLs, warn but continue with page URLs
            file_msg = f"[Warning] {file_msg}\n\nContinuing with page URLs only.\n\n"
        else:
            file_msg = ""

                                                
        # Only process page URLs (file URLs were filtered out above)
        urls = page_urls
        if not urls:
            return file_msg if file_msg else json.dumps({"error": "No valid page URLs provided"})
        
        cached_results = {}
        urls_to_fetch = []

        for url in urls:
            if self.cache_manager and not force_refresh:
                cached = self.cache_manager.get_url_content(url)
                if cached is not None:
                    extracted_text, _ = cached
                    if self._is_successful_content(extracted_text):
                        cached_results[url] = cached
                        continue
            urls_to_fetch.append(url)

                             
        if urls_to_fetch:
            results = await fetch_page_content_async(
                urls=urls_to_fetch,
                use_jina=self.use_jina,
                jina_api_key=self.jina_api_key,
                show_progress=False,
                keep_links=False,
                max_concurrent=32,
                use_webparser=True,                           
                use_crawl4ai=True                            
            )
                               
            for url, content in results.items():
                extracted, full = content
                if self.cache_manager and self._is_successful_content(extracted):
                    self.cache_manager.set_url_content(url, extracted, full)
                cached_results[url] = content

                        
        formatted = []
        if file_msg:
            formatted.append(file_msg)
        
        # Limit per-URL content to prevent context explosion when processing multiple URLs
        MAX_PER_URL_CHARS = 10000  # Hard limit per URL in browse_pages
        
        for url in urls:
            if url in cached_results:
                extracted_text, _ = cached_results[url]

                if extracted_text.startswith("Error:") or has_error_indicators(extracted_text):
                    self._log_fetch_error(
                        tool="browse_pages",
                        url=url,
                        message=extracted_text[:800],
                        extra={"query": query}
                    )

                # Apply compression/truncation with per-URL limit
                # Use extraction intent target for relevance scoring when available
                effective_query = query
                if not effective_query and extraction_intent:
                    effective_query = extraction_intent.target

                if len(extracted_text) > MAX_PER_URL_CHARS:
                    if effective_query:
                        # Try compression first if we have a query/intent
                        compressed = self.compressor.compress_content(
                            content=extracted_text,
                            query=effective_query,
                            max_length=MAX_PER_URL_CHARS,
                            keep_ratio=0.4
                        )
                    else:
                        # No query or intent, just truncate
                        compressed = extracted_text[:MAX_PER_URL_CHARS] + "\n[Content truncated - use query parameter for relevance-based compression]"
                elif effective_query and len(extracted_text) > 6000:
                    # Medium-sized content with query/intent: compress
                    compressed = self.compressor.compress_content(
                        content=extracted_text,
                        query=effective_query,
                        max_length=6000,
                        keep_ratio=0.4
                    )
                else:
                    # Small content or no query: use as-is (but still cap at 8000)
                    compressed = extracted_text[:8000] if len(extracted_text) > 8000 else extracted_text

                formatted.append(f"=== URL: {url} ===\n{compressed}\n")
            else:
                self._log_fetch_error(
                    tool="browse_pages",
                    url=url,
                    message="Failed to fetch",
                    extra={"query": query}
                )
                formatted.append(f"=== URL: {url} ===\nFailed to fetch\n")

        return "\n".join(formatted)

    async def _click_link(self, args: dict, reasoning_context: str = "", extraction_intent: Optional[ExtractionIntent] = None) -> str:
        url = args.get("url", "").strip()
        goal = args.get("goal") or args.get("intent")
        force_refresh = bool(args.get("force_refresh"))

        if not url:
            return json.dumps({"error": "Missing required parameter 'url'"})

        # Avoid fetching binary/file endpoints through click_link (often bloats context and triggers empty responses).
        if self._looks_like_file_url(url):
            return (
                f"This URL looks like a file download (e.g., PDF). "
                f"Use `download_file` instead: download_file(url='{url}', process=true)."
            )

        # Use ExtractionIntent target as goal if no explicit goal provided
        if not goal and extraction_intent:
            goal = extraction_intent.target
            print(f"Goal from ExtractionIntent: {goal[:80]}")
        elif not goal and self.aux_client and reasoning_context:
            try:
                click_intent = await self._analyze_click_intent(
                    url=url,
                    reasoning_context=reasoning_context
                )
                if click_intent:
                    goal = click_intent
                    print(f"Goal from click intent analysis: {goal}")
            except Exception as e:
                print(f"Click intent analysis failed: {e}")

        if not goal:
            goal = "extract relevant information"

        goal_key = self._normalize_goal_key(goal)
        if not force_refresh:
            prev_goals = self._clicked_url_goals.get(url, set())
            if goal_key in prev_goals:
                return (
                    f"You have already clicked this URL for the same goal: {url}. "
                    f"Use the previous information, or set force_refresh=true."
                )

                     
        cached_content = None
        if self.cache_manager and not force_refresh:
            cached_content = self.cache_manager.get_url_content(url)

        if cached_content:
            extracted_text, full_text = cached_content
            if not self._is_successful_content(extracted_text):
                cached_content = None
                extracted_text = ""
                full_text = ""
        else:
            results = await fetch_page_content_async(
                urls=[url],
                use_jina=self.use_jina,
                jina_api_key=self.jina_api_key,
                show_progress=False,
                keep_links=False,
                max_concurrent=1,
                use_webparser=True,                           
                use_crawl4ai=True                            
            )

            if url not in results:
                return f"Failed to fetch URL: {url}"

            extracted_text, full_text = results[url]

                   
            if self.cache_manager and self._is_successful_content(extracted_text):
                self.cache_manager.set_url_content(url, extracted_text, full_text)

                                       
        if extracted_text.startswith("Error:") or "timeout" in extracted_text.lower()[:200]:
                             
            if len(extracted_text) < 1000:            
                message = f"Unable to fetch content from {url}. {extracted_text}. Try other links or check if the page is accessible."
                self._log_fetch_error("click_link", url, extracted_text[:800], extra={"goal": goal})
                return message
        
                                                     
                               
        if has_error_indicators(extracted_text):
                                                 
            if len(extracted_text) < 500:
                message = f"Unable to fetch content from {url}. The page may require JavaScript or be temporarily unavailable. Try other links."
                self._log_fetch_error("click_link", url, extracted_text[:800], extra={"goal": goal})
                return message
                            

                                                                    
        self._clicked_url_goals.setdefault(url, set()).add(goal_key)
        if self.aux_client and len(extracted_text) > 500:
            try:
                content_for_extraction = extracted_text[:20000]

                # Use ExtractionIntent for guided extraction if available
                if extraction_intent:
                    intent_guidance = extraction_intent.to_distillation_prompt()
                    prompt = f"""Extract information from this webpage guided by the extraction intent.

{intent_guidance}

Goal: {goal}

Webpage content:
{content_for_extraction}

Provide:
1. RELEVANT INFO: All information matching the extraction target (be comprehensive)
2. KEY FACTS: Specific numbers, dates, names, locations found
3. CANDIDATES: If factual candidates are found, list as:
   CANDIDATE: entity={{...}} attr={{...}} value={{...}} source={{...}}
4. SUMMARY: Concise summary addressing the goal

Output format:
RELEVANT INFO:
[relevant content]

KEY FACTS:
[facts]

SUMMARY:
[summary]"""
                else:
                    prompt = f"""Extract information relevant to the goal from this webpage.

Goal: {goal}

Webpage content:
{content_for_extraction}

Provide:
1. RELEVANT INFO: All information relevant to the goal (be comprehensive)
2. KEY FACTS: Specific numbers, dates, names, locations found
3. SUMMARY: Concise summary addressing the goal

Output format:
RELEVANT INFO:
[relevant content]

KEY FACTS:
[facts]

SUMMARY:
[summary]"""

                response = await self.aux_client.chat.completions.create(
                    model=self.aux_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=4000
                )
                extraction = response.choices[0].message.content.strip()

                return f"""=== Clicked: {url} ===
Goal: {goal}

{extraction}
"""
            except Exception:
                pass

                                       
        compressed = self.compressor.compress_content(
            content=extracted_text,
            query=goal,
            max_length=8000,
            keep_ratio=0.5
        )

        return f"""=== Clicked: {url} ===
Goal: {goal}

Content:
{compressed}
"""

    def set_question_context(self, question: str):
                               
        self._current_question = question
    
    async def _execute_python(self, args: dict) -> str:
                                                            
        from .python_executor import execute_python_code

        code = args.get("code", "").strip()
        if not code:
            return json.dumps({"error": "Missing required parameter 'code'"})

                       
        return await execute_python_code(
            code,
            timeout=self.python_timeout,
            question=self._current_question,
            base_dir=self.file_base_dir
        )

    def _calculator(self, args: dict) -> str:
        expression = (args.get("expression") or "").strip()
        precision = args.get("precision")
        if not expression:
            return json.dumps({"error": "Missing required parameter 'expression'"})
        try:
            precision_val = int(precision) if precision is not None else None
        except Exception:
            return json.dumps({"error": "Invalid 'precision' parameter; must be an integer"})
        try:
            result = evaluate_expression(expression, precision=precision_val)
            return json.dumps(result, ensure_ascii=False)
        except CalculatorError as e:
            return json.dumps({"error": str(e)})

    async def _solve_math(self, args: dict) -> str:
        problem = (args.get("problem") or "").strip()
        requirements = (args.get("requirements") or "").strip()
        if not problem:
            return json.dumps({"error": "Missing required parameter 'problem'"})
        if not self.aux_client:
            return json.dumps({"error": "Auxiliary model not configured for solve_math"})
        from .python_executor import execute_python_code

        prompt = f"""You are a math solver. Provide a step-by-step plan and Python code to compute the answer.

Problem:
{problem}

Requirements:
{requirements or "N/A"}

Output JSON only with keys:
{{
  "steps": ["short step 1", "short step 2", "..."],
  "python_code": "python code using print() for key values and the final answer"
}}
"""
        try:
            response = await self.aux_client.chat.completions.create(
                model=self.aux_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1200
            )
            raw = response.choices[0].message.content or ""
        except Exception as e:
            return json.dumps({"error": f"solve_math LLM error: {e}"})

        parsed = None
        try:
            cleaned = extract_json_content(raw)
            parsed = json.loads(cleaned)
        except Exception:
            parsed = None

        if not parsed or not isinstance(parsed, dict):
            return json.dumps({
                "error": "Failed to parse solve_math output",
                "raw": raw[:2000]
            })

        steps = parsed.get("steps") if isinstance(parsed.get("steps"), list) else []
        python_code = (parsed.get("python_code") or "").strip()
        if not python_code:
            return json.dumps({
                "error": "No python_code generated by solve_math",
                "steps": steps
            })

        result = await execute_python_code(
            python_code,
            timeout=self.python_timeout,
            question=self._current_question,
            base_dir=self.file_base_dir
        )

        return json.dumps({
            "steps": steps,
            "python_code": python_code,
            "python_result": result
        }, ensure_ascii=False)

    async def _download_file(self, args: dict) -> str:
        url = (args.get("url") or "").strip()
        file_name = (args.get("file_name") or "").strip()
        overwrite = bool(args.get("overwrite") or args.get("force_refresh"))
        process = args.get("process")
        process = True if process is None else bool(process)
        fallback_search = args.get("fallback_search")
        fallback_search = True if fallback_search is None else bool(fallback_search)
        try:
            segment_size_mb = int(args.get("segment_size_mb") or self.download_settings["segment_size_mb"])
        except Exception:
            segment_size_mb = self.download_settings["segment_size_mb"]
        try:
            large_file_threshold_mb = int(args.get("large_file_threshold_mb") or self.download_settings["large_file_threshold_mb"])
        except Exception:
            large_file_threshold_mb = self.download_settings["large_file_threshold_mb"]
        try:
            max_retries = int(args.get("max_retries") or self.download_settings["max_retries"])
        except Exception:
            max_retries = self.download_settings["max_retries"]

        if not url:
            return json.dumps({"error": "Missing required parameter 'url'"})

        result = download_file_to_base(
            url=url,
            base_dir=self.file_base_dir,
            file_name=file_name or None,
            overwrite=overwrite,
            max_retries=max_retries,
            segment_size_mb=segment_size_mb,
            large_file_threshold_mb=large_file_threshold_mb
        )

        if not result.get("success"):
            self._log_fetch_error("download_file", url, str(result.get("error", "download failed")))
            attempted_urls = result.get("attempted_urls") or []
            num_attempts = result.get("num_attempts", len(attempted_urls))
            
            # Build error message
            error_parts = [
                "=== Download Failed ===",
                f"Error: {result.get('error', 'Unknown error')}",
            ]
            
            if num_attempts > 1:
                error_parts.extend([
                    "",
                    f"Attempted {num_attempts} URLs (including fallback URLs):",
                    "\n".join(f"  {i+1}. {u}" for i, u in enumerate(attempted_urls))
                ])
            elif attempted_urls:
                error_parts.append(f"\nAttempted URL: {attempted_urls[0]}")
            
            fallback_parts = error_parts + [
                "",
                "Note: All fallback URLs were automatically tried but failed.",
            ]
            if fallback_search:
                query = ""
                try:
                    from urllib.parse import urlparse
                    import os as _os
                    parsed = urlparse(url)
                    basename = _os.path.basename(parsed.path or "")
                    if file_name:
                        query = f"\"{file_name}\" pdf"
                    elif basename and "." in basename:
                        query = f"\"{basename}\" pdf"
                    else:
                        query = f"{parsed.netloc} file download"
                    uuid_match = re.search(r"[0-9a-fA-F-]{36}", url)
                    if uuid_match:
                        query = f"{uuid_match.group(0)} file"
                except Exception:
                    query = url
                if query:
                    try:
                        search_result = await self._web_search({"query": query})
                        fallback_parts.extend([
                            "",
                            f"Fallback search query: {query}",
                            search_result
                        ])
                    except Exception:
                        pass
            return "\n".join(fallback_parts)

        output = [
            "=== Downloaded File ===",
            f"File: {result.get('file_name')}",
            f"Path: {result.get('file_path')}",
            f"Size: {result.get('size_bytes')} bytes",
            f"Final URL: {result.get('final_url')}",
            f"Segmented: {result.get('segmented')}",
            f"Skipped: {result.get('skipped')}",
        ]

        if process:
            parsed = await process_file_content(result["file_name"], self.file_base_dir)
            output.append("")
            output.append(parsed)

        return "\n".join(output)

    async def _process_file(self, args: dict) -> str:
                                                            
        import os

        file_name = args.get("file_name", "").strip()
        if not file_name:
            return json.dumps({
                "error": "Missing required parameter 'file_name'",
                "usage": "Provide the file name, e.g., 'document.pdf'"
            })

        file_path = os.path.join(self.file_base_dir, file_name)

                              
        if not os.path.exists(file_path):
                                  
            available = self.file_processor.list_files()
            if available:
                files_list = "\n".join(f"  - {f}" for f in available[:15])
                extra = f"\n  ... and {len(available) - 15} more" if len(available) > 15 else ""
                return f"""Error: File not found: {file_name}

Available files in {self.file_base_dir}:
{files_list}{extra}

Please use one of the available file names."""
            else:
                return f"Error: File not found: {file_name}\nNo files found in {self.file_base_dir}"

                      
        result = await process_file_content(file_name, self.file_base_dir)
        return result

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
                                                   
        return [
            {
                "name": "web_search",
                "description": "Search the web using Google. Returns titles, URLs, and snippets. Use specific queries for better results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query - be specific and include key terms"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "browse_pages",
                "description": "Extract content from web pages. Supports multiple URLs and PDFs. Provide a query for relevance-based compression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "URLs to browse"
                        },
                        "query": {
                            "type": "string",
                            "description": "Optional: search query for relevance-based extraction"
                        },
                        "force_refresh": {
                            "type": "boolean",
                            "description": "If true, bypass cache and refetch URLs"
                        }
                    },
                    "required": ["urls"]
                }
            },
            {
                "name": "click_link",
                "description": "Click a link for detailed content extraction. Provide a clear goal when possible; if omitted the system will infer one from context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to click"
                        },
                        "goal": {
                            "type": "string",
                            "description": "What specific information you're looking for. Be clear and specific."
                        },
                        "force_refresh": {
                            "type": "boolean",
                            "description": "If true, refetch even if this URL was already clicked"
                        }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "download_file",
                "description": "Download a file to local storage, then optionally parse it with process_file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "File URL to download"
                        },
                        "file_name": {
                            "type": "string",
                            "description": "Optional file name override"
                        },
                        "overwrite": {
                            "type": "boolean",
                            "description": "Overwrite existing file if it already exists"
                        },
                        "process": {
                            "type": "boolean",
                            "description": "If true, parse the file after download using process_file"
                        },
                        "segment_size_mb": {
                            "type": "integer",
                            "description": "Segment size in MB for large files (default 5)"
                        },
                        "large_file_threshold_mb": {
                            "type": "integer",
                            "description": "File size threshold in MB to enable segmented download (default 20)"
                        },
                        "max_retries": {
                            "type": "integer",
                            "description": "Max retries per segment or full download"
                        },
                        "fallback_search": {
                            "type": "boolean",
                            "description": "If download fails, run a web search to find alternative sources"
                        }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "execute_python_code",
                "description": "Execute Python code for calculations, data analysis, or file operations. Results are returned via print statements.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute. Use print() to output results."
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "process_file",
                "description": "Read and extract content from files (PDF, Excel, CSV, Word, JSON, XML, etc.). Returns the COMPLETE file content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "File name including extension (e.g., 'report.pdf', 'data.xlsx')"
                        }
                    },
                    "required": ["file_name"]
                }
            }
            ,
            {
                "name": "solve_math",
                "description": "Solve a math or quantitative problem with explicit steps and Python-backed computation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "problem": {
                            "type": "string",
                            "description": "The full math problem statement."
                        },
                        "requirements": {
                            "type": "string",
                            "description": "Optional constraints or formatting requirements."
                        }
                    },
                    "required": ["problem"]
                }
            }
            ,
            {
                "name": "calculator",
                "description": "Evaluate a mathematical expression safely (supports + - * / // % **, parentheses, and basic math functions).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression, e.g. '(356870/20.897)/1000'"
                        },
                        "precision": {
                            "type": "integer",
                            "description": "Optional decimal rounding precision (e.g., 2)."
                        }
                    },
                    "required": ["expression"]
                }
            }
        ]

    def save_cache(self):
                                
        if self.cache_manager:
            self.cache_manager.save_all()

    def clear_session(self):
                                                                 
        self._clicked_url_goals.clear()

    def get_cache_stats(self) -> Dict:
                                  
        if self.cache_manager:
            return self.cache_manager.get_stats()
        return {'cache': 'disabled'}
