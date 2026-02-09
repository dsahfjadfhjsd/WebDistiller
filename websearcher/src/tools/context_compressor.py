










import re
from typing import List, Dict, Optional

                              
_NUMBER_PATTERN = re.compile(r'\d+(?:\.\d+)?(?:%|°|km|m|kg|lb|ft|in|cm|mm)?')
_DATE_PATTERN = re.compile(r'\b(?:19|20)\d{2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b', re.IGNORECASE)
_SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')

                                                              
KEY_WORDS = frozenset([
                           
    'however', 'therefore', 'conclusion', 'result', 'consequently',
    'found', 'shows', 'indicates', 'suggests', 'demonstrates',
    'important', 'significant', 'key', 'main', 'primary',
                     
    'table', 'figure', 'data', 'evidence', 'statistics',
                                           
    'answer', 'solution', 'total', 'sum', 'count', 'number',
    'name', 'called', 'known', 'titled', 'named',
    'born', 'died', 'founded', 'established', 'created',
    'located', 'capital', 'population', 'area', 'size',
    'first', 'last', 'only', 'largest', 'smallest', 'highest', 'lowest',
    'winner', 'author', 'director', 'founder', 'inventor',
    'released', 'published', 'launched', 'announced',
                    
    'approximately', 'about', 'around', 'exactly', 'precisely',
    'million', 'billion', 'thousand', 'hundred', 'percent',
])

                                  
AUTHORITY_DOMAINS = frozenset([
    'wikipedia.org', 'nature.com', 'science.org',
    '.gov', '.edu', 'arxiv.org', 'nih.gov',
    'usgs.gov', 'nasa.gov', 'who.int',
    'ieee.org', 'acm.org', 'springer.com',
    'britannica.com', 'imdb.com', 'worldbank.org',
])

                    
_compressor_instance: Optional['ContextCompressor'] = None


def get_context_compressor() -> 'ContextCompressor':

    global _compressor_instance
    if _compressor_instance is None:
        _compressor_instance = ContextCompressor()
    return _compressor_instance


class ContextCompressor:









    def __init__(self):
                                           
        self._sent_tokenize = None
        try:
            from nltk.tokenize import sent_tokenize
            self._sent_tokenize = sent_tokenize
        except ImportError:
            pass

    def _tokenize_sentences(self, content: str) -> List[str]:

        if self._sent_tokenize:
            try:
                return self._sent_tokenize(content)
            except Exception:
                pass
                             
        return [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(content) if s.strip()]

    def compress_content(
        self,
        content: str,
        query: str,
        max_length: int = 15000,                          
        keep_ratio: float = 0.5                                                  
    ) -> str:





                        
        if len(content) <= max_length:
            return content

              
        sentences = self._tokenize_sentences(content)

        if not sentences:
            return content[:max_length]

                       
        query_terms = self._extract_query_terms(query)

                 
        scored_sentences = []
        total = len(sentences)
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, query_terms, i, total)
            scored_sentences.append((score, i, sentence))

               
        scored_sentences.sort(reverse=True, key=lambda x: x[0])

                                                                       
        num_keep = max(5, int(total * keep_ratio))
        selected = scored_sentences[:num_keep]

                        
        selected.sort(key=lambda x: x[1])

              
        compressed = ' '.join([s[2] for s in selected])

                   
        if len(compressed) > max_length:
            compressed = compressed[:max_length] + "..."

        return compressed

    def _extract_query_terms(self, query: str) -> set:

        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from',
                     'and', 'or', 'but', 'if', 'then', 'than', 'that', 'this',
                     'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'how', 'why'}
        terms = set(query.lower().split())
        return terms - stopwords

    def _score_sentence(
        self,
        sentence: str,
        query_terms: set,
        position: int,
        total: int
    ) -> float:








        score = 0.0
        sentence_lower = sentence.lower()
        sentence_terms = set(sentence_lower.split())

                                            
        overlap = len(query_terms & sentence_terms)
        score += overlap * 15

                                                                                    
        for term in query_terms:
            if len(term) > 3 and term in sentence_lower:
                score += 5

                  
        if position < total * 0.15:          
            score += 6
        elif position > total * 0.85:          
            score += 4

                                                                                
        numbers = _NUMBER_PATTERN.findall(sentence)
        if numbers:
            score += len(numbers) * 3                    

                                           
        if _DATE_PATTERN.search(sentence):
            score += 5

                                     
        sentence_words = set(sentence_lower.split())
        key_overlap = len(KEY_WORDS & sentence_words)
        score += key_overlap * 2

                               
        proper_nouns = sum(1 for word in sentence.split() if word and word[0].isupper())
        score += min(proper_nouns, 5)            

        return score

    def prioritize_search_results(
        self,
        results: List[Dict],
        query: str,
        max_results: int = 10
    ) -> List[Dict]:



        query_terms = self._extract_query_terms(query)

        scored_results = []
        for result in results:
            score = self._score_search_result(result, query_terms)
            scored_results.append((score, result))

               
        scored_results.sort(reverse=True, key=lambda x: x[0])

                            
        return [result for _, result in scored_results[:max_results]]

    def _score_search_result(
        self,
        result: Dict,
        query_terms: set
    ) -> float:





        score = 0.0

                               
        title = result.get('title', '').lower()
        title_terms = set(title.split())
        title_overlap = len(query_terms & title_terms)
        score += title_overlap * 4

                     
        snippet = result.get('snippet', '').lower()
        snippet_terms = set(snippet.split())
        snippet_overlap = len(query_terms & snippet_terms)
        score += snippet_overlap * 2

                                               
        for term in query_terms:
            if len(term) > 3 and term in title:
                score += 3

                
        url = result.get('url', '').lower()
        for domain in AUTHORITY_DOMAINS:
            if domain in url:
                score += 5
                break

        return score

    def adaptive_truncate(
        self,
        content: str,
        max_length: int,
        preserve_start: bool = True
    ) -> str:





        if len(content) <= max_length:
            return content

        if preserve_start:
            return content[:max_length] + "..."
        else:
                                                                
            start_ratio = 0.6                        
            start_len = int(max_length * start_ratio)
            end_len = max_length - start_len - 10                           
            return content[:start_len] + "\n...\n" + content[-end_len:]
