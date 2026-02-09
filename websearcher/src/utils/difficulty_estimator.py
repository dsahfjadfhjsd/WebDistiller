









import re
from typing import Tuple, Dict

                        
_NUMBER_PATTERN = re.compile(r'\d+')

                                              
_COMPARISON_WORDS = frozenset([
    'compare', 'difference', 'versus', 'vs',
    'better', 'worse', 'contrast', 'distinguish', 'between'
])

_MULTI_STEP_WORDS = frozenset([
    'first', 'then', 'after', 'next', 'finally',
    'step', 'process', 'procedure', 'sequence', 'followed'
])

_CALCULATION_WORDS = frozenset([
    'calculate', 'compute', 'sum', 'total',
    'average', 'mean', 'percentage', 'ratio', 'count', 'how many'
])

_ANALYSIS_WORDS = frozenset([
    'analyze', 'explain', 'why', 'how',
    'reason', 'cause', 'effect', 'impact'
])

_RESEARCH_WORDS = frozenset([
    'research', 'find', 'identify', 'locate',
    'discover', 'investigate', 'search', 'look up'
])

_TEMPORAL_WORDS = frozenset([
    'when', 'date', 'year', 'time', 'period',
    'history', 'timeline', 'chronology', 'born', 'died', 'founded'
])

_SPATIAL_WORDS = frozenset([
    'where', 'location', 'place', 'region',
    'country', 'city', 'area', 'geography', 'capital'
])

_FILE_WORDS = frozenset([
    'file', 'attachment', 'document', 'spreadsheet', 'excel',
    'pdf', 'image', 'audio', 'video', 'table', 'data'
])


class DifficultyEstimator:









    @staticmethod
    def estimate_difficulty(question: str) -> Tuple[str, int]:






        features = DifficultyEstimator._extract_features(question)
        score = DifficultyEstimator._calculate_score(features)

                                                         
        if score >= 6:
            return "hard", 30
        elif score >= 3:
            return "medium", 20
        else:
            return "easy", 15

    @staticmethod
    def _extract_features(question: str) -> Dict:

        question_lower = question.lower()
        words = question.split()
        word_set = set(question_lower.split())

        features = {
            'length': len(words),
            'has_math': bool(_NUMBER_PATTERN.search(question)),
            'has_comparison': bool(word_set & _COMPARISON_WORDS),
            'has_multi_step': bool(word_set & _MULTI_STEP_WORDS),
            'has_calculation': bool(word_set & _CALCULATION_WORDS),
            'has_analysis': bool(word_set & _ANALYSIS_WORDS),
            'has_research': bool(word_set & _RESEARCH_WORDS),
            'has_temporal': bool(word_set & _TEMPORAL_WORDS),
            'has_spatial': bool(word_set & _SPATIAL_WORDS),
            'has_file': bool(word_set & _FILE_WORDS),                                   
        }

                                                                                     
        entities = sum(1 for i, w in enumerate(words)
                       if w and w[0].isupper() and len(w) > 1 and i > 0)
        features['num_entities'] = entities

        return features

    @staticmethod
    def _calculate_score(features: Dict) -> int:

        score = 0

                  
        if features['length'] > 40:
            score += 3
        elif features['length'] > 25:
            score += 2
        elif features['length'] > 15:
            score += 1

                  
        if features['num_entities'] > 4:
            score += 3
        elif features['num_entities'] > 2:
            score += 2
        elif features['num_entities'] > 0:
            score += 1

                
        if features['has_comparison']:
            score += 2
        if features['has_multi_step']:
            score += 2
        if features['has_calculation']:
            score += 1
        if features['has_analysis']:
            score += 1
        if features['has_research']:
            score += 1
        if features['has_temporal']:
            score += 1
        if features['has_spatial']:
            score += 1

                                                   
        if features['has_file']:
            score += 2

        return score

    @staticmethod
    def get_difficulty_info(question: str) -> Dict:

        features = DifficultyEstimator._extract_features(question)
        score = DifficultyEstimator._calculate_score(features)
        difficulty, max_iterations = DifficultyEstimator.estimate_difficulty(question)

        return {
            'difficulty': difficulty,
            'score': score,
            'max_iterations': max_iterations,
            'features': features,
            'suggestions': DifficultyEstimator._get_suggestions(difficulty)
        }

    @staticmethod
    def _get_suggestions(difficulty: str) -> Dict:

                                      
        suggestions = {
            'easy': {
                'max_iterations': 15,
                'max_tool_calls': 8,
                'search_depth': 'shallow',
                'description': 'Simple question, 1-2 searches should suffice'
            },
            'medium': {
                'max_iterations': 20,
                'max_tool_calls': 15,
                'search_depth': 'moderate',
                'description': 'Medium complexity, may need multiple searches'
            },
            'hard': {
                'max_iterations': 30,
                'max_tool_calls': 25,
                'search_depth': 'deep',
                'description': 'Complex question requiring deep exploration'
            }
        }
        return suggestions.get(difficulty, suggestions['medium'])
