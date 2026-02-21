


import json


def get_search_intent_instruction(prev_reasoning: str) -> str:

                                                       
    recent = prev_reasoning[-2000:] if len(prev_reasoning) > 2000 else prev_reasoning

    return f"""Based on the reasoning below, what is the search intent?

{recent}

Search intent (1-2 sentences):"""


def get_click_intent_instruction(prev_reasoning: str) -> str:

    recent = prev_reasoning[-2000:] if len(prev_reasoning) > 2000 else prev_reasoning

    return f"""Based on the reasoning below, what information is being looked for on this page?

{recent}

Click intent (1-2 sentences):"""


def get_web_page_reader_instruction(goal: str, document: str) -> str:


                                   
    doc = document[:25000] if len(document) > 25000 else document

    return f"""Extract relevant information from this webpage for the given goal.

## Goal
{goal}

## Webpage Content
{doc}

## Output Format (JSON only)
```json
{{
  "rational": "Why this content is relevant to the goal",
  "evidence": "All relevant facts, numbers, dates, names from the page. Be comprehensive and EXACT - preserve original values.",
  "summary": "Concise summary of key findings"
}}
```

If no relevant information exists, set all fields to indicate this.

Output ONLY the JSON:"""


def get_detailed_web_page_reader_instruction(
    query: str,
    search_intent: str,
    document: str
) -> str:

    doc = document[:25000] if len(document) > 25000 else document

    return f"""Extract ALL information relevant to the search from this webpage.

**Query:** {query}
**Looking for:** {search_intent}

**Content:**
{doc}

**Instructions:**
1. Extract ALL relevant facts, numbers, dates, names, links
2. Preserve EXACT values - do not paraphrase or round numbers
3. Pay attention to units (km, miles, years, %, etc.)
4. If no relevant info, output: "No relevant information"

**Extracted Information:**"""


def get_strategy_reflection_prompt(
    question: str,
    factual_memory: str,
    procedural_memory: str,
    experiential_memory: str
) -> str:


    return f"""Reflect on progress toward answering this question and suggest next steps.

Question: {question}

Factual Memory (M^F): {factual_memory[:1000]}
Procedural Memory (M^P): {procedural_memory[:500]}
Experiential Memory (M^E): {experiential_memory[:500]}

Strategy (2-3 sentences - what should be tried next?):"""


def get_tool_intent_analysis_prompt(
    tool_name: str,
    arguments: dict,
    reasoning_context: str
) -> str:

    recent = reasoning_context[-1500:] if len(reasoning_context) > 1500 else reasoning_context

    return f"""What is this tool call trying to accomplish?

Recent reasoning: {recent}

Tool: {tool_name}
Args: {json.dumps(arguments, ensure_ascii=False)}

Intent (1-2 sentences):"""


def get_tool_response_analysis_prompt(
    tool_call: dict,
    tool_call_intent: str,
    tool_response: str
) -> str:


                                  
    response = tool_response[:20000] if len(tool_response) > 20000 else tool_response

    return f"""Extract ONLY information helpful for the intent from the tool response.

Tool: {tool_call.get('name')}
Intent: {tool_call_intent}

Response:
{response}

**Instructions:**
- Extract only relevant information
- Preserve exact numbers, dates, names
- Do not add explanations

Helpful information:"""
