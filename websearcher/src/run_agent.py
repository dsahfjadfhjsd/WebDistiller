

   

import asyncio
import json
import argparse
import hashlib
import re
from typing import Dict, List, Optional
from openai import AsyncOpenAI

from .tools.tool_manager import ToolManager
from .memory.memory_manager import MemoryManager
from .memory.memory_folding import fold_memory
from .memory.extraction_intent import ExtractionIntent, DistilledObservation, synthesize_extraction_intent, synthesize_intent_with_llm
from .prompts.system_prompts import get_system_prompt, SPECIAL_MARKERS
from .utils.text_utils import extract_between, extract_answer, count_tokens, extract_json_content
from .utils.difficulty_estimator import DifficultyEstimator


async def generate_response(
    client: AsyncOpenAI,
    messages: List[Dict[str, str]],
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    stop: Optional[List[str]] = None
) -> str:
    """Generate response from LLM, handling both standard and DeepSeek-R1 formats."""
    for attempt in range(3):
        try:
            safe_messages = messages if messages else [{"role": "user", "content": " "}]
            response = await client.chat.completions.create(
                model=model_name,
                messages=safe_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop or []
            )
            message = response.choices[0].message
            content = message.content or ""

            reasoning_content = None

            if hasattr(message, 'model_extra') and message.model_extra:
                reasoning_content = message.model_extra.get('reasoning_content')

            if not reasoning_content:
                reasoning_content = getattr(message, 'reasoning_content', None)

            if reasoning_content:
                if content:
                    return f"{reasoning_content}\n\n{content}"
                else:
                    return reasoning_content

            return content
        except Exception as e:
            if attempt == 2:
                print(f"Error generating response after 3 attempts: {e}")
                return ""
            await asyncio.sleep(1)
    return ""


def _get_math_tool_hint(question: str) -> Optional[str]:
    q = question.lower()
    has_number = bool(re.search(r"\d", q))
    simple_markers = [
        "%", "percent", "percentage", "round", "rounded", "nearest",
        "round up", "round down", "ceil", "floor", "percentile",
        "取整", "四舍五入", "向上取整", "向下取整", "百分比", "百分", "比例"
    ]
    complex_markers = [
        "probability", "expected", "expectation", "optimize", "maximize",
        "minimize", "derivative", "integral", "equation", "solve for",
        "system of", "matrix", "variance", "distribution", "logarithm",
        "geometry", "trigonometry"
    ]
    simple = has_number and any(marker in q for marker in simple_markers)
    complex_math = any(marker in q for marker in complex_markers)

    if simple and not complex_math:
        return "This looks like a direct calculation. Use `calculator` first, then apply rounding rules."
    if has_number and complex_math:
        return "This appears to require multi-step reasoning; use `solve_math` to plan + compute."
    return None


async def reasoning_loop(
    question: str,
    main_client: AsyncOpenAI,
    aux_client: AsyncOpenAI,
    tool_manager: ToolManager,
    memory_manager: MemoryManager,
    args: argparse.Namespace,
    markers: Dict[str, str]
) -> Dict:


       
                     
    tool_manager.set_question_context(question)
    
                                  
    difficulty_info = DifficultyEstimator.get_difficulty_info(question)
    difficulty = difficulty_info['difficulty']
    suggested_iterations = difficulty_info['max_iterations']

    print(f"\n{'='*80}")
    print(f"Question Difficulty: {difficulty.upper()} (score: {difficulty_info['score']})")
    print(f"Suggested iterations: {suggested_iterations}")
    print(f"{'='*80}\n")

                                                                                                           
    if (not hasattr(args, 'max_iterations')) or (args.max_iterations is None) or (args.max_iterations < suggested_iterations):
        args.max_iterations = suggested_iterations

                          
    tool_definitions = tool_manager.get_tool_definitions()

                             
    base_system_prompt = get_system_prompt(
        question=question,
        tool_definitions=tool_definitions,
        include_examples=True,
        markers=markers
    )

    math_hint = _get_math_tool_hint(question)
    if math_hint:
        system_prompt = f"{base_system_prompt}\n\n## Math Tool Hint\n{math_hint}"
    else:
        system_prompt = base_system_prompt
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Start."}
    ]

    output_parts = []

                      
    state = {
        "iteration": 0,
        "action_count": 0,
        "tool_call_count": 0,
        "total_tokens": count_tokens(system_prompt) + count_tokens("Start."),
        "executed_tool_calls": set(),
        "executed_tool_call_keys": set(),
        "finished": False,
        "interactions": [],
        "consecutive_duplicates": 0,
        "empty_response_count": 0,
        "finalization_active": False,
        "finalization_attempts": 0,
        "steps_since_fold": 0,
        "accumulated_intents": [],
    }

                   
    MAX_ACTION_LIMIT = getattr(args, 'max_action_limit', 50)
    MAX_TOKENS = getattr(args, 'max_context_tokens', None)
    if not MAX_TOKENS:
        MAX_TOKENS = getattr(args, 'max_tokens', 50000)
    fold_threshold = getattr(args, 'fold_threshold', 0.75)
    if fold_threshold is None or fold_threshold <= 0 or fold_threshold > 1:
        fold_threshold = 0.75
    cooldown_steps = getattr(args, 'cooldown_steps', 3)
    tail_length = getattr(args, 'tail_length', 4)
    milestone_trigger = getattr(args, 'milestone_trigger', True)

    async def _apply_memory_fold(reason: str) -> bool:
        nonlocal system_prompt, messages
        if not memory_manager.can_fold():
            return False
        if not (args.enable_factual_memory or args.enable_procedural_memory or args.enable_experiential_memory):
            return False

        if state["steps_since_fold"] > 0 and state["steps_since_fold"] < cooldown_steps:
            print(f"Cooldown active ({state['steps_since_fold']}/{cooldown_steps}), skipping fold")
            return False

        print(f"Memory folding ({reason})...")
        tool_call_history = memory_manager.get_tool_call_history()
        if args.experiential_memory_max_entries is not None:
            if args.experiential_memory_max_entries <= 0:
                tool_call_history = []
            else:
                tool_call_history = tool_call_history[-args.experiential_memory_max_entries:]
        fold_client = main_client if getattr(args, 'single_model', False) else aux_client
        fold_model_name = args.reasoner_model_name if getattr(args, 'single_model', False) else args.manager_model_name
        fold_temperature = args.reasoner_temperature if getattr(args, 'single_model', False) else args.manager_temperature
        fold_max_tokens = args.reasoner_max_tokens if getattr(args, 'single_model', False) else args.manager_max_tokens

        factual_memory, procedural_memory, experiential_memory, strategy_reflection = await fold_memory(
            client=fold_client,
            model_name=fold_model_name,
            question=question,
            current_output="".join(output_parts),
            tool_call_history=tool_call_history,
            available_tools=tool_definitions,
            temperature=fold_temperature,
            max_tokens=fold_max_tokens,
            enable_reflection=True,
            enable_factual=args.enable_factual_memory,
            enable_procedural=args.enable_procedural_memory,
            enable_experiential=args.enable_experiential_memory,
            extraction_intents=state["accumulated_intents"]
        )

        memory_manager.add_fold(factual_memory, procedural_memory, experiential_memory)

        try:
            factual_data = json.loads(factual_memory) if factual_memory.strip().startswith('{') else {}
            for fact in factual_data.get("facts_discovered", []):
                sources = fact.get("sources", [])
                source_url = sources[0] if sources else ""
                domain = ""
                if source_url:
                    from urllib.parse import urlparse
                    try:
                        domain = urlparse(source_url).netloc
                    except Exception:
                        domain = source_url
                memory_manager.add_fact(
                    entity=fact.get("entity", ""),
                    attr=fact.get("attribute", ""),
                    value=fact.get("value", ""),
                    url=source_url,
                    domain=domain,
                    snippet=str(fact.get("value", ""))
                )
        except Exception:
            pass

        try:
            proc_data = json.loads(procedural_memory) if procedural_memory.strip().startswith('{') else {}
            for trace in proc_data.get("decision_trace", []):
                memory_manager.add_procedural_node(
                    subgoal=trace.get("subgoal", ""),
                    intent_family="general",
                    action_type=trace.get("action", ""),
                    outcome=trace.get("outcome", "")
                )
        except Exception:
            pass

        state["interactions"].append({
            "type": "thought_folding",
            "factual_memory": factual_memory,
            "procedural_memory": procedural_memory,
            "experiential_memory": experiential_memory,
            "strategy_reflection": strategy_reflection
        })

        state["steps_since_fold"] = 0

        current_subgoal = question
        try:
            proc_data = json.loads(procedural_memory) if procedural_memory.strip().startswith('{') else {}
            current_subgoal = proc_data.get("current_subgoal", question)
        except Exception:
            pass

        tail_messages = messages[-tail_length:] if len(messages) > tail_length else messages[1:]
        context_pack = memory_manager.build_context_pack(
            subgoal=current_subgoal,
            m=10,
            r=6,
            tail=tail_messages,
            h=tail_length
        )

        memory_text = f"Memory Summary:\n\nFactual Memory (M^F): {factual_memory}\n\nProcedural Memory (M^P): {procedural_memory}\n\nExperiential Memory (M^E): {experiential_memory}"
        if strategy_reflection:
            memory_text += f"\n\nStrategy: {strategy_reflection}"

        display_memory = context_pack if context_pack else memory_text

        system_prompt = get_system_prompt(
            question=question,
            tool_definitions=tool_definitions,
            memory_summary=display_memory,
            include_examples=True,
            markers=markers
        )
        if math_hint:
            system_prompt = f"{system_prompt}\n\n## Math Tool Hint\n{math_hint}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Start."}
        ]
        output_parts.append(
            f"\n\n{markers['BEGIN_MEMORY_SUMMARY']}\n{memory_text}\n{markers['END_MEMORY_SUMMARY']}\n\n"
        )
        state["total_tokens"] = count_tokens(system_prompt) + count_tokens("Start.")
        preview = memory_text if len(memory_text) <= 1200 else memory_text[:1200] + "..."
        print(f"\n[Memory Summary Preview]\n{preview}\n")
        print(f"Memory folded ({memory_manager.fold_count}/{memory_manager.max_folds})")
        return True

    def _recount_tokens() -> int:
        return sum(count_tokens(msg.get("content", "")) for msg in messages)


    def _trim_messages(keep_last: int = 8) -> None:
        nonlocal messages
        if not messages:
            return
        system_msg = messages[0] if messages[0].get("role") == "system" else None
        tail = messages[1:] if system_msg else messages
        if len(tail) > keep_last:
            tail = tail[-keep_last:]
        messages = [system_msg] + tail if system_msg else tail

    async def _force_finalize(reason: str, max_tokens: Optional[int] = None) -> str:
        nonlocal messages, output_parts
        msg = (
            "\n\n<system_message>"
            f"Finalization mode: {reason}. Tool calls are disabled. "
            "You MUST provide your best final answer now based on what you already have. "
            "Return the answer in \\boxed{...} format."
            "</system_message>\n\n"
        )
        messages.append({"role": "system", "content": msg})
        output_parts.append(msg)
        state["total_tokens"] += count_tokens(msg)
        stop = [markers["BEGIN_TOOL_CALL"]]
        return await generate_response(
            client=main_client,
            messages=messages,
            model_name=args.reasoner_model_name,
            temperature=0.2,
            max_tokens=max_tokens or args.finalization_max_tokens,
            stop=stop
        )

    async def _detect_milestone(
        reasoning_context: str,
        current_subgoal: str = "",
        action_type: str = "",
        action_args: dict = None
    ) -> bool:
        """Milestone-based folding trigger G_φ^Milestone(· | q, τ_t, a_t, H, M).

        Uses the Manager model to detect subgoal completion, query pivot,
        or failure branch — triggering a fold if detected.

        Args:
            reasoning_context: Recent interaction history H (text)
            current_subgoal: Current subgoal τ_t
            action_type: Last action type a_t
            action_args: Last action arguments
        """
        if not milestone_trigger:
            return False
        if not aux_client:
            return False

        try:
            recent_context = reasoning_context[-3000:] if len(reasoning_context) > 3000 else reasoning_context

            memory_summary = memory_manager.get_memory_summary()
            memory_snippet = memory_summary[:1000] if memory_summary else "No memory yet."

            prompt = f"""Analyze the reasoning trajectory and determine if a milestone event has occurred.

## Original Question
{question}

## Current Subgoal
{current_subgoal or question}

## Last Action
Type: {action_type or 'N/A'}
Args: {json.dumps(action_args or {}, ensure_ascii=False)[:500]}

## Memory State (summary)
{memory_snippet}

## Recent Interaction History
{recent_context}

## Milestone Events (answer YES if any apply)
1. A subgoal has been COMPLETED (information gathered, verified)
2. A PIVOT has occurred (agent switched to a different search strategy)
3. A FAILURE BRANCH was encountered (dead end, need to try different approach)

Answer with ONLY "YES" or "NO" followed by a brief reason (1 sentence).
Example: "YES: Subgoal of finding the population data was completed."
Example: "NO: Still pursuing the same search strategy."

Answer:"""

            fold_client = main_client if getattr(args, 'single_model', False) else aux_client
            fold_model_name = args.reasoner_model_name if getattr(args, 'single_model', False) else args.manager_model_name

            response = await generate_response(
                client=fold_client,
                messages=[{"role": "user", "content": prompt}],
                model_name=fold_model_name,
                temperature=0.3,
                max_tokens=100
            )

            is_milestone = response.strip().upper().startswith("YES")
            if is_milestone:
                print(f"Milestone detected: {response.strip()}")
            return is_milestone

        except Exception as e:
            print(f"Milestone detection failed: {e}")
            return False

    print(f"Question: {question[:200]}...")
    print(f"Limits: iterations={args.max_iterations}, actions={MAX_ACTION_LIMIT}")

    while not state["finished"] and state["iteration"] < args.max_iterations:
        state["iteration"] += 1
        iteration = state["iteration"]
        print(f"\n--- Iteration {iteration} (Actions: {state['action_count']}/{MAX_ACTION_LIMIT}) ---")

                            
        if state["action_count"] >= MAX_ACTION_LIMIT:
            limit_msg = f"\n\n<system_message>Action limit reached ({MAX_ACTION_LIMIT}). Please provide your final answer now.</system_message>\n\n"
            messages.append({"role": "system", "content": limit_msg})
            output_parts.append(limit_msg)
            state["total_tokens"] += count_tokens(limit_msg)

            response = await generate_response(
                client=main_client,
                messages=messages,
                model_name=args.reasoner_model_name,
                temperature=0.3,
                max_tokens=2000
            )
            output_parts.append(response)
            state["finished"] = True
            break

                           
        if state["total_tokens"] >= MAX_TOKENS:
            if await _apply_memory_fold("token_limit"):
                continue
            limit_msg = "\n\n<system_message>Token limit reached. Please provide your final answer immediately.</system_message>\n\n"
            messages.append({"role": "system", "content": limit_msg})
            output_parts.append(limit_msg)
            state["total_tokens"] += count_tokens(limit_msg)

            response = await generate_response(
                client=main_client,
                messages=messages,
                model_name=args.reasoner_model_name,
                temperature=0.3,
                max_tokens=1000
            )
            output_parts.append(response)
            state["finished"] = True
            break

        if (
            memory_manager.can_fold()
            and fold_threshold
            and state["total_tokens"] >= MAX_TOKENS * fold_threshold
        ):
            if await _apply_memory_fold("pre_call"):
                continue
            _trim_messages(keep_last=6)
            state["total_tokens"] = _recount_tokens()

        remaining = args.max_iterations - iteration
        if not state["finalization_active"] and remaining <= args.finalization_trigger_remaining:
            state["finalization_active"] = True
            print("Entering finalization mode...")
            pre_msg = "\n\n<system_message>Finalization mode: Please finalize now with no further tool calls.</system_message>\n\n"
            messages.append({"role": "system", "content": pre_msg})
            output_parts.append(pre_msg)
            state["total_tokens"] += count_tokens(pre_msg)

        stop_sequences = [markers["END_TOOL_CALL"]]

        response = await generate_response(
            client=main_client,
            messages=messages,
            model_name=args.reasoner_model_name,
            temperature=args.reasoner_temperature,
            max_tokens=args.reasoner_max_tokens,
            stop=stop_sequences
        )

        if response and markers["BEGIN_TOOL_CALL"] in response and markers["END_TOOL_CALL"] not in response:
            response = response + markers["END_TOOL_CALL"]

        if not response:
            state["empty_response_count"] += 1
            print("Empty response, retrying...")
            if state["empty_response_count"] >= 3:
                if await _apply_memory_fold("empty_response"):
                    state["empty_response_count"] = 0
                    continue
                _trim_messages(keep_last=6)
                warn_msg = "\n\n<system_message>Empty response received. Continue and provide output.</system_message>\n\n"
                messages.append({"role": "system", "content": warn_msg})
                state["total_tokens"] = _recount_tokens()
            await asyncio.sleep(0.5)
            continue
        state["empty_response_count"] = 0

        if state["finalization_active"] and markers["BEGIN_TOOL_CALL"] in response:
            state["finalization_attempts"] += 1
            print("Tool call detected during finalization; forcing answer...")
            response = await _force_finalize("Tool calls disabled during finalization")

        print(f"Response: {response[:150]}...")

        output_parts.append(response)
        messages.append({"role": "assistant", "content": response})
        state["total_tokens"] += count_tokens(response)

                             
        if state["finalization_active"] and markers["BEGIN_TOOL_CALL"] in response:
            warn = "\n\n<system_message>Tool calls are disabled in finalization mode. Provide the final answer now.</system_message>\n\n"
            messages.append({"role": "system", "content": warn})
            output_parts.append(warn)
            state["total_tokens"] += count_tokens(warn)
            continue

        if markers["BEGIN_TOOL_CALL"] in response:

            tool_call_text = extract_between(
                response,
                markers["BEGIN_TOOL_CALL"],
                markers["END_TOOL_CALL"]
            )

            def _parse_tool_call_json(raw_text: str) -> Optional[Dict]:
                if not raw_text:
                    return None
                try:
                    return json.loads(raw_text)
                except json.JSONDecodeError:
                    pass

                cleaned = extract_json_content(raw_text)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass

                brace_idx = raw_text.find('{')
                if brace_idx != -1:
                    try:
                        decoder = json.JSONDecoder()
                        obj, _ = decoder.raw_decode(raw_text[brace_idx:])
                        return obj
                    except Exception:
                        return None
                return None

            if tool_call_text:
                try:
                    tool_call = _parse_tool_call_json(tool_call_text)
                    if not tool_call:
                        raise json.JSONDecodeError("Unable to parse tool call JSON", tool_call_text, 0)

                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("arguments", {})

                    dedup_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"

                    is_duplicate = False
                    if dedup_key in state["executed_tool_call_keys"]:
                        is_duplicate = True
                    elif tool_call_text in state["executed_tool_calls"]:
                        is_duplicate = True

                    if is_duplicate:
                        state["consecutive_duplicates"] += 1

                        if state["consecutive_duplicates"] >= 3:
                            force_answer_msg = f"\n\n{markers['BEGIN_TOOL_RESULT']}CRITICAL: You have repeated the same failed action 3 times. This is not productive.\n\nYou MUST provide your best answer NOW based on the information you already have. Use \\boxed{{your answer}} format.{markers['END_TOOL_RESULT']}\n\n"
                            messages.append({"role": "system", "content": force_answer_msg})
                            output_parts.append(force_answer_msg)
                            state["total_tokens"] += count_tokens(force_answer_msg)
                            print("Too many consecutive duplicates - forcing answer...")

                            response = await generate_response(
                                client=main_client,
                                messages=messages,
                                model_name=args.reasoner_model_name,
                                temperature=0.3,
                                max_tokens=2000
                            )
                            output_parts.append(response)
                            state["finished"] = True
                            break

                        dedup_msg = f"\n\n{markers['BEGIN_TOOL_RESULT']}ERROR: You already tried this exact tool call. It did not work. You MUST try a completely different approach:\n- Use different tool parameters\n- Break the problem into smaller steps\n- Try a different strategy entirely\n\nDo NOT repeat the same action. (Attempt {state['consecutive_duplicates']}/3){markers['END_TOOL_RESULT']}\n\n"
                        messages.append({"role": "system", "content": dedup_msg})
                        output_parts.append(dedup_msg)
                        state["total_tokens"] += count_tokens(dedup_msg)
                        print(f"Duplicate tool call detected ({state['consecutive_duplicates']}/3) - prompting for different approach...")
                        continue

                    state["consecutive_duplicates"] = 0

                    if not tool_name:
                        error_msg = "\n\n<system_message>Error: Missing tool name.</system_message>\n\n"
                        messages.append({"role": "system", "content": error_msg})
                        output_parts.append(error_msg)
                        state["total_tokens"] += count_tokens(error_msg)
                        continue

                    required_args = {
                        "web_search": ["query"],
                        "browse_pages": ["urls"],
                        "click_link": ["url"],
                        "execute_python_code": ["code"],
                        "process_file": ["file_name"],
                        "solve_math": ["problem"],
                        "calculator": ["expression"],
                        "download_file": ["url"]
                    }

                    if tool_name in required_args:
                        missing = [a for a in required_args[tool_name] if a not in tool_args or not tool_args[a]]
                        if missing:
                            error_msg = f"\n\n<system_message>Missing required args for {tool_name}: {missing}</system_message>\n\n"
                            messages.append({"role": "system", "content": error_msg})
                            output_parts.append(error_msg)
                            state["total_tokens"] += count_tokens(error_msg)
                            continue

                    print(f"Calling: {tool_name}")
                    state["action_count"] += 1
                    state["steps_since_fold"] += 1

                    reasoning_context = "".join(output_parts[-5000:]) if len(output_parts) > 0 else ""

                    current_subgoal = question
                    if memory_manager.procedural_memory:
                        last_node = memory_manager.procedural_memory[-1]
                        if last_node.subgoal:
                            current_subgoal = last_node.subgoal
                    if tool_args.get("goal"):
                        current_subgoal = tool_args["goal"]
                    elif tool_args.get("intent"):
                        current_subgoal = tool_args["intent"]
                    elif tool_args.get("query") and tool_name == "web_search":
                        current_subgoal = tool_args["query"]

                    extraction_intent = None
                    if getattr(args, 'extraction_intent_enabled', True):
                        if getattr(args, 'use_llm_synthesis', False) and aux_client:
                            extraction_intent = await synthesize_intent_with_llm(
                                client=aux_client,
                                model_name=args.manager_model_name,
                                query=question,
                                subgoal=current_subgoal,
                                action_type=tool_name,
                                action_args=tool_args,
                                reasoning_context=reasoning_context[-3000:]
                            )
                        else:
                            extraction_intent = synthesize_extraction_intent(
                                query=question,
                                subgoal=current_subgoal,
                                action_type=tool_name,
                                action_args=tool_args
                            )
                        state["accumulated_intents"].append(extraction_intent.to_dict())
                        print(f"Intent: {extraction_intent.intent_family} | Target: {extraction_intent.target[:80]}")

                    tool_result = await tool_manager.call_tool(
                        tool_name,
                        tool_args,
                        reasoning_context=reasoning_context,
                        extraction_intent=extraction_intent
                    )

                    state["tool_call_count"] += 1

                    state["executed_tool_calls"].add(tool_call_text)
                    state["executed_tool_call_keys"].add(dedup_key)

                    memory_manager.add_interaction({
                        "type": "tool_call",
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "result": tool_result,
                        "iteration": iteration,
                        "extraction_intent": extraction_intent.to_dict() if extraction_intent else None
                    })

                    state["interactions"].append({
                        "type": "tool_call",
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "result": tool_result[:500] if isinstance(tool_result, str) and len(tool_result) > 500 else tool_result,
                        "iteration": iteration,
                        "extraction_intent": extraction_intent.to_dict() if extraction_intent else None
                    })

                    distillation_candidates = []
                    if (
                        extraction_intent
                        and getattr(args, 'auto_summarize_enabled', True)
                        and isinstance(tool_result, str)
                        and len(tool_result) > getattr(args, 'auto_summarize_threshold', 10000)
                    ):
                        print(f"Active Cognitive Distillation ({len(tool_result)} chars, intent: {extraction_intent.intent_family})...")
                        try:
                            distilled_text, observation = await tool_manager.distill_with_intent(
                                raw_content=tool_result,
                                intent=extraction_intent,
                                question=question
                            )
                            tool_result = distilled_text
                            distillation_candidates = observation.candidates
                            print(f"Distilled to {len(tool_result)} chars (ratio: {observation.compression_ratio:.1%})")
                        except Exception as e:
                            print(f"Distillation failed: {e}, using original content")
                            if len(tool_result) > 20000:
                                tool_result = tool_result[:20000] + f"\n\n[Content truncated from {len(tool_result)} chars]"
                    elif (
                        not extraction_intent
                        and isinstance(tool_result, str)
                        and len(tool_result) > 20000
                    ):
                        tool_result = tool_result[:20000] + f"\n\n[Content truncated from {len(tool_result)} chars]"

                    for cand in distillation_candidates:
                        evidence = cand.get("evidence", {})
                        memory_manager.add_fact(
                            entity=cand.get("entity", ""),
                            attr=cand.get("attr", ""),
                            value=cand.get("value", ""),
                            url=evidence.get("url", ""),
                            domain=evidence.get("domain", ""),
                            snippet=evidence.get("snippet", "")
                        )

                    outcome = tool_result[:200] if isinstance(tool_result, str) and len(tool_result) > 200 else str(tool_result)[:200]
                    intent_family = extraction_intent.intent_family if extraction_intent else "general"
                    memory_manager.add_procedural_node(
                        subgoal=current_subgoal,
                        intent_family=intent_family,
                        action_type=tool_name,
                        outcome=outcome,
                        produced_fact_keys=[],
                        is_failure=isinstance(tool_result, str) and "error" in tool_result.lower()[:100]
                    )

                    result_text = f"\n\n{markers['BEGIN_TOOL_RESULT']}\n{tool_result}\n{markers['END_TOOL_RESULT']}\n\n"
                    tool_message = "[TOOL_RESULT - UNTRUSTED DATA]\n" + result_text.strip()
                    tool_call_id = hashlib.md5(dedup_key.encode("utf-8")).hexdigest()
                    messages.append({
                        "role": "tool",
                        "name": tool_name,
                        "tool_call_id": tool_call_id,
                        "content": tool_message
                    })
                    output_parts.append(result_text)
                    state["total_tokens"] += count_tokens(tool_message)

                    print(f"Result: {tool_result[:100] if isinstance(tool_result, str) else str(tool_result)[:100]}...")

                    if (
                        milestone_trigger
                        and memory_manager.can_fold()
                        and state["steps_since_fold"] >= cooldown_steps
                        and state["tool_call_count"] >= 3
                    ):
                        recent_reasoning = "".join(output_parts[-5000:])
                        if await _detect_milestone(
                            recent_reasoning,
                            current_subgoal=current_subgoal,
                            action_type=tool_name,
                            action_args=tool_args
                        ):
                            if await _apply_memory_fold("milestone"):
                                continue

                    if (
                        memory_manager.can_fold()
                        and fold_threshold
                        and state["total_tokens"] >= MAX_TOKENS * fold_threshold
                    ):
                        if await _apply_memory_fold("auto"):
                            continue
                    continue

                except json.JSONDecodeError as e:
                    error_msg = f"\n\n<system_message>JSON parse error: {e}. Use valid JSON format.</system_message>\n\n"
                    messages.append({"role": "system", "content": error_msg})
                    output_parts.append(error_msg)
                    state["total_tokens"] += count_tokens(error_msg)
                    continue
                except Exception as e:
                    error_msg = f"\n\n<system_message>Tool error: {e}</system_message>\n\n"
                    messages.append({"role": "system", "content": error_msg})
                    output_parts.append(error_msg)
                    state["total_tokens"] += count_tokens(error_msg)
                    continue


        if markers["FOLD_MEMORY"] in response:
            if await _apply_memory_fold("marker"):
                continue
            fold_limit_msg = f"\n\n<system_message>Fold limit reached ({memory_manager.max_folds}). Continue with current information.</system_message>\n\nHmm, I've already"
            messages.append({"role": "system", "content": fold_limit_msg})
            output_parts.append(fold_limit_msg)
            state["total_tokens"] += count_tokens(fold_limit_msg)
            continue


        has_tool_marker = markers["BEGIN_TOOL_CALL"] in response
        has_fold_marker = markers["FOLD_MEMORY"] in response

        if not has_tool_marker and not has_fold_marker:
            full_output = "".join(output_parts)
            extracted_answer = extract_answer(full_output, mode='qa', markers=markers)

            if extracted_answer:
                print(f"Answer found: {extracted_answer}")
                return {
                    "answer": extracted_answer,
                    "reasoning": full_output,
                    "iterations": state["iteration"],
                    "tool_calls": state["tool_call_count"],
                    "actions": state["action_count"],
                    "memory_folds": memory_manager.fold_count,
                    "total_tokens": state["total_tokens"],
                    "interactions": state["interactions"],
                    "success": True
                }
            else:
                if (
                    memory_manager.can_fold()
                    and fold_threshold
                    and state["total_tokens"] >= MAX_TOKENS * fold_threshold
                ):
                    if await _apply_memory_fold("auto"):
                        continue
                print("No answer found, prompting to continue...")
                continue_msg = (
                    "\n\nPlease provide your final answer in \\boxed{} format. "
                    "Include a concrete answer (not empty or only punctuation/backticks).\n\n"
                )
                messages.append({"role": "system", "content": continue_msg})
                output_parts.append(continue_msg)
                state["total_tokens"] += count_tokens(continue_msg)
                continue

                                      
        answer = extract_answer(response, mode='qa', markers=markers)
        if answer:
            print(f"Final answer: {answer}")
            return {
                "answer": answer,
                "reasoning": "".join(output_parts),
                "iterations": state["iteration"],
                "tool_calls": state["tool_call_count"],
                "actions": state["action_count"],
                "memory_folds": memory_manager.fold_count,
                "total_tokens": state["total_tokens"],
                "interactions": state["interactions"],
                "success": True
            }

                            
    print(f"Max iterations ({args.max_iterations}) reached")
    full_output = "".join(output_parts)
    final_answer = extract_answer(full_output, mode='qa', markers=markers)

                                                                                                     
    if not final_answer:
        response = await _force_finalize("Max iterations reached")
        output_parts.append(response)
        full_output = "".join(output_parts)
        final_answer = extract_answer(full_output, mode='qa', markers=markers) or extract_answer(response, mode='qa', markers=markers)
        if not final_answer:
            tail = response.strip().splitlines()
            if tail:
                final_answer = tail[-1].strip()
            else:
                final_answer = "Unknown"

    return {
        "answer": final_answer,
        "reasoning": full_output,
        "iterations": state["iteration"],
        "tool_calls": state["tool_call_count"],
        "actions": state["action_count"],
        "memory_folds": memory_manager.fold_count,
        "total_tokens": state["total_tokens"],
        "interactions": state["interactions"],
        "success": final_answer not in ("No answer found", "Unknown")
    }


async def run_single_question(
    question: str,
    config_path: str = "config/config.yaml",
    max_iterations: int = 20,
    max_context_tokens: Optional[int] = 30000
) -> Dict:


       
    import yaml
    from pathlib import Path
    
                        
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if max_context_tokens is None or max_context_tokens <= 0:
        max_context_tokens = config.get('memory', {}).get('max_context_tokens', 30000)

    reasoning_cfg = config.get('reasoning', {})
    finalization_cfg = reasoning_cfg.get('finalization', {}) if isinstance(reasoning_cfg, dict) else {}
    markers_cfg = reasoning_cfg.get('markers', {}) if isinstance(reasoning_cfg, dict) else {}
    markers = dict(SPECIAL_MARKERS)
    marker_map = {
        "begin_tool_call": "BEGIN_TOOL_CALL",
        "end_tool_call": "END_TOOL_CALL",
        "begin_tool_result": "BEGIN_TOOL_RESULT",
        "end_tool_result": "END_TOOL_RESULT",
        "begin_think": "BEGIN_THINK",
        "end_think": "END_THINK",
        "fold_memory": "FOLD_MEMORY",
        "begin_memory_summary": "BEGIN_MEMORY_SUMMARY",
        "end_memory_summary": "END_MEMORY_SUMMARY"
    }
    for cfg_key, marker_key in marker_map.items():
        value = markers_cfg.get(cfg_key)
        if value:
            markers[marker_key] = value
    
                    
    reasoner_cfg = config['model'].get('reasoner') or config['model'].get('main_model')
    manager_cfg = config['model'].get('manager') or config['model'].get('auxiliary_model')

    main_client = AsyncOpenAI(
        api_key=reasoner_cfg['api_key'],
        base_url=reasoner_cfg['api_base']
    )

    aux_client = AsyncOpenAI(
        api_key=manager_cfg['api_key'],
        base_url=manager_cfg['api_base']
    )

    try:
                                                                               
                                                                       
        serper_keys = config['tools'].get('serper_api_keys')
        serper_key = config['tools'].get('serper_api_key')
        
        python_timeout = (
            config.get('tools', {})
            .get('python_executor', {})
            .get('timeout', 30)
        )
        download_cfg = config.get('tools', {}).get('download', {})
        download_large_mb = download_cfg.get('large_file_threshold_mb', 20)
        download_segment_mb = download_cfg.get('segment_size_mb', 5)
        download_max_retries = download_cfg.get('max_retries', 3)

        tool_manager = ToolManager(
            serper_api_keys=serper_keys,
            serper_api_key=serper_key,
            jina_api_key=config['tools'].get('jina_api_key'),
            file_base_dir=config['tools']['file_base_dir'],
            aux_client=aux_client,
            aux_model_name=manager_cfg['name'],
            python_timeout=python_timeout,
            download_large_file_threshold_mb=download_large_mb,
            download_segment_size_mb=download_segment_mb,
            download_max_retries=download_max_retries
        )

        memory_cfg = config.get('memory', {})
        max_folds = memory_cfg.get('max_folds')
        if not max_folds:
            max_folds = memory_cfg.get('factual_memory', {}).get('max_entries', None)
            if not max_folds:
                max_folds = memory_cfg.get('episode_memory', {}).get('max_episodes', 3)
        memory_manager = MemoryManager(max_folds=max_folds)

        class Args:
            def __init__(self):
                self.max_iterations = max_iterations
                self.max_context_tokens = max_context_tokens
                self.fold_threshold = memory_cfg.get('fold_threshold', 0.75)
                self.enable_factual_memory = (
                    memory_cfg.get('factual_memory', {}).get('enabled', True)
                    if 'factual_memory' in memory_cfg
                    else memory_cfg.get('episode_memory', {}).get('enabled', True)
                )
                self.enable_procedural_memory = (
                    memory_cfg.get('procedural_memory', {}).get('enabled', True)
                    if 'procedural_memory' in memory_cfg
                    else memory_cfg.get('working_memory', {}).get('enabled', True)
                )
                self.enable_experiential_memory = (
                    memory_cfg.get('experiential_memory', {}).get('enabled', True)
                    if 'experiential_memory' in memory_cfg
                    else memory_cfg.get('tool_memory', {}).get('enabled', True)
                )
                self.experiential_memory_max_entries = (
                    memory_cfg.get('experiential_memory', {}).get('max_families', None)
                    or memory_cfg.get('tool_memory', {}).get('max_entries', None)
                )
                self.reasoner_model_name = reasoner_cfg['name']
                self.manager_model_name = manager_cfg['name']
                self.reasoner_temperature = reasoner_cfg['temperature']
                self.manager_temperature = manager_cfg['temperature']
                self.reasoner_max_tokens = reasoner_cfg['max_tokens']
                self.manager_max_tokens = manager_cfg['max_tokens']
                self.finalization_trigger_remaining = finalization_cfg.get('trigger_remaining', 2)
                self.finalization_max_attempts = finalization_cfg.get('max_attempts', 2)
                self.finalization_max_tokens = finalization_cfg.get('max_tokens', 1200)
                auto_summarize_cfg = memory_cfg.get('auto_summarize_tool_results', {})
                self.auto_summarize_enabled = auto_summarize_cfg.get('enabled', True)
                self.auto_summarize_threshold = auto_summarize_cfg.get('threshold_chars', 10000)
                self.auto_summarize_max_tokens = auto_summarize_cfg.get('max_summary_tokens', 4000)
                self.cooldown_steps = memory_cfg.get('cooldown_steps', 3)
                self.tail_length = memory_cfg.get('tail_length', 4)
                self.milestone_trigger = memory_cfg.get('milestone_trigger', True)
                ei_cfg = config.get('extraction_intent', {})
                self.extraction_intent_enabled = ei_cfg.get('enabled', True)
                self.use_llm_synthesis = ei_cfg.get('use_llm_synthesis', False)
                ablation_cfg = config.get('ablation', {})
                self.single_model = ablation_cfg.get('single_model', False)

        args = Args()

                            
        result = await reasoning_loop(
            question=question,
            main_client=main_client,
            aux_client=aux_client,
            tool_manager=tool_manager,
            memory_manager=memory_manager,
            args=args,
            markers=markers
        )

        return result
    finally:
                                                          
        await main_client.close()
        await aux_client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run HybridQA Agent")
    parser.add_argument("--question", type=str, required=True, help="Question to answer")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--max-iterations", type=int, default=20, help="Maximum iterations")
    parser.add_argument("--max-context-tokens", type=int, default=30000, help="Maximum context tokens")

    args = parser.parse_args()

    async def main():
        result = await run_single_question(
            question=args.question,
            config_path=args.config,
            max_iterations=args.max_iterations,
            max_context_tokens=args.max_context_tokens
        )

        print(f"\n{'='*80}")
        print("FINAL RESULT")
        print(f"{'='*80}")
        print(f"Answer: {result['answer']}")
        print(f"Success: {result['success']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Tool Calls: {result['tool_calls']}")
        print(f"Memory Folds: {result['memory_folds']}")
        print(f"Total Tokens: {result['total_tokens']}")
        print(f"{'='*80}")

                                                                     
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")