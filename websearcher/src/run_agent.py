

   

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

            # Handle DeepSeek-R1 format: reasoning_content + content
            # R1 models put chain-of-thought in reasoning_content field
            # It may be in model_extra dict or as direct attribute
            reasoning_content = None

            # Try to get reasoning_content from model_extra first (pydantic v2)
            if hasattr(message, 'model_extra') and message.model_extra:
                reasoning_content = message.model_extra.get('reasoning_content')

            # Fallback: try direct attribute access
            if not reasoning_content:
                reasoning_content = getattr(message, 'reasoning_content', None)

            if reasoning_content:
                # Combine reasoning and content for R1 models
                # The reasoning contains the thinking process which may include tool calls
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
        "finalization_attempts": 0
    }

                   
    MAX_ACTION_LIMIT = getattr(args, 'max_action_limit', 50)
    MAX_TOKENS = getattr(args, 'max_context_tokens', None)
    if not MAX_TOKENS:
        MAX_TOKENS = getattr(args, 'max_tokens', 50000)
    fold_threshold = getattr(args, 'fold_threshold', 0.75)
    if fold_threshold is None or fold_threshold <= 0 or fold_threshold > 1:
        fold_threshold = 0.75

    async def _apply_memory_fold(reason: str) -> bool:
        nonlocal system_prompt, messages
        if not memory_manager.can_fold():
            return False
        if not (args.enable_episode_memory or args.enable_working_memory or args.enable_tool_memory):
            return False

        print(f"Memory folding ({reason})...")
        tool_call_history = memory_manager.get_tool_call_history()
        if args.tool_memory_max_entries is not None:
            if args.tool_memory_max_entries <= 0:
                tool_call_history = []
            else:
                tool_call_history = tool_call_history[-args.tool_memory_max_entries:]
        # Use main model if single_model ablation is enabled
        fold_client = main_client if getattr(args, 'single_model', False) else aux_client
        fold_model_name = args.main_model_name if getattr(args, 'single_model', False) else args.aux_model_name
        fold_temperature = args.main_temperature if getattr(args, 'single_model', False) else args.aux_temperature
        fold_max_tokens = args.main_max_tokens if getattr(args, 'single_model', False) else args.aux_max_tokens
        
        episode_memory, working_memory, tool_memory, strategy_reflection = await fold_memory(
            client=fold_client,
            model_name=fold_model_name,
            question=question,
            current_output="".join(output_parts),
            tool_call_history=tool_call_history,
            available_tools=tool_definitions,
            temperature=fold_temperature,
            max_tokens=fold_max_tokens,
            enable_reflection=True,
            enable_episode=args.enable_episode_memory,
            enable_working=args.enable_working_memory,
            enable_tool=args.enable_tool_memory
        )

        memory_manager.add_fold(episode_memory, working_memory, tool_memory)

        state["interactions"].append({
            "type": "thought_folding",
            "episode_memory": episode_memory,
            "working_memory": working_memory,
            "tool_memory": tool_memory,
            "strategy_reflection": strategy_reflection
        })

        memory_text = f"Memory Summary:\n\nEpisode: {episode_memory}\n\nWorking: {working_memory}\n\nTools: {tool_memory}"
        if strategy_reflection:
            memory_text += f"\n\nStrategy: {strategy_reflection}"

        system_prompt = get_system_prompt(
            question=question,
            tool_definitions=tool_definitions,
            memory_summary=memory_text,
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
            model_name=args.main_model_name,
            temperature=0.2,
            max_tokens=max_tokens or args.finalization_max_tokens,
            stop=stop
        )

    print(f"Question: {question[:200]}...")
    print(f"Limits: iterations={args.max_iterations}, actions={MAX_ACTION_LIMIT}")

    # Reasoning loop
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
                model_name=args.main_model_name,
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
                model_name=args.main_model_name,
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

        # Use stop sequences to stop at first tool call end (WebThinker approach)
        # This ensures only one tool call per iteration
        stop_sequences = [markers["END_TOOL_CALL"]]

        response = await generate_response(
            client=main_client,
            messages=messages,
            model_name=args.main_model_name,
            temperature=args.main_temperature,
            max_tokens=args.main_max_tokens,
            stop=stop_sequences
        )

        # If response was stopped at END_TOOL_CALL, append the marker back
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

        # 始终先把完整响应放入对话历史，后续从中解析出所有工具调用
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

            # Extract the single tool call (stop sequences ensure only one per response)
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
                                model_name=args.main_model_name,
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

                    reasoning_context = "".join(output_parts[-5000:]) if len(output_parts) > 0 else ""

                    tool_result = await tool_manager.call_tool(
                        tool_name,
                        tool_args,
                        reasoning_context=reasoning_context
                    )

                    state["tool_call_count"] += 1

                    state["executed_tool_calls"].add(tool_call_text)
                    state["executed_tool_call_keys"].add(dedup_key)

                    memory_manager.add_interaction({
                        "type": "tool_call",
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "result": tool_result,
                        "iteration": iteration
                    })

                    state["interactions"].append({
                        "type": "tool_call",
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "result": tool_result[:500] if isinstance(tool_result, str) and len(tool_result) > 500 else tool_result,
                        "iteration": iteration
                    })

                    # Auto-summarize large tool results using auxiliary model (or main model if single_model)
                    if (
                        getattr(args, 'auto_summarize_enabled', True)
                        and isinstance(tool_result, str)
                        and len(tool_result) > getattr(args, 'auto_summarize_threshold', 10000)
                    ):
                        use_main_model = getattr(args, 'single_model', False)
                        summary_client = main_client if use_main_model else aux_client

                        if summary_client:
                            model_type = "main model" if use_main_model else "auxiliary model"
                            print(f"📝 Tool result is large ({len(tool_result)} chars), extracting key information with {model_type}...")
                            try:
                                reasoning_context = "".join(output_parts[-3000:]) if len(output_parts) > 0 else ""
                                summary_prompt = f"""Extract the key information from this tool result that is relevant to answering the question.

Question: {question}

Recent reasoning context:
{reasoning_context[-2000:] if reasoning_context else "None"}

Tool: {tool_name}
Tool result (first 20000 chars):
{tool_result[:20000]}

Extract:
1. Key facts, values, dates, names that directly relate to the question
2. Relevant sections or passages (keep original wording for accuracy)
3. Any information that helps answer the question

IMPORTANT:
- Preserve exact values, dates, and names as written
- Extract complete information (don't truncate important details)
- If the question asks for a count or list, extract ALL relevant items
- If the question asks for calculations, extract the raw values needed
- Maintain context around extracted information

Format: Provide a concise summary preserving critical details and accuracy.

Summary:"""

                                max_summary_tokens = getattr(args, 'auto_summarize_max_tokens', 4000)
                                if use_main_model:
                                    summary_response = await summary_client.chat.completions.create(
                                        model=args.main_model_name,
                                        messages=[{"role": "user", "content": summary_prompt}],
                                        temperature=0.3,
                                        max_tokens=min(max_summary_tokens, args.main_max_tokens)
                                    )
                                else:
                                    summary_response = await summary_client.chat.completions.create(
                                        model=args.aux_model_name,
                                        messages=[{"role": "user", "content": summary_prompt}],
                                        temperature=0.3,
                                        max_tokens=min(max_summary_tokens, args.aux_max_tokens)
                                    )
                                summarized = summary_response.choices[0].message.content.strip()
                                if summarized:
                                    original_size = len(tool_result)
                                    model_label = "main model" if use_main_model else "auxiliary model"
                                    tool_result = f"[Summarized from {original_size} chars by {model_label}]\n\n{summarized}"
                                    print(f"✅ Summarized to {len(tool_result)} chars")
                            except Exception as e:
                                print(f"⚠️ Summary extraction failed: {e}, using original content")
                                if len(tool_result) > 20000:
                                    tool_result = tool_result[:20000] + f"\n\n[Content truncated from {len(tool_result)} chars]"

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
    
                    
    main_client = AsyncOpenAI(
        api_key=config['model']['main_model']['api_key'],
        base_url=config['model']['main_model']['api_base']
    )

    aux_client = AsyncOpenAI(
        api_key=config['model']['auxiliary_model']['api_key'],
        base_url=config['model']['auxiliary_model']['api_base']
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
            aux_model_name=config['model']['auxiliary_model']['name'],
            python_timeout=python_timeout,
            download_large_file_threshold_mb=download_large_mb,
            download_segment_size_mb=download_segment_mb,
            download_max_retries=download_max_retries
        )

                               
        memory_cfg = config.get('memory', {})
        max_folds = memory_cfg.get('max_folds')
        if not max_folds:
            max_folds = memory_cfg.get('episode_memory', {}).get('max_episodes', 3)
        memory_manager = MemoryManager(max_folds=max_folds)

                                    
        class Args:
            def __init__(self):
                self.max_iterations = max_iterations
                self.max_context_tokens = max_context_tokens
                self.fold_threshold = memory_cfg.get('fold_threshold', 0.75)
                self.enable_episode_memory = memory_cfg.get('episode_memory', {}).get('enabled', True)
                self.enable_working_memory = memory_cfg.get('working_memory', {}).get('enabled', True)
                self.enable_tool_memory = memory_cfg.get('tool_memory', {}).get('enabled', True)
                self.tool_memory_max_entries = memory_cfg.get('tool_memory', {}).get('max_entries')
                self.main_model_name = config['model']['main_model']['name']
                self.aux_model_name = config['model']['auxiliary_model']['name']
                self.main_temperature = config['model']['main_model']['temperature']
                self.aux_temperature = config['model']['auxiliary_model']['temperature']
                self.main_max_tokens = config['model']['main_model']['max_tokens']
                self.aux_max_tokens = config['model']['auxiliary_model']['max_tokens']
                self.finalization_trigger_remaining = finalization_cfg.get('trigger_remaining', 2)
                self.finalization_max_attempts = finalization_cfg.get('max_attempts', 2)
                self.finalization_max_tokens = finalization_cfg.get('max_tokens', 1200)
                # Auto-summarize configuration
                auto_summarize_cfg = memory_cfg.get('auto_summarize_tool_results', {})
                self.auto_summarize_enabled = auto_summarize_cfg.get('enabled', True)
                self.auto_summarize_threshold = auto_summarize_cfg.get('threshold_chars', 10000)
                self.auto_summarize_max_tokens = auto_summarize_cfg.get('max_summary_tokens', 4000)
                
                # Ablation configuration
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