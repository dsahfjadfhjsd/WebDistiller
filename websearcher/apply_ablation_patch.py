"""
消融实验代码补丁
===============

此脚本用于给现有代码添加消融实验支持。
运行此脚本将自动修改必要的文件。

使用方法:
    python apply_ablation_patch.py

注意: 运行前请备份原始代码！
"""

import re
from pathlib import Path


def patch_run_agent():
    """修改 run_agent.py 添加消融配置支持"""

    file_path = Path("src/run_agent.py")
    content = file_path.read_text(encoding='utf-8')

    # 1. 在 Args 类中添加消融配置
    old_args_end = """                self.auto_summarize_max_tokens = auto_summarize_cfg.get('max_summary_tokens', 4000)

        args = Args()"""

    new_args_end = """                self.auto_summarize_max_tokens = auto_summarize_cfg.get('max_summary_tokens', 4000)

                # Ablation configuration
                ablation_cfg = config.get('ablation', {})
                self.disable_intent_extraction = ablation_cfg.get('disable_intent_extraction', False)
                self.disable_intent_compression = ablation_cfg.get('disable_intent_compression', False)
                self.disable_intent_in_memory = ablation_cfg.get('disable_intent_in_memory', False)
                self.max_concurrent_fetch = ablation_cfg.get('max_concurrent_fetch', 32)
                self.single_model = ablation_cfg.get('single_model', False)

        args = Args()"""

    content = content.replace(old_args_end, new_args_end)

    # 2. 修改 ToolManager 初始化，传入消融配置
    old_tool_manager = """        tool_manager = ToolManager(
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
        )"""

    new_tool_manager = """        # Get ablation config
        ablation_cfg = config.get('ablation', {})

        tool_manager = ToolManager(
            serper_api_keys=serper_keys,
            serper_api_key=serper_key,
            jina_api_key=config['tools'].get('jina_api_key'),
            file_base_dir=config['tools']['file_base_dir'],
            aux_client=aux_client if not ablation_cfg.get('single_model', False) else None,
            aux_model_name=config['model']['auxiliary_model']['name'],
            python_timeout=python_timeout,
            download_large_file_threshold_mb=download_large_mb,
            download_segment_size_mb=download_segment_mb,
            download_max_retries=download_max_retries,
            # Ablation settings
            disable_intent_extraction=ablation_cfg.get('disable_intent_extraction', False),
            disable_intent_compression=ablation_cfg.get('disable_intent_compression', False),
            max_concurrent_fetch=ablation_cfg.get('max_concurrent_fetch', 32)
        )"""

    content = content.replace(old_tool_manager, new_tool_manager)

    file_path.write_text(content, encoding='utf-8')
    print(f"✓ 已修改: {file_path}")


def patch_tool_manager():
    """修改 tool_manager.py 添加消融开关"""

    file_path = Path("src/tools/tool_manager.py")
    content = file_path.read_text(encoding='utf-8')

    # 1. 修改 __init__ 方法添加消融参数
    # 查找 __init__ 定义并添加新参数
    init_pattern = r'(def __init__\(self,[\s\S]*?download_max_retries:\s*int\s*=\s*3)'
    init_replacement = r'''\1,
        # Ablation settings
        disable_intent_extraction: bool = False,
        disable_intent_compression: bool = False,
        max_concurrent_fetch: int = 32'''

    content = re.sub(init_pattern, init_replacement, content)

    # 2. 添加属性存储
    # 查找 self.aux_model_name = aux_model_name 之后添加
    old_attr = "self.aux_model_name = aux_model_name"
    new_attr = """self.aux_model_name = aux_model_name

        # Ablation settings
        self.disable_intent_extraction = disable_intent_extraction
        self.disable_intent_compression = disable_intent_compression
        self.max_concurrent_fetch = max_concurrent_fetch"""

    content = content.replace(old_attr, new_attr)

    # 3. 修改 _analyze_search_intent 方法添加消融检查
    old_search_intent = "async def _analyze_search_intent(self"
    new_search_intent = """async def _analyze_search_intent(self"""

    # 在方法开头添加消融检查
    search_intent_check = '''
        # Ablation: skip intent extraction if disabled
        if self.disable_intent_extraction:
            return ""
'''

    # 查找并修改 _analyze_search_intent
    pattern = r'(async def _analyze_search_intent\(self[^)]*\)[^:]*:)'
    def add_ablation_check(match):
        return match.group(1) + search_intent_check

    content = re.sub(pattern, add_ablation_check, content, count=1)

    # 4. 修改 _analyze_click_intent 方法
    pattern = r'(async def _analyze_click_intent\(self[^)]*\)[^:]*:)'
    content = re.sub(pattern, add_ablation_check, content, count=1)

    # 5. 修改并行fetch的并发数
    old_concurrent = "max_concurrent=32"
    new_concurrent = "max_concurrent=self.max_concurrent_fetch"
    content = content.replace(old_concurrent, new_concurrent)

    file_path.write_text(content, encoding='utf-8')
    print(f"✓ 已修改: {file_path}")


def patch_memory_folding():
    """修改 memory_folding.py 添加意图消融支持"""

    file_path = Path("src/memory/memory_folding.py")
    content = file_path.read_text(encoding='utf-8')

    # 在 fold_memory 函数签名中添加 disable_intent 参数
    old_sig = "async def fold_memory("

    # 查找函数签名并添加参数
    pattern = r'(async def fold_memory\([^)]*)(reasoning_context:\s*str)'
    replacement = r'\1\2,\n    disable_intent_in_memory: bool = False'

    if 'disable_intent_in_memory' not in content:
        content = re.sub(pattern, replacement, content)

    file_path.write_text(content, encoding='utf-8')
    print(f"✓ 已修改: {file_path}")


def create_ablation_configs():
    """创建消融实验配置文件"""

    import yaml

    config_dir = Path("config/ablation")
    config_dir.mkdir(parents=True, exist_ok=True)

    # 读取基础配置
    with open("config/config.yaml", 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    # 消融配置定义
    ablations = {
        "no_episode_memory": {
            "memory": {"episode_memory": {"enabled": False}}
        },
        "no_working_memory": {
            "memory": {"working_memory": {"enabled": False}}
        },
        "no_tool_memory": {
            "memory": {"tool_memory": {"enabled": False}}
        },
        "no_memory_folding": {
            "memory": {
                "fold_threshold": 0.0,
                "episode_memory": {"enabled": False},
                "working_memory": {"enabled": False},
                "tool_memory": {"enabled": False}
            }
        },
        "no_intent_extraction": {
            "ablation": {"disable_intent_extraction": True}
        },
        "no_intent_compression": {
            "memory": {"auto_summarize_tool_results": {"enabled": False}},
            "ablation": {"disable_intent_compression": True}
        },
        "no_parallel_retrieval": {
            "ablation": {"max_concurrent_fetch": 1}
        }
    }

    for name, changes in ablations.items():
        config = base_config.copy()

        # 深度合并配置
        def deep_merge(base, updates):
            for key, value in updates.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value

        deep_merge(config, changes)

        # 添加ablation标识
        if 'ablation' not in config:
            config['ablation'] = {}
        config['ablation']['name'] = name

        # 保存
        output_path = config_dir / f"{name}.yaml"
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        print(f"✓ 已创建配置: {output_path}")


def main():
    print("="*60)
    print("WebSearcher 消融实验补丁")
    print("="*60)

    print("\n[1/4] 修改 run_agent.py...")
    try:
        patch_run_agent()
    except Exception as e:
        print(f"✗ 修改失败: {e}")

    print("\n[2/4] 修改 tool_manager.py...")
    try:
        patch_tool_manager()
    except Exception as e:
        print(f"✗ 修改失败: {e}")

    print("\n[3/4] 修改 memory_folding.py...")
    try:
        patch_memory_folding()
    except Exception as e:
        print(f"✗ 修改失败: {e}")

    print("\n[4/4] 创建消融配置文件...")
    try:
        create_ablation_configs()
    except Exception as e:
        print(f"✗ 创建失败: {e}")

    print("\n" + "="*60)
    print("补丁应用完成！")
    print("="*60)
    print("""
下一步:
1. 运行消融实验:
   python run_ablation.py --dataset data/GAIA/dev.json --max_examples 50

2. 运行单个消融:
   python run_ablation.py --dataset data/GAIA/dev.json --ablation no_working_memory

3. 汇总结果:
   python run_ablation.py --summarize_only --output_dir outputs/ablation --latex
""")


if __name__ == "__main__":
    main()
