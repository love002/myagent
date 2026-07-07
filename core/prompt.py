SYSTEM_PROMPT_CONFIG = {
    "system_prompt": {
        "role": "你是一个全能私人助手 AI Agent。但是希望你回答能简洁明了",
        "task": "暂无",
        "tool": "暂无",
        "constraint": "暂无",
        "process": "暂无",
        "output_format": "暂无"
    }
}

def get_system_prompt():
    """
    从内置的字典配置生成系统提示词。
    
    Returns:
        str: 格式化后的系统提示词。
    """
    prompt_config = SYSTEM_PROMPT_CONFIG['system_prompt']
    
    # 使用更美观的分隔线和间距
    separator = "-" * 20
    prompt = (
        f"{separator}\n"
        f"🤖 角色 (Role):\n{prompt_config['role']}\n"
        f"{separator}\n"
        f"📋 任务 (Task):\n{prompt_config['task']}\n"
        f"{separator}\n"
        f"🛠️  工具 (Tool):\n{prompt_config['tool']}\n"
        f"{separator}\n"
        f"⚠️  约束 (Constraint):\n{prompt_config['constraint']}\n"
        f"{separator}\n"
        f"⚙️  流程 (Process):\n{prompt_config['process']}\n"
        f"{separator}\n"
        f"📄 结构化输出 (Output Format):\n{prompt_config['output_format']}\n"
        f"{separator}\n"
    )
    return prompt

if __name__ == '__main__':
    system_prompt = get_system_prompt()
    print(system_prompt)