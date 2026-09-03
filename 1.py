import os
import re

# 要处理的根目录（请修改为你的项目路径）
ROOT_DIR = r"D:\VScode Project\Python project\Vision_Substrate_Silicone"

# 需要处理的文件扩展名
EXTENSIONS = {".py", ".qss", ".css", ".txt"}  # 按需增减

# 匹配 font-size: 数字(可能带小数) + 单位(px/pt/em等)，并捕获数字和单位
pattern = re.compile(r'font-size\s*:\s*(\d+(?:\.\d+)?)(px|pt|em|rem|%)')

def increase_font_size(match):
    value = float(match.group(1))
    unit = match.group(2)
    # 只对 px 单位增加 3，其他单位可能需要不同处理，这里按需求可调整
    if unit == 'px':
        new_value = value + 3
        # 如果是整数则保留整数形式，否则保留一位小数
        if new_value.is_integer():
            return f"font-size: {int(new_value)}px"
        else:
            return f"font-size: {new_value:.1f}px"
    else:
        # 其他单位不变
        return match.group(0)

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        new_content, count = pattern.subn(increase_font_size, content)
        if count > 0:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"已修改 {filepath}，替换 {count} 处")
        else:
            print(f"跳过（无匹配）: {filepath}")
    except Exception as e:
        print(f"处理 {filepath} 出错: {e}")

def main():
    for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
        # 排除隐藏目录、虚拟环境等
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in {'venv', '__pycache__', 'node_modules'}]
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in EXTENSIONS:
                filepath = os.path.join(dirpath, filename)
                process_file(filepath)

if __name__ == "__main__":
    main()