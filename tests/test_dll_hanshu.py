import pefile

def get_dll_export_functions(dll_path: str):
    pe = pefile.PE(dll_path)
    func_names = []
    if hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
        for exp_symbol in pe.DIRECTORY_ENTRY_EXPORT.symbols:
            # 只保留有名字的导出函数
            if exp_symbol.name:
                func_name = exp_symbol.name.decode("utf-8")
                func_names.append(func_name)
    pe.close()
    # 排序
    func_names = sorted(func_names)
    return func_names


if __name__ == "__main__":
    # ========== 修改这里，smcsh_mbs.dll放在py同目录 ==========
    dll_file = r".\MCDLL_NET.dll"
    # 绝对路径示例：dll_file = r"C:\SMC\smcsh_mbs.dll"

    funcs = get_dll_export_functions(dll_file)
    print(f"\n【MCDLL.dll】导出函数总数：{len(funcs)}")
    print("="*50)
    for name in funcs:
        print(name)

    # 保存全部函数名到文本
    with open("MCDLL_mbs_func_list.txt", "w", encoding="utf-8") as f:
        for name in funcs:
            f.write(name + "\n")
    print("\n✅ 函数列表已保存到 MCDLL_mbs_func_list.txt")