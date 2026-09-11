def check_dll_bit(dll_path: str) -> int:
    """
    读取dll文件PE头，返回 32 / 64；失败返回0
    """
    with open(dll_path, "rb") as f:
        data = f.read()
    # PE签名偏移
    pe_offset = int.from_bytes(data[0x3c:0x40], byteorder="little")
    # 校验PE标记
    if data[pe_offset:pe_offset+4] != b"PE\0\0":
        return 0
    machine = int.from_bytes(data[pe_offset+4:pe_offset+6], byteorder="little")
    # IMAGE_FILE_MACHINE_I386=0x014C (32bit), IMAGE_FILE_MACHINE_AMD64=0x8664 (64bit)
    if machine == 0x014C:
        return 32
    elif machine == 0x8664:
        return 64
    else:
        return 0

if __name__ == "__main__":
    dll_file = r"MCDLL_NET copy.dll" # 修改成你的dll路径
    res = check_dll_bit(dll_file)
    if res == 32:
        print(f"{dll_file} → 32位库")
    elif res == 64:
        print(f"{dll_file} → 64位库")
    else:
        print("无法识别，不是有效的PE库文件")