import os
import sys
import numpy as np

# Make the repo root importable so that `paths_config` resolves regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import CTRATE_TRAIN_IMAGES, require

def validate_and_cleanup_npz(root_dir, delete_on_error=True):
    """
    递归遍历 root_dir 下所有 .npz 文件：
      - 尝试加载；
      - 若失败且 delete_on_error=True，则删除该文件。
    
    参数:
        root_dir (str): 要遍历的根目录路径
        delete_on_error (bool): 是否在加载失败时删除文件（默认 True）
    """
    error_files = []
    deleted_files = []
    total_files = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.npz'):
                total_files += 1
                filepath = os.path.join(dirpath, filename)
                try:
                    with np.load(filepath, allow_pickle=False) as data:
                        # 强制读取 keys 以触发潜在错误（如损坏、不完整）
                        _ = list(data.keys())
                    # print(f"[OK] {filepath}")
                except Exception as e:
                    print(f"[ERROR] {filepath} -> {e}")
                    error_files.append((filepath, str(e)))
                    
                    if delete_on_error:
                        try:
                            os.remove(filepath)
                            print(f"[DELETED] {filepath}")
                            deleted_files.append(filepath)
                        except Exception as del_err:
                            print(f"[FAILED TO DELETE] {filepath} -> {del_err}")

    # 汇总报告
    print("\n" + "="*70)
    print(f"总计检查 .npz 文件: {total_files}")
    print(f"出错文件数量: {len(error_files)}")
    print(f"已删除文件数量: {len(deleted_files)}")
    
    if error_files:
        print("\n出错文件列表（含未删除的）:")
        for fp, err in error_files:
            status = "（已删除）" if fp in deleted_files else "（未删除）"
            print(f"  - {fp} {status}: {err}")
    else:
        print("✅ 所有 .npz 文件均可正常加载！")

# ========================
# 使用示例
# ========================
if __name__ == "__main__":
    require("CTRATE_TRAIN_IMAGES")
    root_folder = CTRATE_TRAIN_IMAGES  # ←← 修改为你的实际路径（默认=训练预处理输出目录）

    # 安全模式：先设 delete_on_error=False 测试
    # validate_and_cleanup_npz(root_folder, delete_on_error=False)

    # 确认无误后，再启用删除
    validate_and_cleanup_npz(root_folder, delete_on_error=True)