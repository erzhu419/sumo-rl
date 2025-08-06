#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：验证模块导入是否正常工作
"""

print("=== 模块导入测试 ===")

import sys
import os

# 获取当前脚本的绝对路径
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
project_root = os.path.dirname(current_dir)

print(f"当前工作目录: {os.getcwd()}")
print(f"脚本文件位置: {current_file}")
print(f"当前脚本目录: {current_dir}")
print(f"项目根目录: {project_root}")

# 确保项目根目录在Python模块搜索路径中
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"✓ 已将项目根目录添加到Python路径")

# 确保当前目录在Python模块搜索路径中
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    print(f"✓ 已将当前目录添加到Python路径")

print("\n当前Python路径的前5个条目:")
for i, path in enumerate(sys.path[:5]):
    print(f"  {i}: {path}")

print("\n=== 尝试导入模块 ===")
success = False

try:
    from g_no_control import create_obj
    from g_no_control import save_data
    print("✓ 成功使用包导入: from g_no_control import create_obj, save_data")
    success = True
    import_method = "包导入"
except ImportError as e:
    print(f"✗ 包导入失败: {e}")

if not success:
    try:
        import create_obj
        import save_data
        print("✓ 成功使用直接导入: import create_obj, save_data")
        success = True
        import_method = "直接导入"
    except ImportError as e:
        print(f"✗ 直接导入也失败: {e}")

if success:
    print(f"\n✓ 导入方式: {import_method}")
    print("✓ 测试成功! 模块可以正常导入")
else:
    print("\n✗ 测试失败! 无法导入模块")
    
print("=== 测试完成 ===")
