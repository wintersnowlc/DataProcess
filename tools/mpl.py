#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
matplotlib样式管理工具

提供长期设置样式、临时设置样式和预览样式的功能
"""

import os
import contextlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# 样式文件目录
STYLE_DIR = Path(__file__).parent / '.mplstyles'


def list_styles():
    """
    列出所有可用的样式

    Returns:
        dict[str, list[str]]: 可用样式名称列表
    """
    if not STYLE_DIR.exists():
        STYLE_DIR.mkdir(exist_ok=True)

    # 获取内置样式
    builtin_styles = plt.style.available

    # 获取自定义样式
    custom_styles = []
    if STYLE_DIR.exists():
        custom_styles = [f.stem for f in STYLE_DIR.glob('*.mplstyle')]

    return {
        'builtin': builtin_styles,
        'custom': custom_styles
    }


def set_style(style_name):
    """
    长期设置matplotlib样式（程序运行期间有效）

    Args:
        style_name (str): 样式名称，可以是内置样式或自定义样式

    Returns:
        bool: 设置是否成功
    """
    styles = list_styles()

    # 检查是否是内置样式
    if style_name in styles['builtin']:
        plt.style.use(style_name)
        return True

    # 检查是否是自定义样式
    if style_name in styles['custom']:
        style_path = STYLE_DIR / f"{style_name}.mplstyle"
        plt.style.use(str(style_path))
        return True

    print(f"样式 '{style_name}' 不存在。可用样式: {styles}")
    return False


@contextlib.contextmanager
def temp_style(style_name):
    """
    临时设置matplotlib样式的上下文管理器

    Args:
        style_name (str): 样式名称，可以是内置样式或自定义样式

    Yields:
        None

    Example:
        with temp_style('sci_paper'):
            plt.plot(x, y)
            plt.savefig('figure.png')
    """
    styles = list_styles()

    # 确定样式路径
    if style_name in styles['builtin']:
        style_path = style_name
    elif style_name in styles['custom']:
        style_path = str(STYLE_DIR / f"{style_name}.mplstyle")
    else:
        print(f"警告: 样式 '{style_name}' 不存在，使用默认样式。")
        style_path = 'default'

    # 使用with语句临时应用样式
    with plt.style.context(style_path):
        yield


def preview_style(style_name, save_path=None):
    """
    预览指定的样式

    Args:
        style_name (str): 样式名称
        save_path (str, optional): 保存预览图的路径，如果为None则显示图形

    Returns:
        bool: 预览是否成功
    """
    styles = list_styles()
    all_styles = styles['builtin'] + styles['custom']

    if style_name not in all_styles:
        print(f"样式 '{style_name}' 不存在。可用样式: {all_styles}")
        return False

    # 创建示例图形
    with temp_style(style_name):
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        fig.suptitle(f"Style Preview: {style_name}", fontsize=14)

        # 折线图
        ax = axes[0, 0]
        x = np.linspace(0, 10, 100)
        for i in range(5):
            ax.plot(x, np.sin(x + i * 0.5), label=f'Line {i+1}')
        ax.set_title('Line Plot')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.legend(loc='upper right', fontsize='small')

        # 散点图
        ax = axes[0, 1]
        x = np.random.rand(50)
        y = np.random.rand(50)
        colors = np.random.rand(50)
        sizes = 1000 * np.random.rand(50)
        ax.scatter(x, y, c=colors, s=sizes, alpha=0.5)
        ax.set_title('Scatter Plot')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')

        # 柱状图
        ax = axes[1, 0]
        x = np.arange(5)
        y = np.random.rand(5)
        ax.bar(x, y, color='skyblue')
        ax.set_title('Bar Plot')
        ax.set_xlabel('Category')
        ax.set_ylabel('Value')

        # 直方图
        ax = axes[1, 1]
        x = np.random.normal(size=1000)
        ax.hist(x, bins=30, alpha=0.7)
        ax.set_title('Histogram')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
            print(f"预览图已保存至: {save_path}")
        else:
            plt.show()

    return True


def add_style(style_name, style_content):
    """
    添加自定义样式

    Args:
        style_name (str): 样式名称
        style_content (str): 样式内容

    Returns:
        bool: 添加是否成功
    """
    if not STYLE_DIR.exists():
        STYLE_DIR.mkdir(parents=True, exist_ok=True)

    style_path = STYLE_DIR / f"{style_name}.mplstyle"

    try:
        with open(style_path, 'w', encoding='utf-8') as f:
            f.write(style_content)
        print(f"样式 '{style_name}' 已成功添加")
        return True
    except Exception as e:
        print(f"添加样式失败: {e}")
        return False


if __name__ == "__main__":
    # 示例用法
    print("可用样式:")
    sts = list_styles()

    print("内置样式:", sts['builtin'])
    print("自定义样式:", sts['custom'])

    # 预览样式
    preview_style('en_paper')

