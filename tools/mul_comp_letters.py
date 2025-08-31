import pandas as pd
import numpy as np

# 字母表，用于生成标记
LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
           'h', 'i', 'j', 'k', 'l', 'm', 'n',
           'o', 'p', 'q', 'r', 's', 't',
           'u', 'v', 'w', 'x', 'y', 'z']


def mul_comp_letters(comp_df: pd.DataFrame) -> pd.DataFrame:
    """
    根据多重比较结果生成字母标记

    :param comp_df: 包含多重比较结果的DataFrame，必须包含'group1'、'group2'和'reject'列
    :return: 包含每个组及其对应字母标记的DataFrame
    """
    assert all(item in comp_df.columns for item in ['group1', 'group2', 'reject'])

    # 按均值排序组列表
    sorted_group_list = _sort_by_mean(comp_df)

    n = len(sorted_group_list)
    reject_df = comp_df.loc[:, ['group1', 'group2', 'reject']].set_index(['group1', 'group2'])

    # 初始化字母矩阵，每个组开始时都分配第一个字母
    letter_matrix = np.ones([n, 1], dtype=bool)

    # 处理拒绝假设的情况，生成字母矩阵
    letter_matrix = _insert_process(letter_matrix, reject_df, sorted_group_list)

    # 对字母矩阵的列进行排序
    letter_matrix = _sort_columns(letter_matrix)

    # 检查是否超出可用字母数量
    if letter_matrix.shape[1] > len(LETTERS):
        raise ValueError('字母数量不足')
    else:
        # 为每个组生成字母标记
        letter_list = [
            ''.join(LETTERS[j] for j, use in enumerate(letter_matrix[idx, :]) if use)
            for idx in range(n)
        ]
        return pd.DataFrame({'group': sorted_group_list, 'letter': letter_list})


def _sort_by_mean(comp_df: pd.DataFrame):
    """
    根据均值差异对组进行排序

    :param comp_df: 包含'group1'、'group2'和'meandiff'列的DataFrame
    :return: 按均值降序排列的组列表
    """
    assert all(item in comp_df.columns for item in ['group1', 'group2', 'meandiff'])

    temp_df = comp_df.loc[:, ['group1', 'group2', 'meandiff']]
    first_group = temp_df.iloc[0, 0]

    # 构建虚拟均值字典，以第一个组为基准
    virtual_mean = {first_group: 0}

    # 计算每个组的相对均值
    group2_is_first_group = temp_df['group2'] == first_group
    for row in temp_df.loc[group2_is_first_group].itertuples():
        virtual_mean[row.group1] = virtual_mean[row.group2] - row.meandiff
    for row in temp_df.loc[~group2_is_first_group].itertuples():
        virtual_mean[row.group2] = virtual_mean[row.group1] + row.meandiff

    # 按均值降序排列组
    sorted_group_list = sorted(virtual_mean.keys(), key=lambda x: virtual_mean[x], reverse=True)
    return sorted_group_list


def _insert_process(matrix, reject_df, sorted_group_list):
    """
    处理拒绝假设的情况，更新字母矩阵

    :param matrix: 初始字母矩阵
    :param reject_df: 包含拒绝假设信息的DataFrame
    :param sorted_group_list: 排序后的组列表
    :return: 更新后的字母矩阵
    """
    group_list_length = matrix.shape[0]
    for i in range(group_list_length):
        for j in range(i + 1, group_list_length):
            group_i = sorted_group_list[i]
            group_j = sorted_group_list[j]

            # 确定索引顺序
            idx = (group_i, group_j) if (group_i, group_j) in reject_df.index else (group_j, group_i)

            # 如果拒绝假设，需要分配不同的字母
            if reject_df.loc[idx].reject:
                # 找出两组共享的字母列
                shared_cols = np.where(matrix[i, :] & matrix[j, :])[0]
                if len(shared_cols):
                    # 创建新列，将共享的字母分开
                    new_cols = matrix[:, shared_cols].copy()
                    new_cols[i, :] = False
                    matrix[j, shared_cols] = False
                    matrix = np.hstack([matrix, new_cols])

                    # 消除冗余列
                    matrix = _absorb_process(matrix)

    return matrix


def _absorb_process(matrix):
    """
    消除冗余列，优化字母矩阵

    :param matrix: 字母矩阵
    :return: 优化后的字母矩阵
    """
    n_cols = matrix.shape[1]

    # 基本情况：如果矩阵只有0或1列，直接返回
    if n_cols <= 1:
        return matrix

    # 检查是否存在冗余列
    for i in range(n_cols - 1):
        for j in range(i + 1, n_cols):
            col_i = matrix[:, i]
            col_j = matrix[:, j]

            # 检查列j是否被列i吸收
            if np.all(np.logical_or(~col_j, col_i)):
                # 删除列j并递归处理剩余矩阵
                reduced_matrix = np.delete(matrix, j, axis=1)
                return _absorb_process(reduced_matrix)

            # 检查列i是否被列j吸收
            if np.all(np.logical_or(~col_i, col_j)):
                # 删除列i并递归处理剩余矩阵
                reduced_matrix = np.delete(matrix, i, axis=1)
                return _absorb_process(reduced_matrix)

    # 如果没有找到冗余列，返回当前矩阵
    return matrix


def _sort_columns(matrix):
    """
    对字母矩阵的列进行排序，使其更有序

    :param matrix: 字母矩阵
    :return: 列排序后的字母矩阵
    """
    # 找出每列第一个为True的行索引
    first_true_row = np.array([
        np.argmax(matrix[:, i]) if np.any(matrix[:, i]) else matrix.shape[0]
        for i in range(matrix.shape[1])
    ])

    # 按第一个True的位置排序列
    sorted_indices = np.argsort(first_true_row)
    return matrix[:, sorted_indices]
