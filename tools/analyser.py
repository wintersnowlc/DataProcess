import logging
import os
import sys
import warnings
from typing import Optional, Dict, Any, Literal

import pandas as pd
import numpy as np

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from tools.mul_comp_letters import mul_comp_letters


class RInitializationError(Exception):
    """R环境初始化失败时的自定义异常"""
    pass


class NotNanGroupLessError(Exception):
    """非空组数小于2时的自定义异常"""
    pass


class Analyser:
    """
    统计分析工具类，本工具必须在纯英文目录中调用，否则再尝试调用R环境会报DecodeError。用于执行以下操作：
    1. 单因素方差分析(ANOVA)
    2. Tukey HSD事后检验
    3. 生成显著性字母标记
    4. 计算统计汇总数据
    支持通过R环境进行更复杂的多重比较分析，可指定便携式R路径
    Attributes:
        _multcomp_view (class attribute): R包multcompView的类级缓存
        result (dict): 存储分析结果的字典，包含以下键：
            - "anova": ANOVA结果表
            - "tukey": Tukey HSD结果表
            - "summary": 统计汇总数据
        group_col (str): 分组变量列名
        value_col (str): 数值变量列名
        alpha (float): 显著性水平阈值
        r_base_path (str): R环境的基路径
        data (pd.DataFrame): 分析用数据集
    Example:
        >>> ana = Analyser(pd.DataFrame({
        ... 'group':  np.repeat(['0', '1', '2', '3', '4'], 10),
        ... 'value': np.arange(50),
        ... }), alpha=0.05).fit()

    """

    _multcomp_view = None  # 类级缓存，存储R包引用

    def __init__(
            self,
            data: pd.DataFrame,
            name: str = 'data',
            group_col: str = 'group',
            value_col: str = 'value',
            alpha: float = 0.05,
            r_base_path: Optional[str] = None,
            letter_method: Literal['r', 'py'] = 'py'
    ):
        """
        初始化分析器对象

        :param data: 待分析的数据集
        :param group_col: 分组变量列名，默认为 'group'
        :param value_col: 数值变量列名，默认为 'value'
        :param alpha: 显著性水平，默认为 0.05
        :param r_base_path: 便携式R环境的基路径，可选
        """
        self.letter_method = letter_method
        self.name = name

        self.result: Dict[str, Any] = {}
        self.group_col = group_col
        self.value_col = value_col
        self.alpha = alpha
        self.r_base_path = r_base_path or os.path.join(os.path.dirname(__file__), 'R-Portable')
        self.data = data  # 使用setter方法

    @property
    def data(self) -> pd.DataFrame:
        """获取当前分析数据集"""
        return self._data

    # 在data setter中添加验证
    @data.setter
    def data(self, data: pd.DataFrame):
        # 验证数据框非空
        if data.empty:
            raise ValueError("输入数据框不能为空")

        # 验证列存在
        if self.group_col not in data.columns:
            raise ValueError(f"分组列'{self.group_col}'不存在于数据框中")
        if self.value_col not in data.columns:
            raise ValueError(f"数值列'{self.value_col}'不存在于数据框中")

        self._data = data.copy()

        try:
            if not pd.api.types.is_numeric_dtype(data[self.value_col]):
                converted = pd.to_numeric(self._data[self.value_col], errors='coerce')
                if converted.isna().any():
                    warnings.warn(f"{self.name} 数值列'{self.value_col}'中有{converted.isna().sum()}个无法转换的值")
                self._data[self.value_col] = converted
        except Exception as e:
            raise ValueError(f"数值列'{self.value_col}'转换失败: {str(e)}")

        self.result.clear()

    @classmethod
    def init_r_environment(cls, base_path: Optional[str] = None):
        """
        初始化R环境

        :param base_path: 便携式R环境的基路径，可选
        :raises ImportError: 当rpy2未安装时抛出
        :raises RInitializationError: 当R环境初始化失败时抛出
        """
        if cls._multcomp_view is not None:
            return  # 已初始化

        try:
            if base_path:
                base_path = os.path.abspath(base_path)
                os.environ["R_ENVIRON"] = "UTF-8"
                os.environ["R_ENVIRON_USER"] = "UTF-8"
                os.environ["R_HOME"] = base_path
                os.environ["R_LIBS"] = os.path.join(base_path, "library")
                os.environ["PATH"] = os.path.join(base_path, "bin") + os.pathsep + os.environ["PATH"]
            # print(os.environ["R_HOME"])
            if sys.platform == 'win32':
                os.environ['PYTHONNET_PYDLL'] = ''
                os.environ['RPY2_CFFI_MODE'] = 'ABI'

            import rpy2.robjects as ro
            from rpy2.robjects.packages import importr

            ro.rinterface.initr()
            cls._multcomp_view = importr('multcompView')

        except ImportError as e:
            raise ImportError("未安装rpy2，无法使用multcompView") from e
        except Exception as e:
            logging.exception(e)
            raise RInitializationError(f"R环境初始化失败: {str(e)}") from e

    def anova_result(self) -> pd.DataFrame:
        """
        执行单因素方差分析

        :return: ANOVA结果表
        """
        groups_with_valid_data = self._data.groupby(self.group_col)[self.value_col].apply(lambda x: x.notna().any())
        valid_groups_count = groups_with_valid_data.sum()

        if valid_groups_count < 2:
            raise NotNanGroupLessError(f"只有{valid_groups_count}个分组均为非空值，无法执行ANOVA")
        else:
            formula = f"{self.value_col} ~ C({self.group_col})"
            model = ols(formula, data=self._data).fit()
            result = anova_lm(model, typ=2)
        self.result["anova"] = result
        return result

    def tukey_result(self) -> Optional[pd.DataFrame]:
        """
        执行Tukey HSD事后检验（仅在ANOVA显著时执行）

        :return: Tukey HSD结果表，ANOVA不显著时返回None
        :warns: 当ANOVA不显著时发出警告
        """
        if "anova" not in self.result:
            try:
                self.anova_result()
            except NotNanGroupLessError as e:
                warnings.warn(f'{self.name} {e}')
                result_df = None
                self.result["tukey"] = result_df
                return result_df

        anova_p = self.result["anova"].iloc[0, 3]  # 提取PR(>F)值

        if anova_p > self.alpha:
            warnings.warn(
                f"{self.name} ANOVA p值 ({anova_p:.4f}) > 显著性水平 ({self.alpha})，跳过Tukey HSD检验"
            )
            result_df = None
        else:
            # 按均值降序排序分组
            group_means = self._data.groupby(self.group_col)[self.value_col].mean().sort_values(ascending=False)
            fill_length = int(np.floor(np.log10(len(group_means.index)))) + 1
            group_map_i = {group: f'{i:>{fill_length}} {group}' for i, group in enumerate(group_means.index)}
            i_map_group = {i: group for group, i in group_map_i.items()}

            groups = self._data[self.group_col].map(group_map_i).to_numpy()
            values = self._data[self.value_col].to_numpy()

            # 执行Tukey检验并生成图表
            tukey_res = pairwise_tukeyhsd(
                endog=values,
                groups=groups,
                alpha=self.alpha
            )

            # import matplotlib.pyplot as plt
            # with plt.rc_context(rc={'figure.figsize': (10, 8), 'font.family': 'SimHei'}):
            #     tukey_res.plot_simultaneous()
            #     plt.title('Tukey HSD检验')
            #     plt.savefig('Tukey_HSD.png')
            # plt.close()

            # 处理结果数据
            summary_data = tukey_res.summary().data
            result_df = pd.DataFrame(summary_data[1:], columns=summary_data[0])
            result_df["group1"] = result_df["group1"].map(i_map_group)
            result_df["group2"] = result_df["group2"].map(i_map_group)

        self.result["tukey"] = result_df

        return result_df

    def letter_result(self) -> pd.DataFrame:
        """
        生成显著性字母标记

        :return: 包含分组和显著性字母的数据框
        :warns: 当R环境不可用时发出警告
        """
        if "tukey" not in self.result:
            self.tukey_result()

        tukey_df = self.result["tukey"]

        # 按均值降序排序分组
        group_means = self._data.groupby(self.group_col)[self.value_col].mean().reset_index()
        group_means = group_means.sort_values(by=self.value_col, ascending=False)
        sorted_groups = group_means[self.group_col].tolist()

        if tukey_df is None:
            return pd.DataFrame({
                "group": sorted_groups,
                "letter": ["a"] * len(sorted_groups)  # 所有组使用相同字母
            }, dtype=str)

        try:
            if self.letter_method == 'r':
                result = self._letter_with_r(tukey_df)
            elif self.letter_method == 'py':
                result = mul_comp_letters(tukey_df)
            else:
                raise ValueError(f"未知字母标记方法'{self.letter_method}'")
            return result
        except (ImportError, RInitializationError) as e:
            warnings.warn(f"{self.name} 字母标记不可用: {str(e)}")
            # 回退方案：返回空字母
            return pd.DataFrame({
                "group": sorted_groups,
                "letter": [np.nan] * len(sorted_groups)
            }, dtype=str)

    def _letter_with_r(self, tukey_df):
        if self.__class__._multcomp_view is None:
            self.init_r_environment(self.r_base_path)
        import rpy2.robjects as ro
        # 准备命名p值向量
        p_values = tukey_df["p-adj"].tolist()
        comparisons = [
            f"{row['group1']}-{row['group2']}"
            for _, row in tukey_df.iterrows()
        ]
        p_vec = ro.FloatVector(p_values)
        p_vec.names = ro.StrVector(comparisons)
        letters = self.__class__._multcomp_view.multcompLetters(
            p_vec,
            threshold=self.alpha,
            reversed=ro.r('FALSE'),  # 'a'分配给最高均值组
        )
        letters_vec = letters.rx2("Letters")
        result = pd.DataFrame({
            "group": letters_vec.names,
            "letter": letters_vec
        })
        return result

    @property
    def summary(self) -> pd.DataFrame:
        """
        获取统计汇总结果（包含显著性字母）

        :return: 包含均值、标准差、计数和显著性字母的数据框
        """
        if "summary" not in self.result:
            # 计算描述性统计
            desc_df = self._data.groupby(self.group_col)[self.value_col].agg(['mean', 'std', 'count']).reset_index()
            # 获取字母标记
            letter_df = self.letter_result()
            summary_df = pd.merge(desc_df, letter_df, on="group")
            summary_df['formated'] = summary_df.apply(
                lambda row: f"{row['mean']:.2f}±{row['std']:.2f} {row['letter']}",
                axis=1
            )
            self.result["summary"] = summary_df
            # 合并结果

        return self.result["summary"]

    def fit(self, force: bool = False) -> 'Analyser':
        """
        执行所有分析

        :param force: 是否强制重新计算
        :return: 当前分析器实例
        """
        if force:
            self.result.clear()

        _ = self.summary  # 触发所有计算
        return self
