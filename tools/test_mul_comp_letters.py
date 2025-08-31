import json
import pathlib
import pickle
import unittest
import os

import numpy as np
import pandas as pd

from tools.mul_comp_letters import mul_comp_letters
from statsmodels.stats.multicomp import pairwise_tukeyhsd


class TestMultcompLetters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 创建缓存目录
        cls.cache_dir = pathlib.Path(__file__).parent / "test_cache"
        cls.cache_dir.mkdir(exist_ok=True)
        cls.cache_file = cls.cache_dir / "test_data.pkl"

    def get_or_create_test_data(self, n_tests=100):
        """获取缓存的测试数据，如果不存在则创建并缓存"""
        if self.cache_file.exists():
            print("使用缓存的测试数据")
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)

        print("生成新的测试数据并缓存")
        test_data = []
        for i in range(n_tests):
            n_groups = np.random.randint(2, 26)
            n_every_groups = np.random.randint(3, 33, size=n_groups)
            locs = np.random.uniform(0, 20, size=n_groups)
            scales = np.random.uniform(0, 10, size=n_groups)
            group = [f'g {i:>3}' for i in range(n_groups) for _ in range(n_every_groups[i])]
            value = np.concatenate([np.random.normal(loc=locs[i], scale=scales[i], size=n_every_groups[i])
                                    for i in range(n_groups)])
            tukey_res = pairwise_tukeyhsd(value, group)
            summary_data = tukey_res.summary().data
            result_df = pd.DataFrame(summary_data[1:], columns=summary_data[0])
            test_data.append(result_df)

        # 缓存测试数据
        with open(self.cache_file, 'wb') as f:
            pickle.dump(test_data, f)

        return test_data

    def test_mul_comp_letters(self):
        test_data = self.get_or_create_test_data()

        for result_df in test_data:
            letter = mul_comp_letters(result_df).set_index('group').loc[:, 'letter']
            for _, (g1, g2, reject) in result_df.loc[:, ['group1', 'group2', 'reject']].iterrows():
                g1_l = letter.loc[g1]
                g2_l = letter.loc[g2]
                has_same_letter = not any(i in g2_l for i in g1_l)
                self.assertEqual(has_same_letter, reject)

    @classmethod
    def tearDownClass(cls):
        # 如果需要清理缓存，可以取消下面的注释
        # if cls.cache_file.exists():
        #     os.remove(cls.cache_file)
        # if cls.cache_dir.exists():
        #     os.rmdir(cls.cache_dir)
        pass


if __name__ == '__main__':
    unittest.main()
