这是一个 Python 数据分析工具集，主要用于统计分析和数据可视化。以下是项目文件的详细摘要：

---

### **项目概览 (Project Overview)**
**主要功能:** 提供进行统计检验（如 ANOVA、Tukey HSD）和生成相应可视化图表的功能，并包含一个管理 matplotlib 样式的工具。
**核心依赖:** `pandas`, `numpy`, `statsmodels`, `matplotlib`, `rpy2` (可选, 用于 R 集成)

---

### **文件摘要 (File Summaries)**

#### 1. `mpl.py` - Matplotlib 样式管理工具
- **功能:** 用于管理和应用 matplotlib 绘图样式。
- **主要函数:**
    - `list_styles()`: 列出所有可用的内置和自定义样式。
    - `set_style(style_name)`: 全局设置绘图样式。
    - `temp_style(style_name)`: 上下文管理器，用于临时设置样式。
    - `preview_style(style_name, save_path=None)`: 生成一个包含多种图表类型的预览图来展示指定样式。
    - `add_style(style_name, style_content)`: 添加自定义的 `.mplstyle` 样式文件。
- **自定义样式目录:** `.mplstyles`

#### 2. `analyser.py` - 统计分析核心模块
- **核心类:** `Analyser`
- **功能:** 执行单因素方差分析 (ANOVA)、Tukey HSD 事后检验，并生成显著性字母标记。
- **工作流程:**
    1. **初始化:** 传入 DataFrame、指定分组列和数值列。
    2. **方差分析 (ANOVA):** `anova_result()` 方法。
    3. **事后检验 (Tukey HSD):** `tukey_result()` 方法（仅在 ANOVA 显著时进行）。
    4. **字母标记 (Lettering):** `letter_result()` 方法，可使用 R 的 `multcompView` 包或纯 Python 实现 (`mul_comp_letters`)。
    5. **结果汇总:** `summary` 属性生成包含均值、标准差、计数和显著性字母的 DataFrame。
- **特性:**
    - **R 集成:** 可选通过 `rpy2` 调用 R 的 `multcompView` 包生成字母（需配置 `R-Portable` 路径）。
    - **纯 Python 回退:** 内置 `mul_comp_letters` 函数作为备选方案。
    - **数据验证:** 自动检查数据有效性（非空、列存在、数值类型转换）。
    - **异常处理:** 定义了自定义异常（如 `RInitializationError`, `NotNanGroupLessError`）。

#### 3. `mul_comp_letters.py` - 显著性字母标记算法 (纯 Python)
- **功能:** 根据 Tukey HSD 检验的结果，生成用于标识组间显著性差异的字母标记。
- **核心函数:** `mul_comp_letters(comp_df)`
- **输入:** 一个包含 `'group1'`, `'group2'`, `'reject'` 等列的 DataFrame (来自 `pairwise_tukeyhsd` 的结果)。
- **输出:** 一个将每个分组与其对应显著性字母相关联的 DataFrame。
- **算法:** 通过构建和操作一个“字母矩阵”来逻辑性地分配字母，确保差异不显著的组共享至少一个相同字母。

#### 4. `test_mul_comp_letters.py` - 单元测试
- **功能:** 对 `mul_comp_letters` 函数进行单元测试。
- **方法:** 使用 `unittest` 框架，通过随机生成大量测试数据（或加载缓存）来验证字母标记的正确性。检查原则是：若 Tukey 检验拒绝原假设（即两组有显著差异），则它们不应共享任何字母；若未拒绝，则应共享至少一个字母。

#### 5. `__init__.py` - 包初始化文件
- 当前为空，表明 `tools` 是一个 Python 包。

#### 6. 其他文件
- `demo.ipynb`: 可能是一个 Jupyter Notebook 演示文件，展示如何使用这些工具。
- `R-Portable/`: 便携版 R 环境的目录，用于支持 `analyser.py` 中的 R 功能。
- `.mplstyles/`: 存储自定义 matplotlib 样式文件的目录。
- `__pycache__/`, `test_cache/`: Python 缓存和测试缓存目录。

---

### **项目使用流程 (Typical Workflow)**
1.  **数据准备:** 将数据加载到 Pandas DataFrame 中，确保包含分组列和数值列。
2.  **初始化分析器:**
    ```python
    from tools.analyser import Analyser
    analyzer = Analyser(df, group_col='group', value_col='value', name='MyExperiment')
    ```
3.  **执行分析:** 调用 `analyzer.fit()` 自动运行 ANOVA、Tukey HSD 和字母标记。
4.  **获取结果:** 从 `analyzer.summary` 获取带有显著性字母的汇总统计表。
5.  **(可选) 可视化:** 使用 `mpl.py` 中的工具设置图表样式，绘制结果。

---

### **关键注意事项 (Key Notes)**
- **R 依赖是可选的:** 但如果需要用到 R 的 `multcompView` 包进行字母标记，必须正确设置 `r_base_path` 并安装 `rpy2`。
- **目录要求:** 代码强调必须在**纯英文路径**下运行，否则 R 环境初始化可能会因编码问题失败。
- **测试覆盖:** `test_mul_comp_letters.py` 确保了纯 Python 字母标记算法的可靠性。

总而言之，这是一个功能集中、结构清晰的数据分析工具包，特别适合于进行组间比较的统计分析并将结果可视化。`analyser.py` 是核心，`mpl.py` 提供了美化输出的支持，`mul_comp_letters.py` 是一个重要的独立算法组件。
