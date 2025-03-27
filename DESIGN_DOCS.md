Target:
"""
### **模块1: Quant-Paper Agent (QPA)**

- **功能**：数据抓取 → 内容解析 → 知识卡生成 → 标签标注 → 知识库存储。
- **子模块设计**：
    1. **arXiv Crawler**
        - 每日定时抓取量化领域论文（关键词过滤：`quant*, algorithmic trading, portfolio, risk model`等）；
        - 支持增量更新，避免重复抓取。
    2. **PDF解析与信息提取**
        - 使用PDF解析工具（如`PyMuPDF`或`GROBID`）提取文本、公式、图表；
        - 结构化分割（标题、摘要、方法、实验）。
    3. **Knowledge Card 生成器**
        - 前置工作：
            - Card scheme 的定义
            - Prompt Engineering
        - 基于LLM（GPT-4或Claude-3.5）的多阶段Prompt Workflow：
            - **第一阶段**：生成摘要、亮点（创新点）、核心方法、实验结果；
            - **第二阶段**：标注标签（如`#Portfolio Optimization`, `#High-Frequency Trading`）；
            - **质量控制**：设计校验规则（如关键术语一致性检查）。
    4. **向量化与存储**
        - 文本嵌入生成（可选模型：OpenAI Embedding、`BAAI /bge-base-en-v1.5`）；
        - 存储至向量数据库（Pinecone/Chroma/FAISS）；
        - 元数据关联（标题、作者、标签、发布日期）。
"""

目前我的想法是先完成 stage1 中设定的基本功能。

"""
> 关于什么是 QPA 可以参考 [QPA  & QDR System Design Docs](https://www.notion.so/QPA-QDR-System-Design-Docs-1a35a782ea2080b7a3d7fd758ccecca1?pvs=21)
> 

## 🎯 Milestone

- [ ]  **每日抓取**量化领域（Quant Finance, Algorithmic Trading, Portfolio Optimization, Market Microstructure, Risk Modeling等）的arXiv论文，并提供封装好的接口。
    - [x]  Daily arxiv crawl
    - [ ]  Encapsulated interface
- [ ]  **自动生成结构化知识卡**（Knowledge Card），包括摘要、亮点、方法、实验结果、文章标签标注。
    - [ ]  PDF → Structure Text （Markdown / json ..）
    - [ ]  Prompt Engineering
    - [ ]  Agent / workflow
- [ ]  **构建可检索的知识库**（知识卡片向量化+相似度分析）。对于 stage1 我们可以先以 abstract 作为输入进行构建。
    - [ ]  Embedding instance utils (like `gemini`)
    - [ ]  Vector Database utils (like `qdrant`)
    - [ ]  Similarity-based knowledge graph build and retrieve
"""