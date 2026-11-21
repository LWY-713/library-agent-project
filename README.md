# 📚 图书馆智能 RAG Agent

基于 **LangChain 1.2 + DeepSeek + Milvus（Milvus Lite）** 的图书馆问答 Agent。
知识库来自 `data/library_knowledge.txt`，覆盖开放时间、馆藏图书、书架位置、研讨室、
活动、设备报修、失物招领等混合资料。文档保留多条带日期的历史通知，但**不标注**哪条
有效、哪条作废，模型必须自己按日期判断；借阅规则按身份分表，同架图书并存，并混入
大量带数字的无关通知，用来测试真实企业文档场景下的检索与推理能力。

## 目录结构

```
library_agent/
├── main.py            # 入口：命令行对话
├── agent_builder.py   # 组装 Agent（DeepSeek + 工具）
├── tools.py           # library_search_tool 检索 + borrow_manage_tool 借还（写操作）
├── rag_module.py      # 切分 / 向量化 / 入库 / 检索
├── middleware.py      # 输入校验 + 调用耗时日志
├── config.py          # 集中配置（读 .env）
├── evaluation/
│   ├── eval_questions.json  # 47 题：39 道知识/拒答 + 8 道“不该调工具”，含 L0-L5 难度标签
│   ├── evaluate.py          # 检索 / 回答 / 工具选择 / 轨迹 / token + 分层准确率
│   ├── ablation.py          # 参数消融（子进程隔离环境，真实跑多组）
│   └── report.json          # 全量评测报告（含每题轨迹）
├── data/
│   ├── library_knowledge.txt
│   └── loans.json             # 当前用户的借阅台账（写工具直接读写）
├── .env               # 密钥与参数（不要提交到 git）
└── milvus_library.db  # 本地向量库（已建好，可直接用）
```

## 运行环境

项目使用的解释器：`D:\Edge_Download\Miniconda\envs\langchain1.2`

主要依赖版本见 `requirements.txt`，已在环境中安装好。换机器时执行：

```bash
pip install -r requirements.txt
```

## 配置

编辑 `.env`：

| 变量 | 说明 |
| --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek 对话模型密钥（必填） |
| `DEEPSEEK_BASE_URL` | 默认 `https://api.deepseek.com` |
| `EMBEDDING_API_KEY` | 向量化服务的密钥 |
| `EMBEDDING_BASE_URL` | 默认百炼 `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | 默认 `text-embedding-v4` / `1024`，两者必须匹配 |
| `EMBEDDING_BATCH_SIZE` | 单次请求的文本条数，百炼最多 10 条 |
| `MILVUS_DB_PATH` | 默认 `./milvus_library.db`（Milvus Lite 本地文件）；用服务端时改设 `MILVUS_URI=http://localhost:19530` |
| `TOP_K` | 每次检索返回的片段数，默认 3 |

> **关于「DeepSeek 向量模型」**：DeepSeek 官方只提供对话模型（`deepseek-flash` /
> `deepseek-v4-pro`），**没有 embedding 接口**，`https://api.deepseek.com/v1/embeddings`
> 会返回 404。所以向量化必须另找服务：
>
> - 默认方案：阿里百炼 `text-embedding-v4`（1024 维），直接复用课程根目录已有的 key；
> - 备选：硅基流动 `BAAI/bge-m3`（1024 维），去 <https://cloud.siliconflow.cn> 注册后
>   把 `.env` 里注释掉的 5 行取消注释即可；
> - 也可以用本地 Ollama 跑 bge-m3，不需要任何 key。
>
> 不管换哪个，`EMBEDDING_MODEL` 和 `EMBEDDING_DIM` 必须匹配，否则建库会报维度不一致。

## 知识库设计决策（不是刷分，是模拟真实文档）

初版知识库的每条图书信息都是固定字段、200 字以内一个 chunk，问题关键词与文档用词
几乎逐字重合，干扰项又都是语义无关的失物/报修记录。结果是 Recall@3 ≈ 94%、
MRR ≈ 0.897——这是数据结构的必然，**不能说明检索能力**。因此按「真实文档就是这样」
的原则做了以下改动：

| 改动 | 原因 |
| --- | --- |
| 删除 `【旧通知（已作废）】`、`【最新…当前有效！】` 等标签，只保留发布日期 | 真实通知不会自带「已作废」徽标；新旧冲突必须靠日期推理，否则等于把答案喂给模型 |
| 删除文档里的 `别名：花书` / `（西瓜书）` | 别名只存在于用户口中；评测题里用「花书」「西瓜书」才是真实的同义检索 |
| 借阅规则改为按本科生/研究生/教师分列的表格 | 真实规则分角色；表格会被 200 字切分打散，还要先判断读者身份才能取值 |
| 三楼 302 号书架 A 区扩到 3 本图书 | 制造 3 组「同书架、不同答案」的近邻干扰，原来的 1 处不足以暴露混淆错误 |
| 新增 30 条带具体数字的无关通知/统计 | 真实知识库有大量冗余段落；数字（人数、册数、金额、临时活动额度）会把向量检索误召回 |
| 评测集打 L0-L5 难度标签 | 总分会掩盖能力边界；分层准确率能说明系统到底会做哪一类题 |

改完后旧指标不再可比，请按下面的步骤重建向量库再评测。测试中若某类题掉分，那是
**题库变真实了**，不是回归 bug。

实测对比（2026‑11‑21，TOP_K=3；旧报告 `evaluation/report.json`，新报告
`evaluation/report_v2.json` / `retrieval_v2.json`）：

| 指标 | 改造前 | 改造后 |
| --- | --- | --- |
| 文档 chunk 数 | 38 | 52 |
| Recall@3 | 94.12% | 87.18% |
| MRR | 0.897 | 0.821 |
| 回答准确率 | 92.86% | 91.49% |
| 工具选择准确率 | 100% | 100% |

分层结果（新）：L1 检索/回答 100%/100%，L2 0%/0%（别名只留在评测题里），
L3 100%/100%，L4 50%/0%（跨片段求和失败），L5 拒答 100%（L5 无关键词，
按检索指标必然记为未命中）。总分只掉几个点是因为 L1 仍占 26/39；真正的能力
边界要看分层表，而不是总分。

## 使用步骤

1. **建库**（首次运行或知识库更新后执行）：

   ```bash
   python rag_module.py
   ```

   脚本会删除旧集合、重新切分写入，并跑一次测试检索。

   只看集合状态：

   ```bash
   python rag_module.py --check
   ```

2. **启动对话**：

   ```bash
   python main.py
   ```

   `main.py` 启动时会检查向量库，为空则自动建库。

3. 输入 `exit` / `quit` / `q` 退出。

### 借还操作（写工具 + 人工确认）

`borrow_manage_tool` 支持 borrow / renew / return，直接读写 `data/loans.json`。
图里加了 `approve` 节点：只要一轮工具调用中含写工具，就通过 LangGraph `interrupt`
挂起，命令行弹出确认，输入 `yes` 才真正执行，`no` 会写入 `USER_REJECTED` 结果并
让 agent 自行说明操作已取消。读工具不受影响。

图必须带 checkpointer（`MemorySaver`），否则 `interrupt` 会报错。命令行会话用
`thread_id="cli-session-1"` 保持上下文；评测脚本给每题独立 `thread_id`，并在
`__interrupt__` 时自动 `Command(resume="yes")`。

可试这三条（当前日期 2026-11-21）：

```
帮我续借《机器学习》              → 确认后新到期日 2026-12-31
《数据库系统概论》再帮我续借一次   → ALREADY_RENEWED（已续借过，比超期判断优先）
我要还《深度学习》                → RETURNED，逾期 10 天、罚款 1.0 元
```

`loans.json` 除两条题设记录外，还种了《深度学习》一条逾期记录，用来让第三条命令
真的走到罚款计算；测试后文件会还原到初始状态。

## 评测与消融

解释器同样是 `D:\Edge_Download\Miniconda\envs\langchain1.2\python.exe`。
在项目根目录执行：

```bash
# 全量评测：检索指标 + 回答准确率 + 工具选择准确率 + 轨迹/token + 多步重试指标
python evaluation/evaluate.py --mode all

# 只跑检索（不花 LLM token）；--limit 用于快速试跑
python evaluation/evaluate.py --mode retrieval
python evaluation/evaluate.py --mode agent --limit 5

# 参数消融：baseline / topk_1 / topk_5 / temp_0 / chunk_400 / chunk_100
python evaluation/ablation.py --limit 12        # 快速对比
python evaluation/ablation.py                   # 全量对比（更慢但更稳）
```

产物：

- `evaluation/report.json`：每题的 `trace`（`tool_calls`、工具轮数、LLM 次数、
  token 用量）以及四项汇总指标；最新一次回归为 `evaluation/report_v5.json`，
  v3 是仅激活重试循环的快照，v4 是加入写工具但未修正身份题评分要点的快照；
- `evaluation/ablation_report.json`：各参数组的指标对比与相对 baseline 的差值；
- `evaluation/ablation/<变体>/report.json`：每组消融的完整明细。

`agent` 汇总里另有多步重试指标：`avg_tool_rounds`、`multi_step_rate`、
`max_tool_rounds_seen`、`rounds_distribution`。工具返回结构化 JSON
（`ok` / `code` / `data` / `hint`），仅在真正零召回时给 `NO_HIT`；相似度分数
不做阈值门控（实测正确/错误检索分数只差 0.0002，任何阈值都会误杀）。片段是否
答非所问由 agent 自己判断，并按 system prompt 换词重试最多 2 次。

每份报告的 `retrieval.by_level` / `agent.by_level` 给出分层结果：

| 层级 | 定义 | 例子 |
| --- | --- | --- |
| L1 | 单跳事实，一个片段可答 | 「图书馆一共几层」 |
| L2 | 同义/别名，问法与文档用词不同 | 「西瓜书在哪」 |
| L3 | 条件规则或版本冲突 | 「研究生能借几本」「现在几点关门」 |
| L4 | 多跳聚合，跨片段求和 | 「302 号书架一共多少本」 |
| L5 | 库外拒答 | 「馆长是谁」 |
| L0 | 不该调工具的闲聊/常识/实时信息 | 「今天天气怎么样」 |

L0 不计入检索指标，只统计工具选择准确率。

评测集里的 `expect_tool_call=false` 是闲聊、常识、实时信息等**不该调工具**的题，
用于计算工具选择准确率；这些题不参与检索指标统计。

## 常见问题

- **提示「未配置向量化密钥」**：`.env` 里的 `EMBEDDING_API_KEY` 为空。
- **建库报「batch size is invalid」**：当前向量服务单次批量上限更小，把
  `EMBEDDING_BATCH_SIZE` 调小（百炼是 10）。
- **建库报维度不一致**：`EMBEDDING_MODEL` 与 `EMBEDDING_DIM` 不匹配，改完删除
  `milvus_library.db` 重新建库。
- **检索报 collection is in state 'released'**：Milvus Lite 重启后集合需要重新 load，
  代码里已在每次检索前自动 `load_collection`，更新代码即可。
- **回答用到了过期时间**：知识库里的历史通知没有「已作废」标签，系统提示词要求按
  发布日期判断；如果仍不稳定，可以把 `TOP_K` 调大（如 5）或调小 `CHUNK_SIZE`。
- **改完 `library_knowledge.txt` 后指标没变**：向量库还是旧的，必须先执行
  `python rag_module.py` 重建，再跑评测。
