-- v1 数据模型：7 张表。向量不进 SQLite（存 Chroma，id 用 chunk_id 对齐）。
-- 建表顺序按外键依赖：papers -> sections -> blocks -> chunks -> assets -> chunk_assets -> parse_cache。
-- 约定：结构化字段（bbox / block_ids）存 JSON 字符串；时间统一存 TEXT。

CREATE TABLE IF NOT EXISTS papers (
    paper_id      TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    citation      TEXT,   -- 作者 / 年份 / 会议的自由文本，如 "Saadi et al. · ACM MM 2025"
    pdf_path      TEXT,
    mineru_dir    TEXT,
    content_hash  TEXT,   -- PDF 字节的 sha256：判断"是不是同一份文件"
    parse_version TEXT,   -- 解析配置指纹：判断 blocks/assets 能不能沿用
    index_version TEXT,   -- 切分+embedding 指纹：判断 chunks/向量能不能沿用
    status        TEXT NOT NULL DEFAULT 'pending',  -- pending | parsed | indexed | failed
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 章节树。消费方见 PLAN §2：切分边界、index_text 的章节路径、证据卡定位串。
CREATE TABLE IF NOT EXISTS sections (
    section_id   TEXT PRIMARY KEY,
    paper_id     TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    parent_id    TEXT REFERENCES sections(section_id) ON DELETE CASCADE,
    level        INTEGER NOT NULL,  -- 1 = 一级标题
    order_index  INTEGER NOT NULL,  -- 论文内全局顺序，建树与排序都靠它
    title        TEXT NOT NULL,
    section_type TEXT,              -- abstract/intro/related/method/experiment/conclusion/reference/other
    page_start   INTEGER,
    page_end     INTEGER,
    UNIQUE (paper_id, order_index)
);

-- 原子块：永不切分的最小单位。公式/表格/图片各自一块，本体另存 assets。
CREATE TABLE IF NOT EXISTS blocks (
    block_id      TEXT PRIMARY KEY,
    paper_id      TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    section_id    TEXT REFERENCES sections(section_id) ON DELETE SET NULL,
    order_index   INTEGER NOT NULL,  -- 论文内全局顺序（= MinerU content_list 下标）
    block_type    TEXT NOT NULL,     -- text | title | list | caption | image | table | equation | other
    heading_level INTEGER,           -- 只有 title 块有：MinerU 的 text_level
    text          TEXT,
    latex         TEXT,
    html          TEXT,
    caption       TEXT,              -- 图注/表注，与本体分开存
    page_idx      INTEGER,
    bbox          TEXT,              -- JSON [x0, y0, x1, y1]
    image_path    TEXT,              -- 相对 mineru_dir 的路径，如 images/abc.jpg
    UNIQUE (paper_id, order_index)
);

-- 检索与生成单元。content 给生成看，index_text 进向量。
-- paper_id 是冗余字段（能从 section 推），但每路召回都要按它过滤，不做 join：
-- 这是"问论文 A 不会答成论文 B"的地基。
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    paper_id    TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    order_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    index_text  TEXT NOT NULL,     -- 章节路径 + caption + 正文
    token_count INTEGER,
    block_ids   TEXT NOT NULL,     -- JSON 数组，指向 blocks.block_id
    page_start  INTEGER,
    page_end    INTEGER,
    chunk_type  TEXT NOT NULL DEFAULT 'text',  -- text | table | equation | figure
    UNIQUE (paper_id, order_index)
);

-- 关键词通道：`chunks.index_text` 的 FTS5 镜像，跟 chunks 在**同一个事务**里先删后写。
-- 不用 external content 模式：那要求 rowid 跟 chunks 主键对齐，而我们的主键是 TEXT。
--
-- **tokenize='trigram' 不是默认值，是实测选的**：默认的 unicode61 对中文完全无效
-- （一整句中文是一个 token，"微表情"永远查不到），trigram 中英通吃。代价是短于
-- 3 个字符的词查不到（"F1"/"AI"）—— 编号类查询走 assets.label_norm 定向，不指望这里。
-- **改分词器必须 DROP 重建 + 重跑切分**，索引是按当时的分词器建的。
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    index_text,
    paper_id UNINDEXED,
    section_type UNINDEXED,
    chunk_type UNINDEXED,
    tokenize='trigram'
);

-- 资产：表格 HTML / 公式 LaTeX / 原图路径。只作为附件进生成上下文，不进向量。
CREATE TABLE IF NOT EXISTS assets (
    asset_id    TEXT PRIMARY KEY,
    block_id    TEXT NOT NULL REFERENCES blocks(block_id) ON DELETE CASCADE,
    paper_id    TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    asset_type  TEXT NOT NULL,  -- table | formula | figure
    page_idx    INTEGER,
    bbox        TEXT,
    raw_content TEXT,           -- 表格 HTML / 公式 LaTeX
    image_path  TEXT,
    caption     TEXT,           -- 图注/表注**原文**，没有就是空（不再补假编号）
    -- 归一化编号（身份）：table:3 / figure:2 / formula:7，抽不出就是 NULL。
    -- **交叉引用只认这一列**：caption 里可能压根没编号，或者写的是罗马数字。
    -- 检索文本不存列，由 assets.index_text_of() 现算（表 → 表注+表头，图 → 图注，
    -- 公式 → 编号）。
    label_norm  TEXT
);

-- chunk ↔ 资产 多对多。relation='primary' 表示资产本体就落在这个 chunk 里。
CREATE TABLE IF NOT EXISTS chunk_assets (
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    asset_id TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
    relation TEXT NOT NULL DEFAULT 'primary',
    PRIMARY KEY (chunk_id, asset_id)
);

-- 解析产物复用表：删论文时**不删**它，也不删 storage/mineru/ 里的目录。
-- 下次上传同一份 PDF 且解析配置没变，就能直接从本地 content_list.json 重建 blocks，
-- 跳过 MinerU 那次花钱的调用（几秒 vs 几分钟）。
-- 注意：这里刻意不做外键指向 papers —— 论文删了这行要留着。
CREATE TABLE IF NOT EXISTS parse_cache (
    content_hash   TEXT NOT NULL,
    mineru_version TEXT NOT NULL,  -- MinerU 调用参数指纹；**不含** converter 版本
    paper_id       TEXT NOT NULL,  -- 仅供参考：当初是谁解析出来的
    mineru_dir     TEXT NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (content_hash, mineru_version)
);

CREATE INDEX IF NOT EXISTS idx_papers_hash ON papers(content_hash);
CREATE INDEX IF NOT EXISTS idx_sections_paper ON sections(paper_id);
CREATE INDEX IF NOT EXISTS idx_sections_type ON sections(paper_id, section_type);
CREATE INDEX IF NOT EXISTS idx_blocks_paper ON blocks(paper_id, order_index);
CREATE INDEX IF NOT EXISTS idx_blocks_section ON blocks(section_id);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id, order_index);
CREATE INDEX IF NOT EXISTS idx_assets_paper ON assets(paper_id, asset_type);
CREATE INDEX IF NOT EXISTS idx_assets_block ON assets(block_id);
CREATE INDEX IF NOT EXISTS idx_assets_label ON assets(paper_id, label_norm);
CREATE INDEX IF NOT EXISTS idx_chunk_assets_asset ON chunk_assets(asset_id);

-- ======================================================================
-- 会话（Step 8c）：多轮对话用。**和论文数据完全解耦** —— 删论文不该动会话，
-- 所以这里没有指向 papers 的外键（论文 id 只当"范围"记着，查不到就退回全库）。
-- ======================================================================

-- scope_paper_id / scope_paper_ids 是 **sticky scope**：这一轮问到哪几篇，
-- 下一轮追问"它呢"就默认还在这几篇里找。NULL = 全库。
-- 两个字段分开存：单篇走 scope_paper_id（简单、好读），多篇走 scope_paper_ids（JSON 数组）。
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    title           TEXT NOT NULL DEFAULT '',   -- 首次提问后自动填（取问题前 30 字）
    scope_paper_id  TEXT,
    scope_paper_ids TEXT,                        -- JSON 数组字符串
    message_count   INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

-- 一轮对话消息。assistant 的 citations 存 JSON（引用校验结果），前端刷新后还能复原。
-- order_index 从 1 开始、会话内连续：**靠它排序，不靠时间戳**（同一秒内两条很常见）。
CREATE TABLE IF NOT EXISTS messages (
    message_id  TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role        TEXT NOT NULL,                 -- user | assistant
    content     TEXT NOT NULL,
    citations   TEXT,                          -- JSON，只有 assistant 有
    created_at  TEXT NOT NULL,
    order_index INTEGER NOT NULL,
    UNIQUE (session_id, order_index)
);

CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, order_index);
