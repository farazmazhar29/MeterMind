-- Applied by src/metermind/data/load_postgres.py on every dataset build.
-- Every statement must be idempotent: this file is re-run, not migrated.

-- The site register. Mirrors data/generated/sites.yaml, which is the source of
-- truth; this table exists so the Week 2 retrieval tools can join site metadata
-- against the document corpus in one query.
CREATE TABLE IF NOT EXISTS sites (
    site_id         TEXT PRIMARY KEY,
    name            TEXT        NOT NULL,
    category        TEXT        NOT NULL,
    city            TEXT        NOT NULL,
    meter_id        TEXT        NOT NULL,
    floor_area_m2   INTEGER,
    base_load_kw    NUMERIC(10, 2),
    peak_load_kw    NUMERIC(10, 2),
    commissioned_on DATE,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS sites_category_idx ON sites (category);

-- Week 2: the RAG corpus. Created now so that `docker compose up` produces a
-- schema that is complete enough to reason about, but left empty until the
-- corpus generator exists.
--
-- EMBEDDING DIMENSION: 384 = sentence-transformers/all-MiniLM-L6-v2, chosen so
-- the project runs offline with no second API key (Claude has no embeddings
-- endpoint). If you switch to a hosted embedding model this must change, and
-- changing a vector column's dimension requires dropping the column -- so make
-- that call before you embed 200 documents, not after.
CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    site_id     TEXT REFERENCES sites (site_id) ON DELETE CASCADE,
    doc_type    TEXT        NOT NULL,   -- maintenance_log | incident | contract | commissioning | shift_notice
    title       TEXT        NOT NULL,
    body        TEXT        NOT NULL,
    doc_date    DATE,
    -- 'synthetic' for everything in this corpus. Kept explicit so no consumer
    -- can mistake generated text for a real operational record.
    source      TEXT        NOT NULL DEFAULT 'synthetic',
    embedding   vector(384)
);

CREATE INDEX IF NOT EXISTS documents_site_idx ON documents (site_id);
CREATE INDEX IF NOT EXISTS documents_date_idx ON documents (doc_date);

-- No ANN index yet. Below a few thousand rows an exact scan is faster than
-- IVFFlat, and IVFFlat needs populated data to train its lists. Add
-- an hnsw/ivfflat index in Week 2 once the corpus is loaded and measured.
