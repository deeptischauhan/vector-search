-- Keel Vector Search — source table preparation
-- Run in the Databricks SQL Editor. The table itself was created by loading
-- data/articles_corpus.csv via Catalog > Create or modify table from files
-- (catalog = workspace, schema = cpt, table = articles).
--
-- For reference, the equivalent schema is:
--   CREATE TABLE workspace.cpt.articles (
--     id       STRING,
--     title    STRING,
--     url      STRING,
--     source   STRING,
--     category STRING,   -- business_acumen | peripheral | positioning
--     skill    STRING,
--     content  STRING    -- the text that gets embedded
--   );

-- 1) Enable Change Data Feed (required for a Delta Sync index — lets the index
--    re-embed only the rows that changed, not the whole table on every sync).
ALTER TABLE workspace.cpt.articles
  SET TBLPROPERTIES (delta.enableChangeDataFeed = true);

-- 2) Confirm the primary key is unique (rows should equal distinct_ids = 23).
SELECT count(*) AS rows, count(DISTINCT id) AS distinct_ids
FROM workspace.cpt.articles;

-- 3) Declare id as a real primary key (Vector Search needs a unique key per row).
ALTER TABLE workspace.cpt.articles ALTER COLUMN id SET NOT NULL;
ALTER TABLE workspace.cpt.articles ADD CONSTRAINT articles_pk PRIMARY KEY (id);

-- 4) Extend deleted-file retention so a Triggered sync never fails by looking
--    for data files that were vacuumed (default retention is 7 days).
ALTER TABLE workspace.cpt.articles
  SET TBLPROPERTIES ('delta.deletedFileRetentionDuration' = 'interval 30 days');
