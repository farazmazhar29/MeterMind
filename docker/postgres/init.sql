-- Bootstrap only. Postgres runs everything in /docker-entrypoint-initdb.d
-- exactly once, when the data directory is empty. Anything that might need to
-- change later must NOT live here -- see docker/postgres/schema.sql, which is
-- applied idempotently from Python on every dataset build.

CREATE EXTENSION IF NOT EXISTS vector;
