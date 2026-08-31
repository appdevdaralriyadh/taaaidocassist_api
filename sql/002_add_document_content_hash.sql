-- Phase 2: adds content-hash based duplicate detection to DarAI_Documents.
-- This is the only schema change Phase 2 needs beyond spec §4's original
-- DDL -- everything else (DarAI_DocumentChunks included) fits as-is.
--
-- Safe to run more than once: only adds the column if it isn't already
-- there.

IF NOT EXISTS (
    SELECT 1
    FROM sys.columns
    WHERE object_id = OBJECT_ID('dbo.DarAI_Documents')
      AND name = 'ContentHash'
)
BEGIN
    -- SHA-256 digest (32 bytes) of the uploaded file's raw bytes. Used to
    -- flag likely duplicate uploads -- see app/services/ingestion.py.
    ALTER TABLE dbo.DarAI_Documents
    ADD ContentHash VARBINARY(32) NULL;
END
GO
