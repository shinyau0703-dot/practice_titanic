-- 建第二張表：fold_results
CREATE TABLE IF NOT EXISTS app.fold_results (
  id      SERIAL PRIMARY KEY,
  run_id  INTEGER NOT NULL,
  fold    INTEGER NOT NULL,
  acc     NUMERIC,
  f1      NUMERIC,
  CONSTRAINT fk_fold_run
    FOREIGN KEY (run_id) REFERENCES app.runs(id) ON DELETE CASCADE
);

-- 常用索引與唯一性（同一個 run，每個 fold 只能一筆）
CREATE INDEX IF NOT EXISTS idx_fold_results_run_id ON app.fold_results(run_id);
CREATE UNIQUE INDEX IF NOT EXISTS ux_fold_results_run_fold ON app.fold_results(run_id, fold);
