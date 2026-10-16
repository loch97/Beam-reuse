# Archive verification

Verified on 2026-10-16 using Python 3.12.

- All four numerical regression tests pass: finite-tone leakage, exact grant-rank probabilities, simulation comparison and RT replay.
- The one-frame `broad_U1` and `narrow_U0` runs reproduce the original implementation's resource metrics exactly. Saved intermediate arrays agree within `1e-12`.
- The paired 4M/16M RT example reproduces all 12 original records' candidate indices, detection probabilities, powers and communication scores within `1e-12`.
- The archive's inference code was run on the original complete 384-block simulation batch. All resource and robustness summary values, paired statistics and confidence intervals match the frozen summaries exactly. This checks aggregation without rerunning the full simulation batch.
- A fresh 10,000-ray CPU trace completed through the new scene-loading and path-extraction entry point. This is an execution smoke check, not a new paper result. Full 4M/16M/64M tracing was not rerun during cleanup.
- The release ZIP was extracted into a separate directory and reproduced all five paper figures. Their rendered pixels match the current manuscript figure PDFs exactly at 144 dpi, including their canvas sizes.
- Code has no `try/except` handlers, automatic backend switching, retry loops, file locks, background completion polling or parent-project import-path modifications.

`archive_manifest.json` records the files and SHA-256 checksums included in the archive. Generated outputs and interpreter caches are excluded from the ZIP and ignored by Git.
