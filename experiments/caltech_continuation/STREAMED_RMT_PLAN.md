# Bounded input and metric preparation

These modules are local preparation for a future authorized DGX run. They do not launch a run or replace the corrected historical driver.

The historical predictor shape is 60 × 20,000 × 1,681. One float32 pool occupies 8,068,800,000 bytes (7.5147 GiB). The list of individual arrays and the new `np.stack` output coexist, giving at least 16,137,600,000 bytes (15.0293 GiB) before targets, architecture means, or later copies. This already exceeds the prepared 10 GiB process ceiling.

`blocked_prediction_pool.py` verifies manifest hashes and headers, opens one read-only NPY mapping at a time, copies only selected cases and pixels, and closes each mapping. A 60 × 1,000 × 32 float32 output is 7.3242 MiB; the largest float64 indexing temporary is 0.2442 MiB. The budget covers these prediction-data buffers, not total process RSS, mapping address space, Python overhead, or OS file cache. A host resource limit and fresh admission remain necessary. Hashes must be checked again before adopting a completed run's output.

`streamed_error.py` accumulates squared residuals and target norms, checks case identity/order and tile shape, rejects duplicate tiles, and requires complete case/pixel coverage before returning mean per-case relative error. A partial calculation cannot acquire a completed metric merely because it produced a scalar.

Matching array dimensions does not prove that predictor rows align with target rows. The full runner must also bind the test-index array and each predictor's native row-order provenance; those records are not supplied by an NPY header. The low-level reader does not claim to validate that semantic relationship.

Full orchestration still needs multiple passes. Accumulate each global convex objective over all pixels using whole-case target norms; select global ridge penalties from scores summed over all pixels; then compute fold candidate errors with those fixed training-fold fits. Fitting separate global weights per pixel block would change the estimator. Final evaluation must use the same pinned input/split/metric contract, with a distinct output and a full integrity recheck.

The file reader and metric accumulator are tested against separate full-array calculations on tiny fixtures. These checks are not a peak-RSS measurement, a real-data score, a DGX deployment, or proof that the complete study fits the process ceiling.

The 11 small tests passed with BLAS restricted to one thread:

```text
python -m unittest -v test_streamed_error.py test_blocked_prediction_pool.py
```
