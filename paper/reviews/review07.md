# Independent review 07 — reproducibility and data separation

**Vote: NO, for submission in its present form.** A revised, explicitly retrospective empirical/methodological paper could merit a YES without collecting a new benchmark, but the present source misdescribes the training fields and makes mutually inconsistent claims about test access. This is not a finding that the numerical results are fabricated or that test labels were used to fit the reported prediction pipeline.

## Scope and evidence

I read the extracted original manuscript and inspected the supplied matching source, particularly `paper/impl.tex`, `paper/theory.tex`, `paper/method.tex`, `paper/discussion.tex`, `common.py`, `krr_oof.py`, `gen_preds.py`, the network-training code, `stack_perpixel.py`, `stack_correct.py`, `hybrid.py`, `campaign/seed_pipeline.py`, `campaign/seed_pipeline_lowdata.py`, `campaign/correct_stack.py`, `campaign/jpl_seeded.py`, and the conformal scripts. I did not consult other reviewers' reports. I ran `python campaign/audit_reported_macros.py` from the repository. It exited successfully with **0 macros disagreeing with the artifacts**, including the reported high-data, low-data, OCO-2, uncertainty and ensemble figures. This verifies aggregation against stored records, not their underlying predictions or an independent rerun.

## Positive findings

The structural-mechanics split code keeps the prescribed 20,000 training-pool/20,000 test boundary fixed and draws validation only from the training pool. The explicit weight and kernel selection paths I inspected minimize validation scores. `correct_stack.py` decides between corrected and uncorrected models using `corr_val < stack_val`; `stack_perpixel.py` compares the two rules on the validation half called B; OCO-2 checkpoints, kernels and coordinate selection use validation labels. Test scores are calculated and printed, but I did not find them entering these prediction optimizers. The sixty-predictor test-optimal weights are explicitly labelled hindsight diagnostics and should not be conflated with deployable selected weights.

The extensive per-seed JSON release and passing macro audit are substantial strengths. The manuscript also already acknowledges repeated benchmark access, same-validation reuse, the pending centering rerun, absent raw arrays, the lack of a proven intrinsic data floor, and the conditional nature of several theoretical results. These disclosures make an honest narrower paper possible.

## Mandatory corrections

### 1. The entire residual mean is not cross-fitted

The paragraph after the conditional power-function theorem says each training row's members are out of fold and denotes the correction-training mean by `m_cf`. The implementation contradicts that description. `gen_preds.py` loads one already-trained neural checkpoint and evaluates it on `tr`, `va`, and `te`; its neural `*_predtr.npy` predictions are in-sample. Both stacking scripts concatenate those in-sample neural predictions with `krr_oof_train.npy`. Only the standalone KRR channel is out of fold. The refiner is itself trained on the same training rows, even though one of its input channels uses the OOF KRR field.

This is not automatically an invalid residual-regression procedure and is not test leakage. It is, however, an objectively incorrect description with consequences for the theorem-to-estimator comparison. Replace `cross-fitted mean` with an explicit description of the mixed training predictions. Define the saved training mean values as a matrix `M_tr`, the deployed mean as `m_full`, and the mismatch as `Delta=M_tr-m_full(X)`. Then, with `c(u)=k(u,X)(K+n lambda I)^{-1}`, the deployed error obeys the pathwise bound

`||G(u)-m_full(u)-c(u)[G(X)-M_tr]|| <= ||G-m_full||_K * Ptilde_lambda(u) + ||c(u) Delta||`.

This expression does not presume that all members were cross-fitted. The second term has not been measured. Do not claim the single-mean theorem directly certifies the deployed correction. A fully cross-fitted new campaign is optional if the method is described honestly; it is mandatory only if complete cross-fitting remains part of the claimed method.

### 2. Historical OOF centering and current code are different experiments

`impl.tex` says the reported OOF fields were centered using the pooled target mean over all 19,000 training rows, including the held-out fold. Current `krr_oof.py` instead computes `muY=Y[fit].mean(0)` inside the fold loop and comments that the released fields used the pooled mean. Therefore the checked-in script does not reproduce the historical fields exactly, and the refiner plus all its downstream stacks and corrections have not been rerun under the changed procedure.

This is **within-training-fold target leakage**, not contamination by the external test set. It does not by itself invalidate held-out scores of the actual historical algorithm. Its effect cannot be declared negligible from the stated `1/sqrt(n)` heuristic: no measured sensitivity is provided, and the final kernel-prediction effect includes a linear kernel weight factor. For a fitting fold F, the pooled-vs-local prediction difference is exactly `(1-k(u,F)(K_F+|F|lambda I)^{-1}1)*(mu_pool-mu_F)`.

Minimal repair has two honest routes: (a) reproduce the historical centering in an explicitly archived mode, label the existing results as results of that mode, and remove every claim of strict OOF purity; or (b) rerun the corrected OOF fields, refiner and affected downstream stages and replace all affected results. A clean rerun cannot be asserted until actually completed. A text-only revision should choose route (a), clearly and prominently.

### 3. The test-access statement contradicts the documented study

The beginning of `impl.tex` states that the test set was never touched during development. Its later `Test-block access` paragraph correctly says the same block was inspected repeatedly across campaigns, ablations, learning curves, hard-case analyses, conformal calibration and post-review controls. The earlier absolute claim must be deleted. Retain the precise narrower claim supported by code: recorded optimization and per-run selection used training/validation data, while the study repeatedly inspected the public test block.

This distinction matters. Repeated evaluation alone does not establish direct test-label optimization, but a study adapted after observing benchmark results cannot use ordinary fixed-procedure uncertainty calculations as proof of generalization of the whole research process. The comparisons can remain explicitly descriptive of the public block. A genuinely untouched final dataset is necessary only for a confirmatory claim, not automatically for a carefully framed empirical paper.

### 4. The validation halves are not independent of all fitted members

Neural checkpoints and kernel settings use the full validation split before the per-pixel half-split comparison. Accordingly B is held out from fitting the pixel weights, but not from building/selecting every member. The final pixel weights are subsequently fitted on all validation labels and the same labels tune the residual kernel. This is ordinary adaptive validation reuse; it does not become test leakage merely because multiple stages use it. The main theory introduction already says fixed-candidate guarantees do not certify this pipeline. Make the method and supplement use that same language consistently: remove unqualified `honesty protocol` implications and avoid describing the half-split as an independent end-to-end validation guarantee.

### 5. Distinguish the three levels of reproducibility

The release supports reproducing table aggregation from JSON, rerunning source code after acquiring data, and (only on request) inspecting retained high-dimensional prediction/residual arrays. Several passages say arrays are released and then later say they are not part of the release. State these access levels consistently. The 3.84 GB structural residual archives and approximately 250 MB OCO-2 prediction arrays should receive stable links/checksums when actually deposited. Their absence is a limitation rather than grounds to allege fabricated results, but array-based claims are not publicly independently verifiable from this bundle alone. Do not describe a future deposit as already done.

### 6. Seed variability needs one explicit exception

The high-data seed changes both validation split and training randomness. `campaign/seed_pipeline_lowdata.py` deliberately fixes the first-1,250/last-250 split at every seed and varies neural training only. The general seed-protocol paragraph names ClimSim as the sole exception but should also name the low-data fixed-split exception. The low-data standard deviation measures training variability conditional on that one split, not variability over low-data sampling choices.

## Publication assessment after repairs

I would reconsider to YES for a journal accepting careful applied methodological studies once these factual contradictions are removed, historical results are clearly attributed to the actual historical algorithm, and the claims remain finite-benchmark and conditional. The manuscript has useful replication, diagnostic analyses, and coherent empirical comparisons. I do not require inventing new experiments to repair the text, and I do not require withdrawal of every existing benchmark number because of training-fold leakage. I do require that a corrected implementation not be advertised as the experiment behind unchanged numbers.

**Final vote on the submitted version: NO. The main reproducibility issues are repairable; direct test-label leakage into the reported predictor optimizers was not established by this review.**
