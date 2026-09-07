# Experiment code and records

Implementation and record paths below are relative to the
[public experiment repository](https://github.com/yspennstate/neural-means-kernel-corrections)
at commit `39d217077899a05766b1b30c8275a1e488a2f071`.
The article and supplement specify the reported methods, metrics, and run settings.
Braces and asterisks denote groups of files; `s0..9` denotes seeds 0 through 9.
A suffix after `::` identifies a function or a JSON field.

For the mechanics kernel fields, select `--target-centering pooled` for the
main pooled-centering protocol and `--target-centering fold-local` for the
corresponding sensitivity variant. The current `krr_oof.py` defaults to
fold-local centering; select the required setting before the downstream
pipeline uses those fields.

| Experiment | Implementation | Records |
|---|---|---|
| Mechanics first five-member / four-member ablation | campaign/seed_pipeline.py; campaign/finalize_fno.py; campaign/stack_correct_seeded5.py; campaign/ensemble_seeded5.py; stack_perpixel.py; campaign/correct_stack.py | campaign/{pix4,pix5,ens5}/; campaign/collected/secmom_seeded5.json |
| Mechanics complete six-member and paired five-member controls | campaign/seed_pipeline.py with EP_FNO=EP_UNET=100; campaign/run_b5.sh; campaign/choose_member_set.py | campaign/collected/dgx/sm_seed_s0..9.json; campaign/collected/dgx/runs/sm_s*/; campaign/collected/secmom6_seeded.json; campaign/collected/secmom5c_seeded.json |
| Mechanics low data | campaign/seed_pipeline_lowdata.py; hybrid.py; campaign/krr_multiscale_lowdata.py | campaign/collected/box1/ld_s0..9.json; runs/krr_lowdata_strict.json |
| Neural architectures and augmentation | train_mlp.py; train_mlp_refine.py; train_fno.py; train_unet.py; train_vit.py; models.py; gen_preds.py | Per-member JSON args under campaign/collected/dgx/runs/sm_s*/ |
| Symmetry diagnostics and TTA comparisons | Neural evaluate routines above; archive macro/record aggregation | runs/mirror_check.json (mean: 0.0113964, covariance: 0.0146094); member test/test_tta fields |
| KF regularizer | train_mlp.py::kf_loss; campaign/dgx_checks/p1_members_eval.py | campaign/collected/dgx/p1_members_eval.json::A_kf_members |
| Residual correlations / floors / diversity | analyze_corr.py; campaign/secmom_seeded5.py; campaign/secmom6.py; campaign/audit_reported_macros.py | runs/corr_div.json; runs/shared_div.json; runs/enstheory_div.json; campaign/collected/secmom*.json |
| Sixty-predictor pool / seeds / drop-one / pooled pipeline | campaign/seedarch.py; campaign/dropone.py; campaign/pool_pipeline.py; campaign/dgx_checks/ens_rmt_dgx.py; campaign/dgx_checks/p1_members_eval.py | campaign/collected/dgx/{seedarch,dropone,pool_pipeline,sm_ens_rmt,p1_members_eval,deployed_on_ev}.json |
| Learning curves | campaign/run_curve.sh; train_mlp.py; krr_scaling.py; campaign/run_analysis.sh | campaign/collected/dgx/learning_curve.json |
| GP-power / raw / disagreement UQ | campaign/uq_conformal_plam.py; campaign/uq_conformal.py; campaign/conformal_seeded5.py; campaign/dgx_checks/uq_fig_dump.py; campaign/fig_uq_seeded.py | campaign/collected/dgx/uq_plam_seeded.json; campaign/collected/dgx/sm_seed_s*.json::uq; campaign/conf5/ |
| RKHS-norm diagnostic | campaign/dgx_checks/sm_norm_check.py; campaign/dgx_checks/jpl_alignment_check.py | campaign/collected/dgx/sm_norm_check_hpix_s{0,1,2}.json; campaign/collected/dgx/jpl_alignment_o2_s0.json |
| Gram spectrum / effective dimension | uq_spectra.py; figure input records | paper/figs/spectra_provenance.json; paper/supp_experiments.tex |
| RMT / learned-kernel controls | campaign/dgx_checks/{rmt_krr,kf_kernels,ens_rmt_dgx,residual_spectrum_test}.py; campaign/verify_synthetic.py | campaign/collected/dgx/{p1_members_eval,sm_ens_rmt}.json; campaign/collected/dgx/exp/residual_spectrum_s0.json |
| OCO-2 three-band main / ridge readouts | campaign/jpl_seeded.py; jpl_data.py; campaign/gen_oco_table.py; campaign/gen_oco_ridge_table.py | campaign/collected/box*/oco_*_s*.json; campaign/collected/dgx/oco_ridge/; campaign/collected/dgx/oco_ridge_report.json |
| OCO second moments, admission, margin analysis | campaign/dgx_checks/oco_ensemble_recheck.py; campaign/oco_secmom_score.py; campaign/oco_secmom_merge.py; campaign/oco_margin_sigma.py | campaign/collected/dgx/oco_ensemble_recheck.json |
| OCO raw-input/ARD scaling and geometry | campaign/scaling_seeded.py; jpl_diagnostics.py; campaign/dgx_checks/jpl_alignment_check.py | campaign/collected/box4/a6_scaling_*.json; campaign/collected/dgx/jpl_alignment_o2_s0.json |
| WCO2 architectural/Fourier sweep | jpl_diagnostics.py::experiment_diversity; fig_floor.py | runs/fourier_wco2.log; paper/figs/floor_provenance.json |
| ClimSim million-sample sweep / kernel budget ladder | campaign/climsim_seeded.py; campaign/make_scaling_fig.py | campaign/collected/box4/a3_climsim_train_*.json; campaign/collected/dgx/climsim_cap/ |
| PDE survey | hidata_benchmark.py; campaign/dgx_checks/hidata_seeded.py; retained single-run descriptions | campaign/collected/dgx/hidata_p400/* (three seeds each); runs/bench_{Helmholtz,NavierStokes}.json; runs/advection_linear.json |
| Centering paired reruns | campaign/run_centering_campaign.py; campaign/fold_centering_sensitivity.py; train_mlp_refine.py; stack routines | paper/table_centering_sensitivity.tex; paper/table_centering_uq.tex; docs/paired_sensitivity_campaign.md |
| Correction-label mismatch | campaign/correction_mismatch.py | paper/table_mismatch_sensitivity.tex; paired-sensitivity protocol |
| Expanded OCO grid | campaign/jpl_grid_sensitivity.py; campaign/run_paper1_followups*.py | paper/table_grid_*; paper/sensitivity_macros.tex; docs/paired_sensitivity_campaign.md |
| Reconstruction/quadrature/sensitivity verification | campaign/benchmark_metric_check.py; campaign/aggregate_sensitivity.py; campaign/collect_sensitivity.py; campaign/render_sensitivity.py | paper/supp_sensitivity.tex; table/macro outputs |
| Residual arrays and storage precision | campaign/export_residuals.py | runs/residuals_manifest.json |
| Kernel-stage computation and memory | campaign/dgx_checks/cost_check.py | paper/impl.tex; paper/supp_experiments.tex |
| First-experiment wall times | campaign/harvest_timing.py | runs/timing_harvest.json |
