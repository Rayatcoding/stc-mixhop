# IJDSA Major Revision Checklist

This checklist maps reviewer comments to concrete repository support and manuscript actions.

## Reviewer 1

1. **FEMA near chance / missing target**  
   - Code: `load_fema()` now requires a real claim/loss target and stops if the input is a policy-only file. `scripts/check_fema_policy_file.py` audits policy-only FEMA files.  
   - Manuscript action: follow Route B. Remove FEMA quantitative tables, ablations, and sensitivity plots from the main manuscript. Mention FEMA only as a limitation/future extension requiring a labeled claims or policy-claims joined table.

2. **Lack of temporal graph baselines**  
   - Code: `TemporalGCN-GRU`, `DySAT-lite`, `EvolveGCN-lite` in `models.py`.  
   - Manuscript action: either report these dependency-light temporal baselines or state clearly if official implementations are not used.

3. **Sender/receiver label conflation**  
   - Code: `--label-mode sender_only`.  
   - Manuscript action: add sender-only variant table and discuss role-based label noise.

4. **Terminology hazard/fraud/risk**  
   - Manuscript action: standardize to `fraud risk`, `suspicious-account risk`, or `risk score`.

5. **Figure 2 legibility**  
   - Code: revised horizontal-bar `plot_overall_figure()`.

6. **PaySim citation mismatch**  
   - Manuscript action: verify all PaySim citations point to the correct PaySim reference.

7. **K=2 optimal claim too strong**  
   - Manuscript action: phrase as `best among the tested values` and acknowledge limited sweep resolution.

## Reviewer 2

1. **Novelty unclear**  
   - Manuscript action: sharpen distinction from traffic forecasting and temporal graph models.

2. **Temporal consistency vague**  
   - Code: STC pretraining temporal positives and time-decay temperature are explicit in `train.py`.  
   - Manuscript action: formalize temporal-positive InfoNCE and short-horizon attention fusion.

3. **Contrastive learning hurts on PaySim**  
   - Code: ablation output includes `w/o contrastive learning`.  
   - Manuscript action: discuss conditional regularization / attribute-dominant smoothing.

4. **Porto graph construction details and FEMA removal**  
   - Code: `build_tabular_similarity_graph()` records Porto graph columns in metadata. FEMA policy-only data is audited but not used for quantitative claims.  
   - Manuscript action: describe Porto heuristic graph construction and state that FEMA quantitative analysis was removed because the available policy-only file lacks a supervised target.

5. **Sensitivity limited**  
   - Code: K, dk, and tau sweeps are supported.  
   - Manuscript action: add tau sensitivity and avoid overclaiming.

6. **High false positives / recall overemphasis**  
   - Code: precision, Fbeta, threshold, PR-AUC all reported.  
   - Manuscript action: add operating-cost discussion.

7. **Deployment metrics**  
   - Code: `--profile` adds inference time and memory columns.

8. **Boundary condition not quantified**  
   - Code: tabular and graph performance are reported side by side; average ranking available.  
   - Manuscript action: add attribute-dominance discussion using tabular-vs-graph gap.

9. **Two-stage training not justified**  
   - Code: supervised-only and self-supervised variants are directly comparable.  
   - Manuscript action: justify as a stability/regularization mechanism rather than guaranteed metric booster.

10. **Interpretability visualization**  
   - Code: embedding PCA and attention export hooks.

## Reviewer 3

1. **Reproducibility**  
   - Code: packaged CLI, README, requirements, config logging, graph metadata.

2. **Related work organization**  
   - Manuscript action: add structured related-work comparison table.

3. **Architecture transparency**  
   - Manuscript action: add module input/output dimension table using `feat_dim`, `emb_dim`, `K`, and `dk` from config.

4. **Baseline selection**  
   - Code: temporal baseline additions.

5. **Statistical tests and visualization**  
   - Code: paired tests, average ranking, loss curves, violin plots, embedding PCA.


## Selected fast-revision route

We use Route B for FEMA/NFIP: remove FEMA quantitative results from the main manuscript. This directly addresses Reviewer 1's concern that the FEMA results are effectively random and prevents unsupported claims based on a policy-only file with no verified claim/loss target.
