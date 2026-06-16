# FEMA/NFIP data note for IJDSA revision

During revision, the inspected FEMA/NFIP sample contained policy metadata, coverage/premium fields, dates, geography, and identifiers, but did not contain a supervised claim-count or loss target such as `totalNumberOfClaims`, `numberOfClaims`, or `totalLossAmount`.

Implications:

1. A policy-only FEMA file cannot support supervised ROC-AUC, PR-AUC, recall, ablation, or sensitivity claims.
2. The revision code intentionally raises an error when no real claim/loss target is present.
3. If a separate labeled claims table or a policy-claims joined table is available, rerun FEMA as an exploratory boundary-case experiment.
4. If only the policy file is available, remove FEMA quantitative results from the main manuscript, or mention it only as an unlabeled-data limitation / future extension.
5. Do not fabricate, randomize, or infer supervised labels from non-target policy identifiers or premium/coverage fields.
