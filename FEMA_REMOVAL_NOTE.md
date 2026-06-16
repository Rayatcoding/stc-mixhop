# FEMA handling for the IJDSA revision

For the fast and safest major-revision strategy, FEMA/NFIP quantitative experiments are **removed from the main manuscript**.

Reason: the inspected FEMA/NFIP file is a policy-only table. It contains policy metadata, coverage, premium, dates, geographic fields, and identifiers, but it does not contain a genuine supervised claim/loss target such as `totalNumberOfClaims`, `numberOfClaims`, or `totalLossAmount`. Therefore, ROC-AUC, PR-AUC, ablation, and sensitivity results on this policy-only file are not valid supervised evidence.

Revision action:

1. Remove FEMA quantitative tables, ablations, and sensitivity plots from the main text.
2. Do not use FEMA to support claims about STC-MixHop effectiveness.
3. Mention FEMA only briefly as a limitation/future extension: labeled NFIP claims or policy-claims joined data would be required for a valid supervised evaluation.
4. Keep the reproducibility code conservative: if a user tries to run FEMA without a real claim/loss target, the loader raises an explicit error instead of fabricating random labels.
5. Retain PaySim as the primary chronological transaction-network experiment and Porto Seguro as the cross-domain attribute-dominant stress test.

Suggested manuscript wording:

> We removed the FEMA/NFIP quantitative experiment from the revised manuscript after rechecking the available public policy file. The inspected file contains policy, coverage, premium, date, and geographic attributes but no verified claim-count or loss target suitable for supervised fraud/risk detection. Reporting ROC-AUC or PR-AUC on such a policy-only file would therefore be misleading. We now treat FEMA/NFIP as a limitation and future extension requiring a labeled claims or policy-claims joined table.
