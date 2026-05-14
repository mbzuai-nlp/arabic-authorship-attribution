

# Reporting the Quadratic Weighted Kappa
python calc_kappa_style_ranking.py style_ranking_all_annotations.tsv

# Reporting % Same Author in Top-3 (the percentage of candidate snippets written by the same author as the reference snippet that are assigned a ranking of 1-3)
python eval_performance_style_ranking.py style_ranking_all_annotations.tsv
