# Arabic Authorship Attribution and Style Transfer
This repo contains data and scripts to reproduce the results in our [paper](http://www.lrec-conf.org/proceedings/lrec2026/pdf/2026.lrec2026-1.576.pdf).

This repo is organized as follows:</br>
1. [A3D_corpus](A3D_corpus/): includes instructions for obtaining the A3D corpus.
2. [authorship_attribution](authorship_attribution/): includes the scripts needed to finetune and evaluate the AA models. It also includes the scripts needed for evaluating the TST output.
3. [style_ranking](style_ranking/): includes the data and scripts for the style ranking experiments.
4. [text_style_transfer](text_style_transfer/): includes the data and scripts for the text style transfer experiments.

## Citation:

If you find the code or data in this repo helpful, please cite our [paper](http://www.lrec-conf.org/proceedings/lrec2026/pdf/2026.lrec2026-1.576.pdf):


```bibtex
@inproceedings{hamed-etal-2026-AA-TST,
    title = "Benchmarking Arabic Authorship Attribution and Style Transfer with Large Language Models",
    author = "Hamed, Injy and
      Alhafni, Bashar  and
      Habash, Nizar  and
      Solorio, Thamar",
    booktitle = "Proceedings of the Language Resources and Evaluation Conference",
    year = "2026",
    pages = "7262--7278",
}
