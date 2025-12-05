# Melanoma-Classification-DL4H
This repository is a minimalistic reproduction of the Conference Paper
%T A Study of Artifacts on Melanoma Classification under Diffusion-Based Perturbations
%A Qixuan Jin
%A Marzyeh Ghassemi
%B Proceedings of the sixth Conference on Health, Inference, and Learning
%C Proceedings of Machine Learning Research
%D 2025
%E Xuhai Orson Xu
%E Edward Choi
%E Pankhuri Singhal
%E Walter Gerych
%E Shengpu Tang
%E Monica Agrawal
%E Adarsh Subbaswamy
%E Elena Sizikova
%E Jessilyn Dunn
%E Roxana Daneshjou
%E Tasmie Sarker
%E Matthew McDermott
%E Irene Chen	
%F pmlr-v287-jin25b
%I PMLR
%P 844--861
%U https://proceedings.mlr.press/v287/jin25b.html
%V 287
%X In melanoma classification, deep learning models have been shown to rely on non-medical artifacts (e.g., surgical markings) rather than clinically relevant features (e.g., lesion asymmetry), compromising their generalizability. In this work, we investigate the impact of artifacts on melanoma classification under two settings: (1) input disruptions, such as bounding boxes and frequency-based filtering, which isolate artifacts by region or frequency, and (2) a novel diffusion-based perturbation method that selectively introduces isolated artifacts into images, generating controlled pairs for direct comparison. We systematically analyze artifact biases in three benchmark datasets: ISIC 2018, HAM10000, and PH2. Our findings reveal that whole-image training outperforms lesion-only or background-only approaches, low-frequency features are essential for melanoma prediction, and classifiers are more sensitive to perturbations for the artifacts of ink markings, rulers, and patches. These results emphasize the need for systematic artifact assessment and provide insights for improving the robustness of melanoma classification models.


Code was forked from: https://github.com/QixuanJin99/dermoscopic_artifacts/blob/main/dermatology_melanoma_classification.ipynb
