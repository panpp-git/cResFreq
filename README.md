# cResFreq
Codes for Complex-Valued Spectrum Estimation Network and Applications in Super-Resolution HRRPs Analysis with Wideband Radars

Refer to "G. Izacard, S. Mohan, and C. Fernandez-Granda, “Data-driven estimation of sinusoid frequencies,” arXiv:1906.00823. [Online]. Available:
https://arxiv.org/abs/1906.00823, 2019."

This repo. includes:
---Training Codes for the cResFreq, spcFreq and DeepFreq models.
1. complexTrain.py is the main function for cResFreq
2. freqDomainTrain.py for spcFreq
3. train.py for DeepFreq

---Experiments of Comparison in papers.
1. Fig.11(a) in the paper / rmse of frequency estimation of a single component is produced by "ACCURACY.py"
2. Fig.11(b) in the paper / FNR of multiple sinusoidals is produced by "FNR.py"
3. Fig. 10 in the paper / Resolution perfomance is produced by "RESOLUTION.py"
4. Fig. 8 in the paper / Visual verification of performance is produced by "VISUAL.py"
5. Fig. 12 in the paper / Detections of weak components is produced by "WEAK_DET2.py"
6. Fig. 6 in the paper / Weights characteristcs is produced by "WEIGHT.py"

--Requirements required for running the codes.
