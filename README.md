# Grad-FEC
Unequal Loss Protection of Deep Features Simulation (Grad-FEC)

This project implements a modified Grad-CAM and trains a proxy model to mimic its behavior. The primary goal is to simulate various loss patterns and detect the importance of features. By identifying these crucial features, the project can assign Forward Error Correction (FEC) packets to protect them during Cloud Edge based transmission and computing. Additionally, various error concealment estimators are investigated to assign FEC more efficiently.

## Contents
- [Overview](#overview)
- [Publications](#publications)

## Overview
The following figure gives a system overview of Collaborative Intelligence strategies implemented in Grad-FEC.

<img src="https://github.com/krcnynk/UnequalLossProtectionDeepFeatures_CI/blob/main/overviewPipeline.png" width="800" height="400">

The following figure gives an example of proxy model output.

<img src="https://github.com/krcnynk/UnequalLossProtectionDeepFeatures_CI/blob/main/heatmap.jpg" width="400" height="400">

## Publications
One peer reviewed conference papers were published on work done with Grad-FEC.
* Korcan Uyanik, S. Faegheh Yeganli, Ivan V. Bajić, [**Grad-FEC: Unequal Loss Protection of Deep Features in Collaborative Intelligence**](https://arxiv.org/abs/2307.01846), IEEE ICIP 2023.
  
## References

### [1] Selvaraju, Ramprasaath R. and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," published at International Journal of Computer Vision. [[pdf](https://arxiv.org/abs/1610.02391)]  

### [2] Ashiv Dhondea, Robert A. Cohen, Ivan V. Bajić, **CALTeC: Content-Adaptive Linear Tensor Completion for Collaborative Intelligence**. [[pdf](https://arxiv.org/abs/2106.05531)]

### [3] A. Dhondea, R. A. Cohen, and I.V.Bajić, **DFTS2: Deep feature transmission simulator for collaborative intelligence**. [[repo](https://github.com/ashivdhondea/dfts2)]

## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/krcnynk/UnequalLossProtectionDeepFeatures_CI/blob/master/LICENSE) file for details.
