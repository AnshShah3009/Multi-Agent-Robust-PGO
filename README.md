# Multi-Agent Mapping with FinderNet and Graduated Non-Convexity

Multi-robot mapping is a crucial technique for efficiently covering large areas. However, finding the relative poses of robots is challenging due to noise and outliers in the data. Existing methods often use standard optimizers like LM or Gauss Newton with robust kernels such as TLS (Truncated Least Squares) for solving the pose graph problem. These methods, while effective, depend on a good initial guess and may not eliminate all outliers, potentially skewing the map. Additionally, they require sharing point clouds between robots, demanding high communication bandwidth and limiting mapping range. This poses a challenge to design a multi-agent loop closure system that handles noisy data and reduces communication bandwidth requirements.

## Approach

In this report, we present a novel approach leveraging FinderNet and Graduated Non-Convexity (GNC). FinderNet, a deep learning-based loop detection system, is combined with feature-based RANSAC and ICP for loop registration. GNC, the latest outlier rejection algorithm in GTSAM (Georgia Tech Smoothing and Mapping), enhances pose graph optimization.

### FinderNet

FinderNet excels in loop registration and detection by operating in the latent space, significantly reducing data transfer size to the central server. This efficiency is achieved by computing everything in a smaller latent embedding. Our evaluation on various datasets demonstrates that our proposed method produces superior maps and localizations while maintaining robustness to outliers.

### Graduated Non-Convexity (GNC)

GNC, integrated into the GTSAM framework, enhances outlier rejection during pose graph optimization. This contributes to the overall robustness and accuracy of our mapping system.

## Evaluation

We thoroughly tested our approach on multiple datasets, showcasing its performance against existing methods. The results highlight the capability of our method to generate high-quality maps and achieve accurate localization, even in the presence of outliers.

## Simulation on Husky

To validate the effectiveness of our approach, we conducted simulations using the Husky robot in an indoor office environment. The showcased results demonstrate the ability of our method to create precise maps while effectively handling outliers in loop closures.

## Conclusion

Our multi-agent mapping system, combining FinderNet and GNC, addresses the challenges posed by noisy data and communication bandwidth limitations. The comprehensive evaluation demonstrates the superiority of our approach in generating accurate maps and achieving robust localization. The simulation results further validate the practical applicability of our method in real-world scenarios.
