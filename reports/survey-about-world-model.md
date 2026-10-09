# Modern World Models for Embodied AI and Autonomous Control: A Comprehensive Survey

## TL;DR
- Modern world models have transitioned from passive pixel-predictive simulators to active "World Action Models" (WAMs) that unify generative physical simulation with embodied control [1].
- Evaluating world models has shifted from surface-level perceptual metrics (e.g., FID, FVD) to rigorous decision-centric frameworks, such as the L0–L7 evaluation ladder and driving benchmarks like WorldLens, which reveal that high visual realism does not guarantee control utility or route completion [2][3].
- Unsupervised foundational interactive simulators like Genie [4] and exploration-driven frameworks like GenieRedux utilizing the AutoExplore Agent [5] demonstrate that rich interaction data and latent actions can be learned from unlabelled internet videos.
- Advanced representation learning approaches such as decoder-free next-embedding prediction (NE-Dreamer) [6] and Joint Embedding Predictive Architectures (RoboJEPA) [7] establish robust scaling laws and temporal predictive alignment in latent space.

## Background
World models serve as internal simulation engines that capture environmental dynamics, allowing artificial agents to perform forward and counterfactual rollouts for perception, planning, and decision-making [8]. Historically rooted in cognitive science frameworks (such as mental model theories) and early model-based reinforcement learning (MBRL) like Ha and Schmidhuber's neural world models, early systems relied heavily on reconstructing high-dimensional pixel inputs [1]. 

A central tension in the field distinguishes between two core paradigms: understanding the underlying causal mechanisms of the environment versus merely predicting future surface observations [9]. While generative models traditionally prioritized pixel-level future prediction, recent perspectives argue that predictive accuracy is often a poor proxy for control utility, giving rise to decision-aware model learning and latent representation paradigms [3].

## Architectural Foundations and Representation Learning
The architectural landscape of world models spans recurrent state-space models, transformers, diffusion models, and Joint Embedding Predictive Architectures (JEPAs) [8]. To mitigate the heavy computational overhead and visual artifact accumulation of pixel-level reconstruction, recent work has advanced representation-space learning [6]. For instance, NE-Dreamer introduces a decoder-free MBRL agent that leverages temporal transformers to predict next-step encoder embeddings directly from latent state sequences, utilizing redundancy-reduction alignment (Barlow Twins) to improve spatial and memory reasoning on control benchmarks [6]. Complementing this, RoboJEPA applies JEPAs across 12 robotic embodiments, establishing that latent imagination error follows a second-order power law in compute [7].

## Unsupervised Interaction and Foundation Simulators
Foundation model paradigms have unlocked unsupervised interaction learning. Genie demonstrated that an 11B parameter foundation world model could be trained on over 200,000 hours of unlabelled internet gaming videos without ground-truth action or text labels, utilizing a spatiotemporal video tokenizer, a MaskGIT-based dynamics model, and a latent action model that infers discrete actions unsupervised between frames [4]. To scale such interactive environments without expensive human demonstrations, exploration-driven frameworks like GenieRedux employ the AutoExplore Agent to collect interaction data driven entirely by model uncertainty [5].

## Decision-Centric Benchmarking and Evaluation
Evaluating world models has traditionally suffered from an objective mismatch where visual plausibility metrics fail to reflect downstream decision-making performance [3]. Addressing this, recent work proposes structured evaluation hierarchies, such as the L0–L7 world-model evaluation ladder ranging from basic visual plausibility to policy optimization utility [3]. This evaluation gap is further highlighted by decision-centric analyses showing that naive generative metrics fail to capture interventional reasoning and long-horizon rollout stability [3].

## Domain-Specific Simulation and Policy Evaluation
In specialized domains like autonomous driving, the WorldLens benchmark evaluates driving world models across 24 standardized dimensions spanning generation, reconstruction, action-following, downstream tasks, and human preference [2]. Findings across these benchmarks reveal that representative models struggle across axes—texture-rich models violate geometry, and under closed-loop control, top models complete fewer than 14% of navigation routes [2]. Similarly, in robotics, evaluation frameworks like WMBench analyze hundreds of thousands of simulated policy rollouts, showing that evaluator quality depends critically on long-horizon action-faithfulness rather than short-term visual realism [10].

## Trends and open problems
The two years leading up to 2026 have witnessed a convergence toward World Action Models (WAMs) that couple generative simulation with explicit action representations [1]. Despite these strides, several open challenges persist:
- **Sim-to-Real and Long-Horizon Consistency:** Maintaining physical and behavioral consistency over long rollouts remains difficult, as evidenced by low closed-loop route completion rates in complex autonomous driving environments [2] and policy evaluation gaps in robotic manipulation [10].
- **Objective Mismatch in Evaluation:** While frameworks like the L0–L7 ladder and decision-centric evaluation criteria provide rigorous guidance [3], establishing a universally standardized, domain-agnostic metric for control utility remains an open problem.
- **Cross-Embodiment Generalization:** Scaling world models across heterogeneous robotic morphologies (as explored in multi-embodiment JEPAs) while preserving zero-shot planning accuracy requires addressing severe distribution shifts [7].
- **Unifying Understanding and Prediction:** Reconciling the dual objectives of representing causal environmental mechanisms and predicting high-fidelity future trajectories within a single foundational architecture is an ongoing theoretical and empirical quest [8][9].

## References
[1] World Models and World Action Models (WAM): From Foundation Simulators to Embodied Action. web. https://jinxindeep.github.io/papers/WAM/WM_WAM.pdf (n.d.)
[2] Is Your Driving World Model an All-Around Player?. web. https://arxiv.org/html/2605.10858 (2026-05-11)
[3] How Should World Models Be Evaluated for Embodied Decision-Making? A Decision-Making-Centric Position. web. https://arxiv.org/html/2606.15032v2 (2026-06-28)
[4] Genie: Generative Interactive Environments. arxiv. https://arxiv.org/abs/2402.15391 (n.d.)
[5] Exploration-Driven Generative Interactive Environments. arxiv. https://arxiv.org/abs/2504.02515 (n.d.)
[6] Next Embedding Prediction Makes World Models Stronger. arxiv. https://arxiv.org/abs/2603.02765 (n.d.)
[7] RoboJEPA: Scaling Robotic Latent World Models. hf-daily. https://huggingface.co/papers/2610.10515 (2026-10-07)
[8] World Models: A Comprehensive Survey of Architectures, Methodologies, Reasoning Paradigms, and Applications. arxiv. https://arxiv.org/abs/2606.00133 (n.d.)
[9] Understanding World or Predicting Future? A Comprehensive Survey of World Models. web. https://dl.acm.org/doi/10.1145/3746449 (2025-09-09)
[10] GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation. web. https://arxiv.org/html/2607.02642v1 (n.d.)
