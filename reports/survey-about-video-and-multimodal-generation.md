# Survey about Video and Multimodal Generation

## TL;DR
- Video generation has evolved from early GAN and VAE frameworks to sophisticated Diffusion Transformers (DiTs), tracing a comprehensive trajectory across foundational architectures [1].
- Pretraining loss in diffusion transformers follows predictable power-law scaling relationships relative to compute investment [2], while hybrid models like Chimera combine linear attention and Mixture-of-Experts to achieve high computational efficiency [3].
- Advanced cross-modal frameworks such as BridgeDiT, Kling-Foley, and HunyuanVideo-Foley address modal interference, representation alignment, and synchronization in text-to-sounding-video generation [4][5][6].
- Maintaining long-horizon spatiotemporal consistency is tackled by specialized architectures like TECO and large-scale 3D VAE diffusion transformers such as CogVideoX [7][8].

## Background
Generative modeling is formally defined as the task of estimating and sampling from high-dimensional probability distributions, encompassing foundational families such as Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), normalizing flows, and diffusion models [9]. In the video domain, early generative approaches struggled with training instability or blurred temporal outputs, motivating the shift toward diffusion probabilistic models [1]. Modern video generators frequently leverage 3D Variational Autoencoders (VAEs) to compress video streams along spatial and temporal dimensions into efficient latent spaces [8]. CogVideoX exemplifies this paradigm by employing an expert transformer equipped with expert adaptive LayerNorm for deep fusion between text prompts and video latent representations [8].

## Architectural Evolution and Scaling Laws
The rapid scaling of video generation models is grounded in rigorous empirical scaling laws. Investigations into Diffusion Transformers (DiTs) demonstrate that pretraining loss follows a precise power-law relationship with compute budgets, providing predictable scaling trajectories [2]. To circumvent the quadratic computational complexity of dense spatiotemporal attention, recent architectures adopt hybrid designs. Chimera introduces a hybrid visual diffusion backbone combining Kimi Delta Attention (KDA) for linear-complexity long-context state tracking, interleaved Multi-head Latent Attention (MLA), and Sparse Mixture-of-Experts (MoE) layers [3]. Utilizing a principled "HeteroP" module-wise scaling recipe, Chimera successfully trains an 11B-parameter model with high computational efficiency relative to standard full-attention baselines [3]. Furthermore, comprehensive surveys trace the broader taxonomy of video diffusion models from early formulations to industry-scale implementations [10].

## Multimodal Integration and Cross-Modal Alignment
Multimodal generation extends beyond unimodal video synthesis to encompass joint text-to-audio-visual production, where maintaining synchronization across modalities is paramount. In text-to-sounding video (T2SV) tasks, modal interference and semantic misalignment often arise between visual dynamics and auditory cues [4]. The BridgeDiT framework addresses this by employing a Cross-Referential Rewriter (CRR) caption framework and Dual Cross-Attention (DCA) as a bidirectional bridge between video and audio streams [4]. Similarly, Kling-Foley integrates a visual semantic representation module, an audio-visual synchronization module, and a universal latent audio codec for frame-level alignment [5]. HunyuanVideo-Foley further advances this domain through a representation alignment (REPA) strategy guided by self-supervised audio features and a scalable 100k-hour multimodal data pipeline [6]. Beyond audio-visual synthesis, Large Language Models (LLMs) serve vital roles in multimodal generation and editing pipelines as evaluators, labelers, instruction processors, and semantic planners [11].

## Efficiency and Spatiotemporal Consistency
Inference efficiency remains a central challenge in video diffusion due to the compounded computational load of iterative denoising across spatiotemporal token grids [12]. Optimization paradigms are systematically categorized into step distillation, efficient attention mechanisms, model compression, and cache/trajectory optimization [12]. Concurrently, ensuring spatiotemporal consistency—encompassing spatial consistency (subject identity, scene layout, static semantics) and temporal consistency (motion smoothness, flicker suppression, dynamic semantics)—is crucial for high-fidelity outputs [13]. The Temporally Consistent Transformer (TECO) addresses long-horizon temporal consistency by compressing input sequences into fewer embeddings, applying a temporal transformer, and expanding back via a spatial MaskGit prior [7].

## Theoretical Frameworks and Evaluation Strategies
The conceptualization of generation has evolved beyond density estimation toward a unified task-first framework connecting prediction, compression, and decision-making across diverse generative families [9]. In parallel, evaluation methodologies have transitioned from traditional pixel-space or distribution-based metrics (such as FID and FVD) toward holistic, perception-aligned evaluation suites [1]. The integration of LLMs as automated evaluators and semantic labelers further standardizes the assessment of generative quality and instruction alignment [11].

## Trends and open problems
Recent trends in video and multimodal generation emphasize the scaling of hybrid linear-attention transformers [3] and the deployment of MLLM-driven editing workflows [11]. Nevertheless, several open problems persist:
1. **Long-Horizon Drift**: While architectures like TECO demonstrate long-term consistency over hundreds of frames by preserving visual elements across camera movements, generating temporally stable videos over even more extended horizons remains an open challenge [7].
2. **Modal Competition**: Disentangling multimodal semantics in joint video-audio synthesis requires specialized conditioning pipelines to prevent modality dominance [4][6].
3. **Computational Bottlenecks**: Compounding token growth in high-resolution video diffusion necessitates ongoing advancements in step distillation and efficient attention kernels [12].
4. **Evaluation Disparities**: Bridging automated perceptual metrics with subjective human evaluation standards continues to complicate standardized cross-model comparisons [1][12].

## References
[1] Bridging Text and Video Generation: A Survey. hf-search. https://huggingface.co/papers/2510.04999 (2025-10-06)
[2] Scaling Laws for Diffusion Transformers. arxiv. https://arxiv.org/abs/2410.08184 (n.d.)
[3] Chimera: Designing and Chinchilla-Scaling Hybrid Visual Diffusion Transformers. arxiv. https://arxiv.org/abs/2607.28611 (2026-07-30)
[4] Taming Text-to-Sounding Video Generation via Advanced Modality Condition and Interaction. web. https://link.springer.com/chapter/10.1007/978-3-032-37574-2_8 (2026-09-13)
[5] Kling-Foley: Multimodal Diffusion Transformer for High-Quality Video-to-Audio Generation. hf-search. https://huggingface.co/papers/2506.19774 (2025-06-24)
[6] HunyuanVideo-Foley: Multimodal Diffusion with Representation Alignment for High-Fidelity Foley Audio Generation. hf-search. https://huggingface.co/papers/2508.16930 (n.d.)
[7] Temporally Consistent Transformers for Video Generation. web. https://proceedings.mlr.press/v202/yan23b/yan23b.pdf (n.d.)
[8] CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. hf-search. https://huggingface.co/papers/2408.06072 (2024-08-12)
[9] If generative AI is the answer, what is the question?. arxiv. https://arxiv.org/abs/2509.06120 (2025-09-07)
[10] Survey of Video Diffusion Models: Foundations, Implementations, and Applications. arxiv. https://arxiv.org/abs/2504.16081 (n.d.)
[11] LLMs Meet Multimodal Generation and Editing: A Survey. web. https://arxiv.org/abs/2405.19334 (n.d.)
[12] Efficient Video Diffusion Models: Advancements and Challenges. web. https://arxiv.org/abs/2604.15911 (n.d.)
[13] A Survey: Spatiotemporal Consistency in Video Generation. web. https://arxiv.org/abs/2502.17863 (n.d.)
