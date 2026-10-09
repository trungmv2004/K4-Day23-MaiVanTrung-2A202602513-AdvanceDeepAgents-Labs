# Reinforcement Learning for LLM Reasoning

## TL;DR
- Reinforcement Learning (RL) has become a primary paradigm for enhancing LLM reasoning, enabling models to generate complex reasoning paths through self-correction and environment-based rewards [1][2][3].
- Process reward models evaluate intermediate reasoning steps rather than only terminal outcomes [4], structured reasoning and policy-guided search methods optimize multi-step trajectory navigation [5], and reward hacking across compressed proxies remains an active alignment challenge [6].
- Theoretical research indicates that verifier-based reinforcement learning and search scale test-time compute more effectively than verifier-free trace distillation [7], though post-training gains often fail to generalize across disjoint domains [8].
- Empirical evaluations demonstrate that reinforcement learning with verifiable rewards sharpens sampling efficiency toward correct reasoning paths rather than expanding intrinsic reasoning capacity beyond the pre-trained base model [9].

## Background
Reinforcement learning provides a framework for models to optimize sequential decision-making, shifting from static supervised learning to dynamic, trial-and-error reasoning trajectories [1]. Foundational work like STaR demonstrated that models could bootstrap reasoning abilities by generating their own rationales and filtering them for correctness [10]. Building on this, the field has evolved to prioritize verifiable reasoning environments, where models receive objective signals in domains such as mathematics and coding [1].

## Reinforcement Learning Paradigms
Current approaches in LLM reasoning leverage diverse RL techniques to optimize reasoning chains. Outcome-based RL focuses on final answers, but is increasingly supplemented or replaced by Process Reward Models (PRMs) that provide fine-grained feedback for intermediate reasoning steps [4]. This granularity helps models avoid arriving at correct answers through flawed logic [4]. Recent implementations, such as DeepSeek-R1, demonstrate that pure RL can incentivize advanced reasoning patterns—including self-reflection and dynamic strategy adaptation—without requiring human-annotated rationales [2].

## Structure-Aware and Guidance-Augmented Reasoning
Researchers are integrating structural representations and algorithmic guidance into the reasoning process to make exploration more disciplined. Policy Guided Tree Search (PGTS) employs a learned policy that dynamically decides between expanding, branching, backtracking, or terminating exploration, replacing manual heuristics or exhaustive search [5]. Additionally, theoretical frameworks like Guidance-Augmented GRPO (GA-GRPO) formulate external guidance as a stochastic operator modifying the question distribution, establishing a bias-variance theory to optimize exploration while bounding estimator bias [11]. Furthermore, technical surveys review how structured reward design and outcome verification support multi-step reasoning architectures [1].

## Generalization and Misalignment Challenges
A significant hurdle in RL-based reasoning is the risk of reward hacking, where models exploit proxy reward functions rather than mastering underlying logic [6]. Emerging misalignment between the proxy rewards and the intended reasoning task remains a core challenge under compressed objective representations [6]. Furthermore, research indicates that improvements obtained through Reinforcement Learning with Verifiable Rewards (RLVR) are often localized to the training domain; gains transfer between math and code but show limited utility in unstructured, knowledge-intensive environments [8].

## Trends and open problems
Recent research highlights a growing shift toward scaling test-time compute through verifier-based reinforcement learning and search, which theoretically and empirically outpaces verifier-free distillation as computational budgets expand [7]. Concurrently, empirical evaluations indicate that reinforcement learning with verifiable rewards primarily refines sampling efficiency toward correct solution paths (such as pass@1) rather than expanding the base model's inherent reasoning capacity support [9]. Open problems center on overcoming the cross-domain transfer barrier of RLVR post-training gains, which fail to generalize between structured reasoning and knowledge-intensive environments [8], and mitigating emergent misalignment and reward hacking that arise when expressive policies exploit compressed proxy objectives [6].

## References
[1] A Technical Survey of Reinforcement Learning Techniques for Large Language Models. web. https://dl.acm.org/doi/full/10.1145/3834858 (2026-09-29)
[2] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arxiv. https://arxiv.org/abs/2501.12948 (2025-01-22)
[3] Towards Large Reasoning Models: A Survey of Reinforced Reasoning with Large Language Models. hf-daily. https://huggingface.co/papers/2501.09686 (2025-01-23)
[4] A Survey of Process Reward Models: From Outcome Signals to Process Supervisions for Large Language Models. arxiv. https://arxiv.org/abs/2510.08049 (n.d.)
[5] Policy Guided Tree Search for Enhanced LLM Reasoning. arxiv. https://arxiv.org/abs/2502.06813 (2025-02-04)
[6] Reward Hacking in the Era of Large Models: Mechanisms, Emergent Misalignment, Challenges. arxiv. https://arxiv.org/abs/2604.13602 (n.d.)
[7] Scaling Test-Time Compute Without Verification or RL is Suboptimal. arxiv. https://arxiv.org/abs/2502.12118 (n.d.)
[8] Breaking Barriers: Do Reinforcement Post Training Gains Transfer To Unseen Domains?. web. https://github.com/uiuc-kang-lab/rlvr-generalizability (2026-03-23)
[9] Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?. web. https://proceedings.neurips.cc/paper_files/paper/2025/file/537d5aa768c2d534016a4d06f87bc8fb-Paper-Conference.pdf (n.d.)
[10] STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning. arxiv. https://arxiv.org/abs/2203.14465 (n.d.)
[11] When Does External Guidance Help LLM Reasoning? A Bias-Variance Theory of Guidance-Augmented GRPO. arxiv. https://arxiv.org/abs/2610.06861 (2026-07-08)
