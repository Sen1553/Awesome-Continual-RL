# awesome-Continual-RL
This repository collects important papers in the field of **Continual Reinforcement Learning (CRL)** and **Lifelong Reinforcement Learning (LRL)**, aiming to provide a comprehensive and up-to-date resource for researchers. Continual and lifelong RL are valuable research directions that help agents learn adaptively and robustly in **non-stationary environments**. Contributions and suggestions for missing important works are always welcome.

## Attention
We aim to keep this repository up-to-date with the latest research on CRL and LRL. We are continuously improving and expanding this repository to make it a more comprehensive and well-organized resource for the Continual and Lifelong RL community.

Feel free to contribute either with a PR or by opening an issue.

## Research Papers

### Survey

- Zuffer A, Burke M, Harandi M. **Advancements and Challenges in Continual Reinforcement Learning: A Comprehensive Review**[J]. arXiv preprint arXiv:2506.21899, 2025.[[Paper]](https://arxiv.org/abs/2506.21899)

    `TL;DR: This paper serves as a comprehensive and systematic survey about CL and CRL. While it offers a solid and accessible entry point for newcomers to the field, the classification of CRL approaches is relatively limited and does not cover some of the most recent advancements.`  


- Pan C, Yang X, Li Y, et al. **A Survey of Continual Reinforcement Learning**[J]. arXiv preprint arXiv:2506.21872, 2025.[[Paper]](https://arxiv.org/abs/2506.21872)

    `TL;DR: This paper provides an up-to-date and comprehensive survey of CRL. It introduces a new taxonomy that categorizes methods into four types: policy-focused, experience-focused, dynamics-focused, and reward-focused, from the perspective of knowledge storage and transfer.`  


- Khetarpal K, Riemer M, Rish I, et al. **Towards continual reinforcement learning: A review and perspectives**[J]. Journal of Artificial Intelligence Research, 2022, 75: 1401-1476.[[Paper]](https://www.jair.org/index.php/jair/article/view/13673)


### ICLR 2025
- Ahn H, Hyeon J, Oh Y, et al. **Prevalence of negative transfer in continual reinforcement learning: Analyses and a simple baseline**[C]//The Thirteenth International Conference on Learning Representations. 2025.[[Paper]](https://openreview.net/forum?id=KAIqwkB3dT)[[Code]](https://github.com/hongjoon0805/Reset-Distill)

    `TL;DR: This paper demonstrates that negative transfer is prevalent in CRL. Reset & Distill (R&D) tackles this with a dual-learner approach: an online network resets its parameters for each new task to eliminate negative transfer and efficiently learn, while an offline network sequentially distills knowledge from the online actor and previous expert policies to prevent catastrophic forgetting.`  


### NeurIPS 2022
- Wolczyk M, Zając M, Pascanu R, et al. **Disentangling transfer in continual reinforcement learning**[J]. Advances in Neural Information Processing Systems, 2022, 35: 6304-6317.[[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2938ad0434a6506b125d8adaff084a4a-Abstract-Conference.html)
    
    `TL;DR: This paper empirically investigates how SAC components impact transfer in CRL, highlighting the critic's paramount role and providing key design recommendations. Based on experimental observations, it proposes ClonEx-SAC, leveraging behavioral cloning and exploration strategy to significantly boost performance and forward transfer on Continual World.`  


## Benchmarks

### NeurIPS 2021
- Wołczyk M, Zając M, Pascanu R, et al. **Continual world: A robotic benchmark for continual reinforcement learning**[J]. Advances in Neural Information Processing Systems, 2021, 34: 28496-28510.[[Paper]](https://proceedings.neurips.cc/paper/2021/hash/ef8446f35513a8d6aa2308357a268a7e-Abstract.html)[[Code]](https://github.com/awarelab/continual_world)

    `TL;DR: This paper proposes Continual World, a new benchmark for CRL built on Meta-World. It empirically evaluates seven CL methods on Continual World and finds that existing CL methods effectively mitigate forgetting but largely fail to achieve positive forward transfer, reflecting an inefficient utilization of previously learned knowledge.`