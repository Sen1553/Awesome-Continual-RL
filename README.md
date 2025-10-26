# 📢 Awesome-Continual-RL


This repository collects important papers in the field of **Continual Reinforcement Learning (CRL)** and **Lifelong Reinforcement Learning (LRL)**, aiming to provide a comprehensive and up-to-date resource for researchers. Continual and lifelong RL are valuable research directions that help agents learn adaptively and robustly in **non-stationary environments**. Contributions and suggestions for missing important works are always welcome.

- [📢 Awesome-Continual-RL](#-awesome-continual-rl)
  - [🚀 Attention](#-attention)
  - [📝 Survey](#-survey)
  - [📖 Research Papers](#-research-papers)
    - [ICML 2025](#icml-2025)
    - [ICLR 2025](#iclr-2025)
    - [ICRA 2025](#icra-2025)
    - [NeurIPS 2024](#neurips-2024)
    - [ICML 2023](#icml-2023)
    - [NeurIPS 2022](#neurips-2022)
    - [ICLR 2022](#iclr-2022)
    - [ICML 2018](#icml-2018)
  - [🌎 Benchmarks](#-benchmarks)
    - [NeurIPS 2021](#neurips-2021)
    - [CoRL 2019](#corl-2019)


## 🚀 Attention
We aim to keep this repository up-to-date with the latest research on CRL and LRL. We are continuously improving and expanding this repository to make it a more comprehensive and well-organized resource for the Continual and Lifelong RL community.

Feel free to contribute either with a PR or by opening an issue.



## 📝 Survey

- Zuffer A, Burke M, Harandi M. **Advancements and Challenges in Continual Reinforcement Learning: A Comprehensive Review**[J]. arXiv preprint arXiv:2506.21899, 2025.[[Paper]](https://arxiv.org/abs/2506.21899)

    `TL;DR: This paper serves as a comprehensive and systematic survey about CL and CRL. While it offers a solid and accessible entry point for newcomers to the field, the classification of CRL approaches is relatively limited and does not cover some of the most recent advancements.`  


- Pan C, Yang X, Li Y, et al. **A Survey of Continual Reinforcement Learning**[J]. arXiv preprint arXiv:2506.21872, 2025.[[Paper]](https://arxiv.org/abs/2506.21872)

    `TL;DR: This paper provides an up-to-date and comprehensive survey of CRL. It introduces a new taxonomy that categorizes methods into four types: policy-focused, experience-focused, dynamics-focused, and reward-focused, from the perspective of knowledge storage and transfer.`  


- Khetarpal K, Riemer M, Rish I, et al. **Towards continual reinforcement learning: A review and perspectives**[J]. Journal of Artificial Intelligence Research, 2022, 75: 1401-1476.[[Paper]](https://www.jair.org/index.php/jair/article/view/13673)

## 📖 Research Papers

### ICML 2025
- Liu Z, Fu G, Du C, et al. **Continual Reinforcement Learning by Planning with Online World Models** (Spotlight)[J]. arXiv preprint arXiv:2507.09177, 2025.[[Paper]](https://arxiv.org/abs/2507.09177)

- Sun Y, Fu H, Littman M, et al. **Knowledge Retention for Continual Model-Based Reinforcement Learning**[J]. arXiv preprint arXiv:2503.04256, 2025.[[Paper]](https://arxiv.org/abs/2503.04256)[[Code]](https://github.com/YixiangSun/drago)

- Tang H, Obando-Ceron J, Castro P S, et al. **Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn**[J]. arXiv preprint arXiv:2506.00592, 2025.[[Paper]](https://arxiv.org/abs/2506.00592)[[Code]](https://github.com/bluecontra/C-CHAIN)

- Mesbahi G, Panahi P M, Mastikhina O, et al. **Position: Lifetime tuning is incompatible with continual reinforcement learning**[C]//Forty-second International Conference on Machine Learning Position Paper Track.[[Paper]](https://openreview.net/forum?id=JMoWFkwnvv)


### ICLR 2025
- Ahn H, Hyeon J, Oh Y, et al. **Prevalence of negative transfer in continual reinforcement learning: Analyses and a simple baseline**[C]//The Thirteenth International Conference on Learning Representations. 2025.[[Paper]](https://openreview.net/forum?id=KAIqwkB3dT)[[Code]](https://github.com/hongjoon0805/Reset-Distill)

    `TL;DR: This paper demonstrates that negative transfer is prevalent in CRL. Reset & Distill (R&D) tackles this with a dual-learner approach: an online network resets its parameters for each new task to eliminate negative transfer and efficiently learn, while an offline network sequentially distills knowledge from the online actor and previous expert policies to prevent catastrophic forgetting.`  

### ICRA 2025
- Roy K, Dissanayakc A, Tidd B, et al. **M2distill: Multi-modal distillation for lifelong imitation learning**[C]//2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025: 1429-1435.[[Paper]](https://ieeexplore.ieee.org/abstract/document/11128857/)
  
    `TL;DR: This paper proposes a multi-modal distillation-based lifelong imitation learning approach, which imposes constraints on the alterations in latent representations and action distributions. Experimental results on the LIBERO benchmark show superior performance in retaining old skills while learning new tasks.`

### NeurIPS 2024
- Chung W, Cherif L, Meger D, et al. **Parseval regularization for continual reinforcement learning**[J]. Advances in Neural Information Processing Systems, 2024, 37: 127937-127967.[[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/e6df4efa20adf8ef9acb80e94072a429-Paper-Conference.pdf)[[Code]](https://github.com/wechu/parseval_reg)

    `TL;DR: This paper introduces Parseval regularization to mitigate plasticity loss in CRL. It adds a term to the objective function to maintain the orthogonality of weight matrices, applied to both the policy and value networks across all layers except the last, thereby preserving optimization properties and improving the agent’s ability to learn new tasks. `

### ICML 2023
- Sokar G, Agarwal R, Castro P S, et al. **The dormant neuron phenomenon in deep reinforcement learning**[C]//International Conference on Machine Learning. PMLR, 2023: 32145-32168.[[Paper]](https://proceedings.mlr.press/v202/sokar23a/sokar23a.pdf)[[Code]](https://github.com/timoklein/redo)

    `TL;DR: This paper attributes the reduced expressivity of reinforcement learning networks to the dormant neuron phenomenon, showing through extensive experiments that the proportion of inactive neurons increases during training. It introduces ReDo, a simple yet effective method that periodically reinitializes inactive neurons based on their activation statistics to restore network expressivity.`

### NeurIPS 2022
- Wolczyk M, Zając M, Pascanu R, et al. **Disentangling transfer in continual reinforcement learning**[J]. Advances in Neural Information Processing Systems, 2022, 35: 6304-6317.[[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2938ad0434a6506b125d8adaff084a4a-Abstract-Conference.html)
    
    `TL;DR: This paper empirically investigates how SAC components impact transfer in CRL, highlighting the critic's paramount role and providing key design recommendations. Based on experimental observations, it proposes ClonEx-SAC, leveraging behavioral cloning and exploration strategy to significantly boost performance and forward transfer on Continual World.`  

### ICLR 2022
- Lyle C, Rowland M, Dabney W. **Understanding and preventing capacity loss in reinforcement learning**[J]. arXiv preprint arXiv:2204.09560, 2022.[[Paper]](https://openreview.net/forum?id=ZkC8wKoLbQ7)[[Code]](https://github.com/timoklein/infer)

    `TL;DR: This paper investigates capacity loss in RL, where networks gradually lose the ability to fit new targets. Using feature rank to quantify representational capacity, the authors propose Initial Feature Regularization (InFeR), which constrains a subspace of features to stay close to initial values, preserving feature rank and mitigating capacity loss across diverse RL settings.`  

### ICML 2018
- Schwarz J, Czarnecki W, Luketina J, et al. **Progress & compress: A scalable framework for continual learning**[C]//International conference on machine learning. PMLR, 2018: 4528-4537.[[Paper]](https://proceedings.mlr.press/v80/schwarz18a.html?ref=https://githubhelp.com)

    `TL;DR: This paper introduces Progress & Compress (P&C), a scalable continual learning framework with online EWC for robust knowledge retention. An active column learns new tasks in the progress phase, while a knowledge base consolidates them via distillation in the compress phase. Experiments on handwritten alphabet classification, Atari, and 3D maze navigation validate the effectiveness of P&C.`

## 🌎 Benchmarks

### NeurIPS 2021
- Wołczyk M, Zając M, Pascanu R, et al. **Continual world: A robotic benchmark for continual reinforcement learning**[J]. Advances in Neural Information Processing Systems, 2021, 34: 28496-28510.[[Paper]](https://proceedings.neurips.cc/paper/2021/hash/ef8446f35513a8d6aa2308357a268a7e-Abstract.html)[[Code]](https://github.com/awarelab/continual_world)

    `TL;DR: This paper proposes Continual World, a new benchmark for CRL built on Meta-World. It empirically evaluates seven CL methods on Continual World and finds that existing CL methods effectively mitigate forgetting but largely fail to achieve positive forward transfer, reflecting an inefficient utilization of previously learned knowledge.`

### CoRL 2019
- Yu T, Quillen D, He Z, et al. **Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning**[C]//Conference on robot learning. PMLR, 2020: 1094-1100. [[Paper]](http://proceedings.mlr.press/v100/yu20a)[[Code]](https://github.com/Farama-Foundation/Metaworld)

     `TL;DR: This paper proposes Meta-World, a benchmark for meta-reinforcement learning and multi-task learning consisting of 50 distinct robotic manipulation tasks. It evaluates seven meta-RL and multi-task learning algorithms, including MTSAC, to assess generalization across diverse tasks. Meta-World has become a widely used benchmark in CRL research.`