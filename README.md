# 📢 Awesome-Continual-RL


This repository collects important papers in the field of **Continual Reinforcement Learning (CRL)** and **Lifelong Reinforcement Learning (LRL)**, aiming to provide a comprehensive and up-to-date resource for researchers. Continual and lifelong RL are valuable research directions that help agents learn adaptively and robustly in **non-stationary environments**. Contributions and suggestions for missing important works are always welcome.

- [📢 Awesome-Continual-RL](#-awesome-continual-rl)
  - [🚀 Attention](#-attention)
  - [📝 Survey](#-survey)
  - [📖 Research Papers](#-research-papers)
    - [arXiv](#arxiv)
      - [2025](#2025)
    - [2017](#2017)
      - [2016](#2016)
    - [ICML 2025](#icml-2025)
    - [ICLR 2025](#iclr-2025)
    - [ICRA 2025](#icra-2025)
    - [Nature 2024](#nature-2024)
    - [NeurIPS 2024](#neurips-2024)
    - [ICML 2023](#icml-2023)
    - [ICLR 2023](#iclr-2023)
    - [Conference on Lifelong Learning Agents (CoLLAs) 2023](#conference-on-lifelong-learning-agents-collas-2023)
    - [NeurIPS 2022](#neurips-2022)
    - [ICLR 2022](#iclr-2022)
    - [NeurIPS 2019](#neurips-2019)
    - [ICML 2018](#icml-2018)
  - [🤖 Robot](#-robot)
    - [Nature Machine Intelligence 2025](#nature-machine-intelligence-2025)
    - [Entropy 2024](#entropy-2024)
  - [🌎 Benchmarks](#-benchmarks)
    - [NeurIPS 2023](#neurips-2023)
    - [CoLLAs 2022](#collas-2022)
    - [NeurIPS 2021](#neurips-2021)
    - [CoRL 2019](#corl-2019)
  - [🔗 Related Repositories](#-related-repositories)


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

### arXiv
#### 2025
- Elelimy E, Szepesvari D, White M, et al. **Rethinking the foundations for continual reinforcement learning**[J]. arXiv preprint arXiv:2504.08161, 2025.[[Paper]](https://arxiv.org/abs/2504.08161)

    `TL;DR: This paper argues that key assumptions of standard RL, such as MDP formulations, optimal policy focus, and episodic evaluation, are misaligned with the goals of CRL. It proposes a new paradigm tailored to CRL based on the history process, providing a more appropriate foundation for lifelong learning.`  

### 2017
- Fernando C, Banarse D, Blundell C, et al. **Pathnet: Evolution channels gradient descent in super neural networks**[J]. arXiv preprint arXiv:1701.08734, 2017.[[Paper]](https://arxiv.org/pdf/1701.08734)

    `TL;DR: PathNet can be viewed as an evolutionary-masked neural network that uses a genetic algorithm to select and train task-specific parameter paths, freezing them afterward to avoid catastrophic forgetting; by reusing high-fitness modules while isolating irrelevant ones, it achieves faster and more reliable transfer than fine-tuning or learning from scratch across both supervised and RL tasks.`  
#### 2016
- Rusu A A, Rabinowitz N C, Desjardins G, et al. Progressive neural networks[J]. arXiv preprint arXiv:1606.04671, 2016.[[Paper]](https://arxiv.org/abs/1606.04671)
  
    `TL;DR: This paper introduces Progressive Neural Networks, which add a new neural network (a column) for each task and freeze earlier ones to avoid catastrophic forgetting. Lateral connections enable effective feature reuse, and experiments on Atari, Pong variants, and 3D mazes show consistent positive transfer beyond standard finetuning in deep RL.`  



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

### Nature 2024
- Dohare S, Hernandez-Garcia J F, Lan Q, et al. **Loss of plasticity in deep continual learning**[J]. Nature, 2024, 632(8026): 768-774.[[Paper]](https://www.nature.com/articles/s41586-024-07711-7)[[Code]](https://github.com/shibhansh/loss-of-plasticity)

    `TL;DR: This paper systematically demonstrates that both supervised and reinforcement learning methods based on backpropagation suffer from loss of plasticity in continual learning tasks, and proposes Continual Backpropagation, which maintains long-term learning ability by continually and randomly reinitializing a small fraction of less-used units.`

### NeurIPS 2024
- Chung W, Cherif L, Meger D, et al. **Parseval regularization for continual reinforcement learning**[J]. Advances in Neural Information Processing Systems, 2024, 37: 127937-127967.[[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/file/e6df4efa20adf8ef9acb80e94072a429-Paper-Conference.pdf)[[Code]](https://github.com/wechu/parseval_reg)

    `TL;DR: This paper introduces Parseval regularization to mitigate plasticity loss in CRL. It adds a term to the objective function to maintain the orthogonality of weight matrices, applied to both the policy and value networks across all layers except the last, thereby preserving optimization properties and improving the agent’s ability to learn new tasks. `

### ICML 2023
- Sokar G, Agarwal R, Castro P S, et al. **The dormant neuron phenomenon in deep reinforcement learning**[C]//International Conference on Machine Learning. PMLR, 2023: 32145-32168.[[Paper]](https://proceedings.mlr.press/v202/sokar23a/sokar23a.pdf)[[Code]](https://github.com/timoklein/redo)

    `TL;DR: This paper attributes the reduced expressivity of reinforcement learning networks to the dormant neuron phenomenon, showing through extensive experiments that the proportion of inactive neurons increases during training. It introduces ReDo, a simple yet effective method that periodically reinitializes inactive neurons based on their activation statistics to restore network expressivity.`

### ICLR 2023
- Gaya J B, Doan T, Caccia L, et al. **Building a subspace of policies for scalable continual learning**[J]. arXiv preprint arXiv:2211.10445, 2022.[[Paper]](https://arxiv.org/abs/2211.10445)[[Code]](https://github.com/facebookresearch/salina/tree/main/salina_cl)

    `TL;DR: This paper introduces Continual Subspace of Policies (CSP), a scalable continual reinforcement learning method that incrementally builds a subspace of policies to balance performance and model size, achieving sublinear growth, no forgetting, and strong transfer on Brax and Continual World benchmarks.`

### Conference on Lifelong Learning Agents (CoLLAs) 2023
- Abbas Z, Zhao R, Modayil J, et al. **Loss of plasticity in continual deep reinforcement learning**[C]//Conference on lifelong learning agents. PMLR, 2023: 620-636.[[Paper]](https://proceedings.mlr.press/v232/abbas23a/abbas23a.pdf)[[Code]](https://github.com/timoklein/crelu-pytorch)

    `TL;DR: This paper investigates plasticity loss in CRL by tracking the evolution of weights, gradients, and activations. The authors observe activation collapse, where only a few neurons remain active in non-stationary environments, leading to vanishing gradients and stagnant network weights. Replacing ReLU with CReLU activations mitigates loss of plasticity and sustains learning over time.`


### NeurIPS 2022
- Wolczyk M, Zając M, Pascanu R, et al. **Disentangling transfer in continual reinforcement learning**[J]. Advances in Neural Information Processing Systems, 2022, 35: 6304-6317.[[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2938ad0434a6506b125d8adaff084a4a-Abstract-Conference.html)
    
    `TL;DR: This paper empirically investigates how SAC components impact transfer in CRL, highlighting the critic's paramount role and providing key design recommendations. Based on experimental observations, it proposes ClonEx-SAC, leveraging behavioral cloning and exploration strategy to significantly boost performance and forward transfer on Continual World.`  

- Schöpf P, Auddy S, Hollenstein J, et al. **Hypernetwork-ppo for continual reinforcement learning**[C]//Deep Reinforcement Learning Workshop NeurIPS 2022. 2022.[[Paper]](https://openreview.net/forum?id=s9wY71poI25)

    `TL;DR: This paper proposes HyperNetwork-PPO (HN-PPO), where a shared hypernetwork generates task-specific actor–critic parameters from task embeddings. Regularized hypernetwork updates prevent drastic changes to prior policies, enabling continual adaptation without forgetting.`

### ICLR 2022
- Lyle C, Rowland M, Dabney W. **Understanding and preventing capacity loss in reinforcement learning**[J]. arXiv preprint arXiv:2204.09560, 2022.[[Paper]](https://openreview.net/forum?id=ZkC8wKoLbQ7)[[Code]](https://github.com/timoklein/infer)

    `TL;DR: This paper investigates capacity loss in RL, where networks gradually lose the ability to fit new targets. Using feature rank to quantify representational capacity, the authors propose Initial Feature Regularization (InFeR), which constrains a subspace of features to stay close to initial values, preserving feature rank and mitigating capacity loss across diverse RL settings.`  

### NeurIPS 2019

- Traoré R, Caselles-Dupré H, Lesort T, et al. **DISCORL: Continual reinforcement learning via policy distillation**[C]//NeurIPS workshop on Deep Reinforcement Learning. 2019.[[Paper]](https://arxiv.org/abs/1907.05855)[[Code]](https://github.com/anonymous-authors-2018/DisCoRL)

    `TL;DR: This paper presents DisCoRL, a CRL framework that combines state representation learning and policy distillation to sequentially learn multiple tasks without catastrophic forgetting or task labels. The method distills previously learned policies into a single network using stored soft-labeled data, enabling a robot to automatically infer and perform all tasks in both simulation and real-world settings.`

- Rolnick D, Ahuja A, Schwarz J, et al. **Experience replay for continual learning**[J]. Advances in neural information processing systems, 2019, 32.[[Paper]](https://proceedings.neurips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html)

    `TL;DR: This paper introduces CLEAR, a simple replay-based method for CRL that mixes on-policy learning from new experience with off-policy learning and behavioral cloning from replay buffers. CLEAR greatly reduces catastrophic forgetting without requiring task labels or complex architectural changes, outperforming methods like EWC and P&C.`

### ICML 2018
- Schwarz J, Czarnecki W, Luketina J, et al. **Progress & compress: A scalable framework for continual learning**[C]//International conference on machine learning. PMLR, 2018: 4528-4537.[[Paper]](https://proceedings.mlr.press/v80/schwarz18a.html?ref=https://githubhelp.com)

    `TL;DR: This paper introduces Progress & Compress (P&C), a scalable continual learning framework with online EWC for robust knowledge retention. An active column learns new tasks in the progress phase, while a knowledge base consolidates them via distillation in the compress phase. Experiments on handwritten alphabet classification, Atari, and 3D maze navigation validate the effectiveness of P&C.`


## 🤖 Robot

### Nature Machine Intelligence 2025
- Meng Y, Bing Z, Yao X, et al. **Preserving and combining knowledge in robotic lifelong reinforcement learning**[J]. Nature Machine Intelligence, 2025: 1-14.[[Paper]](https://www.nature.com/articles/s42256-025-00983-2)[[Code]](https://github.com/Ghiara/LEGION)

    `TL;DR: This paper presents LEGION, a robotic lifelong reinforcement learning framework that continually preserves and recombines knowledge across tasks. Leveraging a Bayesian non-parametric knowledge space and language embeddings, it integrates an upstream task inference module with a downstream single policy conditioned on latent task representations, enabling efficient transfer and long-horizon task execution.`  

### Entropy 2024
- Gai S, Lyu S, **Zhang H, et al. Continual reinforcement learning for quadruped robot locomotion**[J]. Entropy, 2024, 26(1).[[Paper]](https://www.mdpi.com/1099-4300/26/1/93)

    `TL;DR: This paper introduces a CRL approach for quadruped robot to overcome catastrophic forgetting and plasticity loss. It uses a modified Piggyback dynamic network that protects learned parameters and enhances exploration via re-initialization and entropy regularization, outperforming baselines on complex locomotion tasks.`  
## 🌎 Benchmarks
### NeurIPS 2023
- Liu B, Zhu Y, Gao C, et al. **Libero: Benchmarking knowledge transfer for lifelong robot learning**[J]. Advances in Neural Information Processing Systems, 2023, 36: 44776-44791.[[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html)[[Code]](https://libero-project.github.io/)

    `TL;DR: This paper presents LIBERO, a new benchmark for Lifelong Learning in Decision-Making (LLDM) in robot manipulation, comprising 130 multi-modal tasks across four suites. It investigates five major research topics in lifelong learning, focusing on declarative and procedural knowledge transfer across visual, state, and language modalities.`  

### CoLLAs 2022
- Powers S, Xing E, Kolve E, et al. **Cora: Benchmarks, baselines, and metrics as a platform for continual reinforcement learning agents**[C]//Conference on Lifelong Learning Agents. PMLR, 2022: 705-743.[[Paper]](https://proceedings.mlr.press/v199/powers22b.html)[[Code]](https://github.com/AGI-Labs/continual_rl)

### NeurIPS 2021
- Wołczyk M, Zając M, Pascanu R, et al. **Continual world: A robotic benchmark for continual reinforcement learning**[J]. Advances in Neural Information Processing Systems, 2021, 34: 28496-28510.[[Paper]](https://proceedings.neurips.cc/paper/2021/hash/ef8446f35513a8d6aa2308357a268a7e-Abstract.html)[[Code]](https://github.com/awarelab/continual_world)

    `TL;DR: This paper proposes Continual World, a new benchmark for CRL built on Meta-World. It empirically evaluates seven CL methods on Continual World and finds that existing CL methods effectively mitigate forgetting but largely fail to achieve positive forward transfer, reflecting an inefficient utilization of previously learned knowledge.`

### CoRL 2019
- Yu T, Quillen D, He Z, et al. **Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning**[C]//Conference on robot learning. PMLR, 2020: 1094-1100. [[Paper]](http://proceedings.mlr.press/v100/yu20a)[[Code]](https://github.com/Farama-Foundation/Metaworld)

     `TL;DR: This paper proposes Meta-World, a benchmark for meta-reinforcement learning and multi-task learning consisting of 50 distinct robotic manipulation tasks. It evaluates seven meta-RL and multi-task learning algorithms, including MTSAC, to assess generalization across diverse tasks. Meta-World has become a widely used benchmark in CRL research.`



## 🔗 Related Repositories

| Repository | URL |
|:--------------------------------------|:-------------------------------------------------------------|
| **Papers-Of-Continual-RL** | [Link](https://github.com/datake/Papers-Of-Continual-RL) |
| **Awesome-Robotics-Manipulation** | [Link](https://github.com/BaiShuanghao/Awesome-Robotics-Manipulation/blob/main/contents/bottlenecks.md?tab=readme-ov-file#lifelongcontinualincremental-learning) |
| **continual-learning-papers** | [Link](https://github.com/ContinualAI/continual-learning-papers?tab=readme-ov-file#continual-reinforcement-learning) |



