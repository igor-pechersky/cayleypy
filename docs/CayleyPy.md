## CayleyPy Project: ML Pathfinding on Large Cayley Graphs

### 1. Project Overview and Purpose
The CayleyPy project proposes a novel machine learning-based approach for solving the pathfinding problem on **extremely large graphs**, primarily focusing on **Cayley graphs**. The initiative aims to create an open-source machine learning Python framework for analysing Cayley graphs and contributing to the emerging field of machine learning applications in theoretical sciences.

**Key purposes of the project include:**
*   **Pathfinding on Large Graphs**: Addressing the challenge of finding paths on graphs with an unprecedented number of elements, such as up to 10^74 elements for 5x5x5 Rubik's Cubes, where standard methods are impractical.
*   **Optimal or Near-Optimal Solutions**: Delivering unprecedentedly short solution lengths, outperforming existing solvers, including those from large-scale challenges like the Kaggle Santa 2023.
*   **Efficiency**: Achieving significant computational efficiency in terms of solving speed and model training time compared to state-of-the-art competitors.
*   **Mathematical Advancements**: Applying AI methods to support mathematical conjectures and provide insights into properties of Cayley graphs, such as diameters and growth functions.

### 2. Core Principles: Graph Pathfinding as a Planning Problem
Solving puzzles like the Rubik's Cube or planning robot movements are specific instances of **planning problems**, which involve determining a sequence of actions to transition between two given states.
*   **Mathematical Framework**: Such problems are mathematically framed as **pathfinding on graphs**, specifically **state transition graphs**.
*   **Graph Representation**: In this framework, all possible states are represented as **nodes**, and **edges** correspond to **transitions between states based on actions (moves)**. The planning task then reduces to finding a path from a given initial node to one or more desired nodes.
*   **Cayley-type Graphs**: Rubik's Cubes and similar puzzles are represented by a specific class of highly symmetric state transition graphs known as **Cayley-type graphs** (or Schreier graphs for subgroups), where the puzzle's symmetry group can transform any node into another.
*   **Computational Challenges**: Finding the shortest paths on generic finite Cayley graphs is an **NP-hard problem**. Traditional methods like brute-force breadth-first search (BFS) or Dijkstra's, while effective for smaller graphs, require extremely large computational resources and are impractical for the "unprecedented sizes" of interest to CayleyPy (e.g., 4x4x4 and 5x5x5 Rubik's Cubes with 7.4 × 10^45 and 1.2 × 10^74 elements, respectively). Modern computer algebra systems like GAP also fail on sufficiently large groups, such as the 4x4x4 Rubik's Cube.

### 3. Machine Learning Approach: Algorithms and Components
CayleyPy's approach does not rely on prior human expertise about the graphs. It comprises two primary components: a **neural network model** and a **graph search algorithm**. The model is trained to guide moves towards the destination, and the search algorithm uses these predictions to find a path.

#### 3.1. Diffusion Distance Pathfinding (CayleyPy-1 Method)
This approach estimates "diffusion distance" – roughly, the length of a random path – which is computationally cheaper to estimate than true graph distance.
*   **Training Data Generation via Random Walks**:
    *   Starts from a **selected destination node** (e.g., the "solved state" for puzzles).
    *   Generates N random walk trajectories, each up to Kmax steps. For the Rubik's Cube, these correspond to random scrambling from the solved state, recording positions and the number of scrambles.
    *   For encountered nodes, stores pairs (v, k), where v is the node's feature vector (permutation description) and k is the number of steps required to reach it via the random walk. This conceptually measures "diffusion distance".
    *   **Random walks are computationally cheap** and can be generated directly during training, in contrast to the DAVI approach.
    *   **Non-backtracking random walks** are used to forbid moves to the previous state, making the number of steps more similar to the true distance and leading to faster mixing.
*   **Neural Network Training (ResMLP)**:
    *   The generated (v, k) pairs form the training set: v as input (feature vector) and k as the target (output to predict).
    *   The neural network's predictions estimate the diffusion distance from a given node to the selected destination node.
    *   A **multilayer perceptron (MLP) architecture with several residual blocks (ResMLP)** is utilised, a generalised form of MLPs used in prior works.
    *   Models are **pre-trained before the solving phase**.
    *   The training procedure uses the Adam optimizer with a fixed learning rate of 0.001 and mean squared error as the loss function.
    *   A new dataset of 1M examples is generated before each training epoch.

#### 3.2. Combined Diffusion Distance and Deep Q-Learning (CayleyPy-RL Method)
This novel method addresses the limitations of the pure diffusion distance approach (e.g., performance stagnation with increasing training data, non-monotonic relationship with true distance) by combining it with a modified Deep Q-learning (DQN) strategy. This approach aims to approximate the true distance more precisely while avoiding the **sparse reward problem** typical in standard RL for unweighted graphs.
*   **Rationale**:
    *   Diffusion distance is easy to estimate but less precise.
    *   DQN can approximate true distance but is computationally heavier.
    *   **Sparse reward problem**: In large unweighted graphs, rewards are rare (only at the destination node), making initial random exploration impractical.
*   **Proposed Method Steps**:
    1.  **Warm-up Diffusion Distance Training**: Preliminary training of the neural network using the diffusion distance approach. This provides meaningful initial targets, avoiding the "cold start" issue of random weights.
    2.  **Modified DQN Training**:
        *   Generate N **non-backtracking random walk trajectories** starting from the selected destination node.
        *   For the obtained nodes, compute **Bellman equation predictions**: t(g) = 1 + min_n:neighbors of g d(n) with t(e) = 0 (where e is the destination node).
        *   **Clip the predictions** by the number of steps of the random walk, as the true distance is always smaller than or equal to the number of steps; also clip negative values.
        *   Run **gradient descent** to minimize the loss between the clipped t(g) (target) and the neural network's predictions.
*   **Advantages of the Combined Method**: Eliminates the sparse reward problem, ensures meaningful targets from the outset, and effectively clips overestimated predictions. Experiments confirm that additional training by DQN improves the results. Pure DQN training alone does not perform well.

### 4. Graph Search Algorithm: Beam Search
**Beam search** is the chosen graph pathfinding technique, proving highly effective for CayleyPy. It compensates for potential inaccuracies in neural network predictions.
*   **Mechanism**:
    *   A positive integer W (known as "beam width" or "beam size") is fixed.
    *   Starting from a given node, the algorithm considers all its neighbours and computes neural network predictions for them.
    *   The W nodes with the **best (minimal) predictions** (closest to destination) are selected.
    *   This process is iteratively repeated: take neighbours of the selected W nodes, drop duplicates, recompute predictions, and select the top W nodes, until the destination is found or a step limit is exceeded.
*   **Efficiency**: CayleyPy developed an original and efficient PyTorch implementation to support beam sizes with millions of nodes. Increasing beam size linearly increases computation and memory, but is crucial for performance and increases the probability of finding a path while shortening paths.
*   **Local Minima Avoidance**: Beam search can bypass local minima where greedy search might get stuck, effectively acting as an "exchange of ideas" by selecting the top nodes across the entire beam neighborhood.
*   **Modifications**: CayleyPy implements a modified beam search using **hash functions to remove duplicates**, which reduces computational complexity.
*   **Prior Knowledge for LRX Graphs**: A curious finding for LRX Cayley graphs is that a single-line code modification (e.g., `if action == 'X' and current-state < current-state: continue`) in beam search dramatically improves performance, extending feasible pathfinding from n=30+ to n=100+. This condition, while specific to LRX, prevents suboptimal swaps. This prevents finding optimal paths and can lead to slightly longer paths for small `n`, but is beneficial for larger `n`.

#### 4.1. Multi-Agency
To enhance solution quality, CayleyPy employs a **multi-agent approach**.
*   **Diversity**: Due to the random nature of training data generation (random walks), each trained neural network ("agent") approximates distances differently, leading to diverse solution paths.
*   **Ensemble Selection**: To solve any given state, it is solved by **all agents**, and the **shortest solution path among all agents** is chosen as the final output.
*   **Impact**: A larger number of agents robustly provides more optimal pathfinding and increases the average solution rate. Even "worst agents" can contribute by providing the shortest solution for one or two scrambles, making the ensemble more efficient than single-model approaches.
*   **Scalability**: The approach is highly scalable and can run on distributed hardware using dozens of independent agents.

### 5. Parameter Influence on Performance
CayleyPy's solver performance is influenced by several key parameters:
*   **Number of Agents (A)**: Increasing the number of agents robustly improves the average solution length and optimality. For example, for 5x5x5 cubes, training 69 agents allowed beating all Kaggle Santa 2023 scrambles, though only 10 composed the final output.
*   **Beam Width (W)**: This is considered the **most important parameter**. Increasing W effectively reduces the average solution length, showing an approximately linear decrease with the logarithm of W.
*   **ResMLP Model Parameters (N1, N2, Nr, P)**:
    *   Larger and deeper networks generally provide shorter solutions.
    *   A higher number of layers (depth) is **more significant than a larger total number of parameters** (P).
    *   Even small models (e.g., 1M parameters) can achieve competitive average solution lengths comparable to larger state-of-the-art networks.
*   **Trainset Size (T)**: Surprisingly, increasing the trainset size beyond a certain point (e.g., 8 billion states for Rubik's Cube) has **limited impact** on pathfinder performance, leading to performance stagnation. This finding helped optimize computational resources by avoiding unnecessary training.

### 6. Performance and Achieved Results
CayleyPy demonstrates **superior performance** across multiple metrics and Rubik's Cube sizes, as well as general Cayley graph pathfinding.

**Rubik's Cubes**:
*   **Unprecedented Solutions**: First machine learning-based method to successfully solve **4x4x4 and 5x5x5 Rubik's Cubes**, with 7.4 × 10^45 and 1.2 × 10^74 elements, respectively.
*   **Solution Lengths**: Achieves **unprecedentedly short solution lengths** for 4x4x4 and 5x5x5 cubes, outperforming all available solvers, including the top results from the Kaggle Santa 2023 challenge. For example, the average solution length for 4x4x4 is 46.51, which is below the conjectural diameter of 48.
*   **Optimality**: For the 3x3x3 Rubik's Cube, CayleyPy achieves an optimality rate **exceeding 98%** (specifically 98.4% on the DeepCubeA dataset), matching task-specific solvers and significantly outperforming prior ML solutions like DeepCubeA (60.3%) and EfficientCube (69.6-69.8%). A single agent can achieve 90.4% optimality for the DeepCubeA dataset.
*   **Computational Efficiency**:
    *   **Solving Speed**: More than **26 times faster** in solving 3x3x3 Rubik's Cubes compared to EfficientCube (e.g., 10.91s vs 287.78s per scramble).
    *   **Training Time**: Requires up to **18.5 times less model training time** than the most efficient state-of-the-art competitor (e.g., 4h 40m vs 86h 25m for EfficientCube).

**General Cayley Graphs (LRX Generators of Symmetric Group Sn)**:
*   **Outperforms GAP**: CayleyPy's AI methods significantly **outperform classical computer algebra system GAP**, dealing with Sn up to n around 30 without prior knowledge, compared to GAP's limit of n around 20. With a small addition of prior knowledge (the "X-condition"), it can solve up to Sn with n around 90-100.
*   **Mathematical Contributions**: Supports the OEIS-A186783 conjecture for LRX Cayley graphs, which states that the diameter is n(n-1)/2. The conjecturally longest element follows a clear pattern as a product of transpositions. Rigorous lower bound n(n-1)/2 - n/2 - 1 and upper bound n(n-1)/2 + 3n for the diameter have been proven. The paper also presents an algorithm with an empirically better complexity of n(n-1)/2 + n/2, though not yet rigorously proven.
*   **Conjectures**: Observes numerical evidence suggesting the growth function of LRX graphs follows an asymmetric Gumbel distribution for large `n`. The spectrum of LRX graphs appears to show a surprisingly uniform distribution of eigenvalues.

### 7. Growth Computations and Specific Puzzle Analysis
The CayleyPy project also develops efficient algorithms and implementations for growth computations in Cayley and Schreier coset graphs, outperforming GAP by 100-1000 times in timing.

**Key findings regarding growth and specific puzzle analysis:**
*   **Diameter of Symmetric Group Sn**: Experiments explore maximal and minimal diameters for Cayley graphs of `Sn` with various generator sets. It is hypothesised that the maximum diameter is often achieved on sets of two generators.
*   **Pancake Graphs**: Growth functions and diameters are computed for Pancake graphs, including those with restricted prefix reversals (cubic Pancake graphs).
*   **Biologically Relevant Generators**: Cayley graph methods have direct applications in computational biology for estimating evolutionary distance, particularly for genome rearrangement events like inversions and translocations that form closed generating sets of a group. The reversal graph `Rn` has a diameter of `n-1`.
*   **Puzzle Groups**: CayleyPy has been used to compute growth functions and analyze properties of various puzzle groups:
    *   **Skewb**: God's number is 11 in the face-turn metric, and CayleyPy's computed growth function perfectly matches published results.
    *   **Pyraminx/Tetraminx Family**: Pyraminx has a God's number of 15, while Tetraminx has 11. CayleyPy computed layer counts for various sizes (regular, master, professor, royal, emperor).
    *   **Professor Tetraminx**: Growth function computed up to layer 6.
    *   **Megaminx**: Growth function computed up to layer 7 matches Tomas Rokicki's values. Lower bounds for God's number are 48 (HTM) and 59 (QTM), with an upper bound of 194 (HTM). The length of the smallest superflip sequence is believed to be close to the diameter.
    *   **Dino Cube**: Diameter is 13. Growth computations have been performed, showing states at distances from the solved state.
    *   **Dino + Little Chop**: A more complex variant with additional slice-turn operations, significantly expanding the generator set.
    *   **2x2x2 + Little Chop**: A hybrid puzzle with standard rotations augmented by shell moves, leading to a vastly expanded permutation space (approx. 4.39 × 10^29 states).
    *   **2x2x2 + Dino**: Growth data has been computed for this hybrid puzzle.
    *   **Curvy Copter**: Growth function computed and matched with Tomas Rokicki's data.
    *   **Megaminx + Chopasaurus**: An extended Megaminx with added slice-turn operations, with growth computed up to layer 5.
    *   **Radio Chop**: A complex vertex-turning and slice-based puzzle with intricate slice moves, for which generators are provided.
    *   **Christopher's Jewel**: Growth function computed and matched with Tomas Rokicki's data.
    *   **Master FTO**: Growth function computed up to layer 6.
    *   Other puzzles mentioned include Skewb Diamond, Big Chop, Chopasaurus, Pyraminx Crystal, Yottaminx, Zetaminx, Examinx, Petaminx, Teraminx, Gigaminx, Eitan's Star, Icosaminx, Redicosahedron (with and without centers), Regular Astrominx (+ Big Chop), Icosamate, Pentultimate, Master Pentultimate, Elite Pentultimate, Starminx Combo, Master Pyramorphix, and Starminx.

### 8. Applications and Future Directions
The method's scope is broad, applicable to various planning tasks that can be reformulated as graph pathfinding problems.
*   **Current and Future Applications**:
    *   **Mathematical Problems**: Estimating evolutionary distances in bioinformatics, processor interconnection networks, coding theory, cryptography, machine learning, and quantum computing.
    *   **Planning Tasks**: Robotics, game theory (e.g., chess, Go), inspired by works like AlphaGo/AlphaZero.
    *   **Specific Mathematical Areas**: Future exploration includes applications to mathematical, bioinformatic, and programming tasks.

### 9. Code and Data Availability
The CayleyPy project emphasizes open science with publicly available resources:
*   **Weights and Datasets**: Used in experimental studies are openly available on Zenodo.
*   **Hardest Scrambles Dataset**: A specific subset of 16 challenging scrambles from the DeepCubeA dataset that were not solved optimally by the approach is also available on Zenodo.
*   **Source Code**: The source code for experimental studies is available on GitHub.
*   **Kaggle Resources**: Notebooks related to CayleyPy project development are available on Kaggle, including public challenges to stimulate research and benchmarking. Kaggle infrastructure allows free execution on cloud servers and convenient benchmarking.
