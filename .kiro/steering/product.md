# CayleyPy Product Overview

CayleyPy is an AI-based Python library for working with extremely large graphs, specifically focusing on Cayley graphs and Schreier coset graphs. The library addresses the challenge of pathfinding on graphs with unprecedented scale (e.g., up to 10^74 elements for 5x5x5 Rubik's Cubes) where traditional methods are computationally impractical.

## Core Purpose

- **Large-scale graph pathfinding**: Solving planning problems on state-transition graphs using machine learning
- **Mathematical applications**: Supporting research in group theory, combinatorics, and computational mathematics
- **Puzzle solving**: Providing optimal or near-optimal solutions for complex puzzles like Rubik's Cubes
- **Research tool**: Enabling mathematical conjectures and insights into Cayley graph properties

## Key Features

- ML/RL methods for pathfinding using neural networks and beam search
- Efficient BFS implementations for smaller subgraphs
- Growth function and diameter estimation
- Support for various puzzle types (Rubik's Cubes, Skewb, Pyraminx, etc.)
- Pre-trained models for common graph types
- JAX support for GPU/TPU acceleration

## Target Users

- Researchers in mathematics, computer science, and computational group theory
- Machine learning practitioners working on planning problems
- Puzzle enthusiasts and competitive speedcubers
- Students learning about graph theory and group theory applications