# Cellular Automata Phase Classification

_A three-level machine learning project to discover dynamical phases in cellular automata_

## What Is This Project?

This project uses unsupervised machine learning to automatically discover and classify the different types of long-term behavior (phases) in two-dimensional cellular automata. Think of it as teaching a computer to recognize when a system is dead, static, periodic, or chaotic — without ever telling it what these terms mean.

The work is structured in three levels, each asking a more ambitious question:

| Level | Question                                                                                     |
| ----- | -------------------------------------------------------------------------------------------- |
| **1** | Can ML discover different phases in Conway's Game of Life as we vary initial density?        |
| **2** | How do these phases respond to perturbations? (Are they stable or chaotic?)                  |
| **3** | Can we classify all 262,144 possible 2-state CA rules into Wolfram's four dynamical classes? |

## Why Does This Matter?

Cellular automata are simple models that can exhibit incredibly complex behavior — pattern formation, self-organization, chaos, and even computation. They serve as toy models for understanding real physical systems like:

- Phase transitions in materials
- Pattern formation in biological systems
- Self-organization in active matter
- The nature of chaos and complexity

Traditional approaches require scientists to manually identify and classify patterns. This project asks: **Can machine learning do this automatically, and might it discover things we've missed?**
