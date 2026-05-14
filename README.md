# Cellular Automata Phase Classification

_A seven-level machine learning project to discover dynamical phases in cellular automata_

## What Is This Project?

This project uses unsupervised machine learning to automatically discover and classify the different types of long-term behavior (phases) in two-dimensional cellular automata. Think of it as teaching a computer to recognize when a system is dead, static, periodic, or chaotic — without ever telling it what these terms mean.

The work is structured in seven levels, each asking a more ambitious question:

| Level | Question                                                                                             |
| ----- | ---------------------------------------------------------------------------------------------------- |
| **1** | Can ML discover different phases in Conway's Game of Life as we vary initial density?                |
| **2** | How do these phases respond to perturbations (damage spreading / stability)?                         |
| **3** | Can we classify all 262,144 outer-totalistic rules into Wolfram-style dynamical classes?             |
| **4** | Do the discovered phases correspond to real phase transitions (finite-size scaling and criticality)? |
| **5** | What are the spatial and temporal scaling laws at large system sizes?                                |
| **6** | Can mean-field theory explain why the transition is first-order?                                     |
| **7** | Do these finite-size signatures generalize across multiple Life-like rules?                          |

## Why Does This Matter?

Cellular automata are simple models that can exhibit incredibly complex behavior — pattern formation, self-organization, chaos, and even computation. They serve as toy models for understanding real physical systems like:

- Phase transitions in materials
- Pattern formation in biological systems
- Self-organization in active matter
- The nature of chaos and complexity

Traditional approaches require scientists to manually identify and classify patterns. This project asks: **Can machine learning do this automatically, and might it discover things we've missed?**
