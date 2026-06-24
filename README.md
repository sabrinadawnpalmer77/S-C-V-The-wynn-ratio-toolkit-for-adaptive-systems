# wynn ratio toolkit

**S = C/V — The wynn ratio for adaptive systems**

Open-source Python package implementing the wynn ratio framework from Sabrina Dawn Palmer (SSRN Author ID 8168956, June 2026).

[![Paper](https://img.shields.io/badge/SSRN-8168956-blue)](https://ssrn.com/abstract=8168956)

## What it does

- Reproduces the exact phase transition shown in the paper (critical window S ≈ 0.8–1.1)
- Simulates two-agent coordination with tunable constraint (C) and variance (V)
- Implements Common-Knowledge Coordination Signals (CKCS) as engineered constraint
- Provides entropy-based variance estimation, Lee Constraint inference, and avalanche analysis
- Includes financial crisis validation module (reproduces the corrected 3/3 shield pattern)

## Quick Start

```bash
pip install numpy scipy matplotlib pandas
git clone https://github.com/sabrinadawnpalmer77/wynn-ratio-toolkit.git
cd wynn-ratio-toolkit
