# Decision Transformer (UMC-203 Project)

This project implements the Decision Transformer (DT) model for offline reinforcement learning. It supports both **Atari (via Minari datasets)** and **MuJoCo environments** with training, evaluation, and visualization capabilities.

## 📖 Overview

Decision Transformer treats trajectory modeling as a sequence modeling problem. Given a return-to-go and past actions/observations, it predicts the next action using a GPT-style transformer architecture.

Supported features:
- Training DT on Minari Atari datasets (Breakout, Pong, Qbert, Seaquest)
- Training DT on MuJoCo datasets (Hopper, Walker2d, HalfCheetah, Reacher)
- Evaluation and trajectory visualization (including gifs for MuJoCo)
- Flexible dataset handling (HDF5 for Atari, Pickle for MuJoCo)
## 🗂️ Project Structure

```plaintext
📂 Decision-Transformer
├── 📂 cloned_repo
│   ├── Cloned it from official decision transformer repository: https://github.com/kzl/decision-transformer/tree/master
├── 📂 modified
|   ├── atari
|   ├── gym
|
├── .gitignore
├── AI_ML_Project_ppt.pdf
├── AI_ML_report.pdf
├── README.md

```
