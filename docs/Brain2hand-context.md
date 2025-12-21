
This is a High-Fidelity Project Context Document. It contains every specific detail regarding your architecture, constraints, team, and code style.
Save this as B2H_Project_Context.md. Whenever you open a new chat with an AI, upload this file or paste the text below.
Project Context: Team B2H (Huawei ICT Competition)
1. Project Identity & Team
Project Name: Brain2Hand: AI-Driven EEG Motor-Imagery Decoder for Hands-Free Control
Competition: Huawei ICT Competition 2025–2026 (Innovation Track)
University: Cambodia Academy of Digital Technology (CADT)
Team Name: B2H
Team Members:
Instructor: Nesta Hou
Students: Sopanha Lay (Captain), Chehnintchesda You
Core Objective: Develop a low-cost, cloud-powered Brain-Computer Interface (BCI) that translates EEG motor imagery (imagining hand movements) into control signals for prosthetics using Huawei technologies.
2. Critical Competition Constraints (The "Rules")
Mandatory Framework: MindSpore (Huawei’s Deep Learning Framework).
Strict Rule: Solutions using PyTorch/TensorFlow without MindSpore will be disqualified.
Current Phase: Preliminary Round (Deadline: Dec 21, 2025).
Hardware Status:
Now (Preliminary): Local Laptop (Pop!_OS Linux, CPU) or Cloud VM.
Future (Regional): Huawei Cloud ModelArts (Ascend 310/910 NPU) via CANN.
Evaluation Criteria:
Innovation (60%): Novelty of the algorithm (Transformer vs. CNN) and scenario.
Application Value (40%): Real-world impact on disability.
Completeness: A fully trained commercial model is not required for Preliminary, but a functional MindSpore prototype code is mandatory.
3. Technical Architecture & Stack
A. Environment
OS: Pop!_OS (Linux) x86_64.
Python: Version 3.9 (Managed via Conda b2h_mindspore env).
MindSpore Version: 2.7.1 (or latest stable).
Device Target: Currently CPU. Code must be written to be compatible with Ascend.
B. Data Pipeline (Preprocessing)
Library: mne (MNE-Python) for EEG signal handling.
Datasets:
PhysioNet EEG Motor Movement/Imagery Dataset: (Large scale).
BCI Competition IV-2a: (4-class motor imagery: Left, Right, Feet, Tongue).
Preprocessing Steps:
Loading: Read .gdf or .edf files using MNE.
Filtering: Bandpass Filter 8–30 Hz (Mu & Beta rhythms).
Epoching: Slicing data based on event markers (e.g., T=0 to T=4s).
Normalization: StandardScaler (sklearn) applied per subject.
C. Model Architecture (The Innovation)
Model Type: Spatial-Temporal Transformer (replacing traditional CNNs like EEGNet).
Why Transformer? To capture long-range temporal dependencies in brain signals that CNNs miss.
MindSpore Implementation Details:
Base Class: mindspore.nn.Cell.
Key Layers: mindspore.nn.TransformerEncoder, mindspore.nn.MultiHeadAttention.
Optimization: AdamWeightDecay optimizer.
Loss Function: SoftmaxCrossEntropyWithLogits.
4. Implementation Roadmap (Step-by-Step)
Step 1 (Done): Environment Setup (Conda + MindSpore 2.7.1).
Step 2 (Current): Data Ingestion Script. Write a Python script to load 1 subject from BCI IV-2a, preprocess it using MNE, and convert it to MindSpore Tensor.
Step 3: Model Definition. Define the Transformer class in MindSpore syntax.
Step 4: Training Loop. Create a minimal train.py that runs 1 epoch to prove the pipeline works (generate loss logs for screenshots).
Step 5: Deployment Simulation. Create a dummy script simulating "Inference" to show how the system would control a robotic arm.
5. Coding Standards for AI Assistant
Imports: Always use import mindspore and import mindspore.nn as nn. Do not import torch.
Context: Always verify context at the start of scripts:
code
Python
import mindspore.context as context
# Use CPU for local debugging, ready to switch to Ascend later
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
Data Types: Ensure input data is cast to mindspore.float32 before feeding the model.