# Hardware Hacking and Reverse Engineering

## Project Overview
This repository contains the course project for *Hardware Hacking and Reverse Engineering*. The project focuses on the design, implementation, and security evaluation of a **Crossover Ring Oscillator (CRO) based Physically Unclonable Function (PUF)** for cryptographic key generation in embedded and IoT systems.

The work analyses the CRO PUF from a **hardware hacking perspective**, with emphasis on statistical security properties and resistance to **machine-learning-based modeling attacks**.

---

## Project Objectives
- Design and implement a CRO PUF architecture for cryptographic key generation  
- Evaluate key security using **uniqueness** and **Shannon entropy** metrics  
- Analyse robustness against **machine-learning modeling attacks**  
- Demonstrate practical feasibility through **gate-level simulation and FPGA implementation**

---

## Repository Contents
- `/proteus_design/` – Gate-level CRO PUF schematic and simulation files  
- `/fpga_implementation/` – FPGA design files and implementation results  
- `/evaluation_codes/` – Python codes for:
  - Uniqueness analysis  
  - Shannon entropy calculation  
  - Machine-learning modeling attacks (SVM, Naive Bayes, Random Forest)  
- `/results/` – Processed outputs, tables, and figures used in the report  
- `/report/` – Project report (PDF / LaTeX source)

---

## Methodology Summary
The CRO PUF generates 128-bit cryptographic keys by concatenating responses from multiple CRO blocks. Security evaluation is performed using:
- **Statistical analysis** (uniqueness and entropy)
- **Machine-learning modeling attacks**, where classifiers are trained on a subset of challenge–response pairs to assess predictability

Low prediction accuracy close to random guessing indicates strong resistance to modeling-based reverse engineering.

---

## Tools and Technologies
- Proteus Design Suite (gate-level simulation)  
- FPGA platform (hardware realization)  
- Python 3 (security evaluation and machine-learning analysis)  
- Scikit-learn (SVM, Naive Bayes, Random Forest)

---

## Course Information
- **Course**: Hardware Hacking and Reverse Engineering  
- **Focus**: Secure hardware design, adversarial evaluation, and reverse engineering awareness

---

## Disclaimer
This repository is created for **academic and educational purposes only**. The work is intended to demonstrate hardware security evaluation techniques and should not be used for malicious activities.

---

## Author
**Arafat Miah**

