
# Tox-NIO: Neural Input Optimization for Inverse Drug Design & Projected Gradient Descent 
![Status](https://img.shields.io/badge/Status-Research_Prototype-blue) ![Framework](https://img.shields.io/badge/PyTorch-2.0-red) ![Cheminformatics](https://img.shields.io/badge/RDKit-Chemistry-green)

> **⚠️ LEGAL NOTICE:** This project is the sole intellectual property of **Nandhu S Kumar**. Unauthorized copying, adaptation, or use is strictly prohibited and will be subject to legal action.

**Tox-NIO** is a computational framework that leverages **Neural Input Optimization (NIO)** to perform inverse molecular design. Instead of screening millions of molecules to find a safe candidate (traditional QSAR), Tox-NIO takes a known toxic scaffold and uses gradient-based optimization to mathematically deduce the blueprint of a safer analog, effectively curing the molecule of its toxicity *in silico*.

## Scientific Abstract

In traditional drug discovery, predictive models map **Chemical Space $\to$ Property Space** ($f(x) = y$). This project inverts that paradigm. We freeze the weights of a trained Residual Neural Network and perform gradient descent on the *input vector itself* to minimize a specific toxicity endpoint (NR-AhR).

The resulting optimized vector represents a theoretical blueprint—a set of physicochemical properties (LogP, TPSA, Molecular Weight, etc.) that would result in a non-toxic classification. We then map this theoretical blueprint back to discrete chemical space by querying the **ZINC-250k** database for the nearest synthesizable neighbor.

## The Blueprint Paradigm
**Why don't we just invent new atoms?**

One of the central challenges in continuous optimization of molecules is that chemical space is discrete (you cannot have 0.5 of a carbon atom), while neural networks operate in continuous space.

If we allow the AI to optimize freely, it might suggest a molecule with a Molecular Weight of 300.5 and a Ring Count of 2.3. While mathematically optimal for the loss function, such a molecule cannot exist physically. Therefore, Tox-NIO treats the AI's output not as a final structure, but as a **property blueprint**. We use this blueprint to search a "warehouse" of real molecules (ZINC-250k) to find the one that best matches the AI's ideal specifications.

## Datasets

### 1. Tox21 (Training Data)
The **Toxicology in the 21st Century (Tox21)** challenge dataset contains qualitative toxicity measurements for 8,000+ compounds on 12 different biological targets.
* **Target Selected:** `NR-AhR` (Nuclear Receptor Aryl hydrocarbon Receptor). Activation of this pathway is linked to dioxin-like toxicity and carcinogenesis.
* **Role:** Used to train the surrogate ResNet model to distinguish between toxic and non-toxic physicochemical profiles.

### 2. ZINC-250k (Search Space)
A curated subset of the ZINC database containing 250,000 commercially available, drug-like compounds.
* **Role:** Acts as the discrete manifold for our realism check. After the AI hallucinates an optimal set of features, we query this database to find the closest real-world compound.

## Methodology

### Step 1: Feature Engineering (RDKit)
We map SMILES strings to a high-dimensional feature space using 12 differentiable descriptors:
* **Electronic:** TPSA (Topological Polar Surface Area), MaxPartialCharge.
* **Steric/Structural:** Molecular Weight, FractionCSP3 (saturation), NumRotatableBonds.
* **Drug-Likeness:** QED, MolLogP (Lipophilicity).

### Step 2: Surrogate Model Training (ResNet)
We train a **Residual Neural Network (ResNet)** with skip connections.
* **Architecture:** The residual blocks allow gradients to flow unimpeded through the network. This is critical not just for training, but for the *inversion* step, where gradients must propagate all the way back to the input layer.
* **Performance:** The model achieves ~0.85 AUC on the Tox21 test set.

### Step 3: Neural Input Optimization (The Inverse Loop)
We select a toxic molecule $x_{orig}$ and initialize a latent vector $z$. We optimize $z$ to minimize a composite loss function:

$$\mathcal{L} = \mathcal{L}_{tox}(f(g(z))) + \lambda \cdot \mathcal{L}_{dist}(g(z), x_{orig})$$

Where:
* $\mathcal{L}_{tox}$ represents the predicted toxicity probability (goal: push to 0).
* $\mathcal{L}_{dist}$ is the Euclidean distance between the new and old features.
* $\lambda$ (Lambda) is a regularization coefficient that forces the AI to preserve the scaffold's identity (scaffold hopping prevention).

## Results Case Study

**Input Molecule:** `C=CCOc1c(Br)cc(Br)cc1Br`
* *Status:* Toxic (Prob: 0.54)
* *Scaffold:* Aromatic Ether

**AI Optimization:**
The NIO algorithm suggested reducing the aromatic ring density and increasing the $SP^3$ carbon fraction (3D complexity) to mitigate toxicity.

**ZINC Match:**
* **Candidate:** `CCN(Cc1ccc(I)cc1)C(F)(F)F`
* **Predicted Toxicity:** 0.07 (Safe)
* **Distance:** 1.08 (Euclidean)

The system successfully identified a commercially available analog that retains the core lipophilicity profile of the original but eliminates the AhR-mediated toxicity risk.

## Tech Stack & Requirements

* **Python 3.8+**
* **PyTorch:** For differentiable programming and autograd.
* **RDKit:** For cheminformatics and descriptor calculation.
* **Scikit-Learn:** For high-dimensional nearest neighbor search (k-NN).

```bash
pip install torch rdkit-pypi scikit-learn pandas numpy seaborn
