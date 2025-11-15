# 🤖 GenCELLAgent: GenCellAgent: Generalizable, Training-Free Cellular Image Segmentation via Large Language Model Agents

This repository provides a comprehensive guide and implementation for GenCELLAgent from scratch using Google's Gemini as the Large Language Model (LLM) of choice.


## 📚 Contents

GenCELLAgent is a **training-free, multi-agent large language model system** designed for **generalizable cellular image segmentation**. It orchestrates multiple vision and segmentation tools — such as Cellpose, µSAM, ERNet, MitoNet, LISA, and SegGPT — through a collaborative framework. The agent follows a structured **plan–execute–evaluate** loop, enhanced with memory and self-evolution mechanisms.

Unlike traditional models that require fine-tuning or dataset-specific retraining, GenCELLAgent dynamically routes tasks across specialized and generalist models. It intelligently adapts to various imaging modalities (phase-contrast, fluorescence, confocal, EM, and histology) and even to novel biological structures via **in-context learning (ICL)** and **text-guided prompt refinement**.

### ✨ Key Features
- **Tool-Orchestrated Segmentation:** Integrates domain-specific and generalist segmenters for optimal results.
- **Adaptive Planning:** Uses Gemini to select segmentation strategies based on image style, context, and prior success.
- **Iterative Refinement:** Employs multi-step feedback using evaluator models to improve segmentation quality.
- **Human-in-the-Loop (HITL):** Supports interactive corrections with point, polygon, or region editing.
- **Self-Evolving Memory:** Stores past results and configurations to improve future segmentation sessions.

### 📈 Performance
Across multiple benchmark datasets, including **LiveCell**, **TissueNet**, **PlantSeg**, **Lizard** and **CellMap organelle data**, GenCELLAgent achieves:
- +15.7% mean segmentation accuracy improvement over specialist models.
- +37.6% average IoU gain on ER and mitochondria datasets.
- Strong generalization to unseen organelles (e.g., Golgi) using iterative refinement and test-time scaling.

### 🧠 Architecture Overview
GenCELLAgent’s architecture includes three coordinated modules:
1. **Planner (LLM):** Analyzes task context and decides segmentation routes.
2. **Executor:** Runs selected tools (Cellpose, µSAM, etc.) and aggregates outputs.
3. **Evaluator:** Uses VLM-based quality scoring to iteratively refine predictions.

This design enables GenCELLAgent to perform segmentation tasks robustly without retraining, adapting to new domains with minimal supervision.
## 🚀 Getting Started


### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yuxi120407/GenCELLAgent.git
   cd GenCELLAgent
   ```

2. Set up a virtual environment:
   ```
    conda env create -f env.yaml -n myenvname
   ```

### API Key Configuration 

Set API Key via Environment Variable

```bash
export GOOGLE_API_KEY=<your-gemini-api-key>
```
