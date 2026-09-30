# Kerala Name Generator 

A character-level language model built from scratch in PyTorch to generate authentic Kerala (Malayali) names. Inspired by Andrej Karpathy's Makemore series.

## 📂 The Models
This repository explores two different architectures for name generation:
* **`name_generator_bigram.ipynb`**: A basic bigram model that uses just **1 previous character** to predict the next.
* **`name_generator_transformer.ipynb`**: A more advanced model that uses a context of **multiple previous characters** to predict the next.

## 🧠 Rebuilding PyTorch Under the Hood
To deeply understand the math and mechanics of deep learning, this project avoids high-level PyTorch modules. Instead, the core functionalities are implemented entirely from scratch:
* Custom `Linear` layers
* Custom `BatchNorm1d`
* Custom `Tanh` activations
* Manual backpropagation, gradient tracking, and weight updates.

## 📊 Dataset & Diagnostics
* **Dataset**: Trained on `processed_names.csv`, a custom curated list of traditional and modern Kerala names.
* **Diagnostics**: The notebooks include visualization blocks to monitor network health—plotting `tanh` output distributions (to check for saturation/dead neurons), layer-by-layer gradients, and update-to-data ratios.

## 🛠️ Usage
Clone the repository and run the notebooks sequentially. 
```bash
pip install torch pandas matplotlib jupyter
