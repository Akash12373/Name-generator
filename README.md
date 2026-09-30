# Kerala Name Generator (Character-Level Language Model)

This project is a character-level language model built from scratch in Python and PyTorch, specifically trained to generate **names of people from Kerala (Malayali names)**. Inspired by Andrej Karpathy's Makemore tutorials, this notebook implements a Multi-Layer Perceptron (MLP) to generate authentic-sounding Kerala names character by character. 

Instead of relying entirely on PyTorch's built-in neural network modules, this project features a custom implementation of core neural network layers to deeply understand the mechanics of deep learning under the hood.

## 🚀 Features

* **Kerala-Specific Dataset**: Trained on a curated list of traditional, contemporary, Syrian Christian, and Mappila names unique to Kerala.
* **Custom Neural Network API**: Implements classes for `Linear`, `BatchNorm1d`, and `Tanh` activation layers that closely mimic the official `torch.nn` API.
* **Configurable Context Length**: Can be adjusted to predict the next character based on a single character input (Bigram-style) or a 3-character rolling context.
* **Manual Backpropagation**: The training loop calculates the cross-entropy loss and updates weights manually using custom learning rate decay, visualizing the raw gradients.
* **Network Health Diagnostics**: Includes custom visualization blocks to monitor the health of the neural network during training:
  * `tanh` output activation distributions (to check for saturation).
  * Layer-by-layer gradient distributions.
  * Weight gradient to data standard deviation ratios.

## 📂 Project Structure

* `Word_creator_transformer.ipynb`: The main Jupyter Notebook containing the data preprocessing, model architecture, training loop, and diagnostic plots.
* `processed_names.csv`: The dataset containing the Malayali names used to train the model. *(Note: Ensure this is in the same directory as the notebook).*

## 🧠 How it Works

1. **Vocabulary Building**: The model reads the list of Kerala names, appending a special `.` token to denote the start and end of a word. It builds a String-to-Integer (`stoi`) and Integer-to-String (`itos`) mapping for the 27 possible characters (26 alphabet letters + 1 special token).
2. **Dataset Creation**: Words are broken down into inputs (`X`) consisting of a configurable number of previous characters, and targets (`Y`) representing the predicted next character.
3. **Architecture**: 
   * Embedding Layer
   * Hidden layers utilizing Custom `Linear` transformations followed by `BatchNorm1d` and `Tanh` activations.
   * Output layer mapping to the 27 character logits.
4. **Optimization**: Trained using Stochastic Gradient Descent (SGD) with learning rate decay.

## 📊 Diagnostics
The notebook actively monitors the "dead neurons" issue in Tanh layers and tracks the running mean/variance in the custom Batch Normalization layer, plotting the distributions to ensure gradients are flowing properly without vanishing or exploding.

## 🛠️ Usage

To run this notebook locally:
1. Ensure you have Python installed along with `torch`, `pandas`, and `matplotlib`.
2. Clone this repository.
3. Open the Jupyter Notebook and run the cells sequentially.

```bash
pip install torch pandas matplotlib jupyter
jupyter notebook
