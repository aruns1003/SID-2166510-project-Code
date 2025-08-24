---

Generative Adversarial Networks (GANs) & CycleGAN Project

This repository contains experiments with **Generative Adversarial Networks (GANs)** and **CycleGANs** for image-to-image translation tasks, including datasets such as **cartoons** and both **paired** and **unpaired image sets**.

The project was developed as part of a **Postgraduate AI & Big Data program**.

---

📂 Repository Structure

🔹 1. `Basic GAN.ipynb

* Implements a **simple GAN** from scratch.
* Trains a generator and discriminator on a basic dataset to understand the fundamentals of GAN training.
* Focuses on the adversarial learning process.

---

🔹 2. `Cartoon Dataset.ipynb

* Prepares and explores a **cartoon image dataset**.
* Includes preprocessing (resizing, normalization, augmentations).
* Used as training data for GAN and CycleGAN experiments.

---

🔹 3. CycleGAN with Corresponding.ipynb

* Implements **CycleGAN with paired datasets**.
* Trains on images where input-output pairs are available (e.g., photo → cartoon).
* Evaluates the quality of translations when correspondence exists.

---

🔹 4. CycleGAN without Corresponding.ipynb

* Implements **CycleGAN on unpaired datasets**.
* Demonstrates the power of CycleGAN in learning mappings without direct image pairs.
* Example: translating between two domains (e.g., horses ↔ zebras, photos ↔ cartoons).

---

🔹 5. CycleGAN test1.ipynb` & `CycleGAN test2.ipynb

* Experimental notebooks for testing CycleGAN variations.
* Includes different architectures, hyperparameters, and dataset configurations.
* Used for fine-tuning results.

---

🔹 6. GAN test.ipynb

* Additional testing notebook for GAN experiments.
* Compares performance of trained models with variations in learning rate, optimizers, etc.

---

 ⚙ Requirements

* Python 3.8+
* Jupyter Notebook
* TensorFlow / PyTorch (depending on implementation in notebooks)
* NumPy
* Matplotlib
* Pillow
* OpenCV (for image preprocessing)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

▶️ How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/gan-cyclegan-project.git
   cd gan-cyclegan-project
   ```

2. Open Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

3. Run any of the `.ipynb` files to explore models and experiments.

---

📊 Datasets

* **Cartoon Dataset** – used for training cartoon-style GAN models.
* **Paired Dataset** – used in *CycleGAN with Corresponding*.
* **Unpaired Dataset** – used in *CycleGAN without Corresponding*.

*(Datasets are not included in the repo due to size. Please download separately and update paths in the notebooks.)*

---

🚀 Results

* Basic GAN successfully generates synthetic images from random noise.
* CycleGAN demonstrates realistic **photo ↔ cartoon** and **domain transfer** with both paired and unpaired datasets.
* Test notebooks show experimental improvements and hyperparameter tuning.

---

 📌 Future Work

* Extend to larger, more diverse datasets.
* Improve training stability using techniques like Wasserstein GAN.
* Deploy trained CycleGAN model as a web app for real-time image translation.

--
