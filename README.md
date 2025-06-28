# Graph Neural Network Generalization With Gaussian Mixture Model Based Augmentation (GRATIN)

**_[ICML 2025]_**  This repository contains the Pytorch implementation code of our paper accepted to the *Forty-second International Conference on Machine Learning (ICML 2025).*  
📄 **Read the paper on OpenReview → [Graph Neural Network Generalization With Gaussian Mixture Model Based Augmentation](https://openreview.net/forum?id=JCKkum1Qye)**

## 👥 Authors
Yassine Abbahaddou, Fragkiskos D. Malliaros  , Johannes F. Lutzeyer, Amine M. Aboussalah  , Michalis Vazirgiannis

---


## ✂️ Abstract
*Graph Neural Networks (GNNs) have shown great promise in tasks like node and graph classification, but they often struggle to generalize, particularly to unseen or out-of-distribution (OOD) data. These challenges are exacerbated when training data is limited in size or diversity. To address these issues, we introduce a theoretical framework using Rademacher complexity to compute a regret bound on the generalization error and then characterize the effect of data augmentation. This framework informs the design of GRATIN, an efficient graph data augmentation algorithm leveraging the capability of Gaussian Mixture Models (GMMs) to approximate any distribution. Our approach not only outperforms existing augmentation techniques in terms of generalization but also offers improved time complexity, making it highly suitable for real-world applications.*  

---
## 🖼️ Method Diagram
![GRATIN Method Overview](Asset/GRATIN_Diagram.png)

---
## 🚀 Run the Code
Follow these steps to run GRATIN augmentation method:

```bash
# 1. Clone the repo
git clone https://github.com/<your-org>/gratin.git
cd gratin

# 2. (Recommended) create a fresh virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Train GRATIN on the default dataset
python main.py --config configs/default.yaml

# 5. Evaluate a pretrained checkpoint
python main.py --mode eval --checkpoint checkpoints/best.ckpt

# 6. (Optional) run the unit tests
pytest
```
## Citing
If you find this work interesting or helpful for your research, please consider citing this paper and give your star

```bibtex
@inproceedings{
abbahaddou2025graph,
title={Graph Neural Network Generalization With Gaussian Mixture Model Based Augmentation},
author={Yassine ABBAHADDOU and Fragkiskos D. Malliaros and Johannes F. Lutzeyer and Amine M. Aboussalah and Michalis Vazirgiannis},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=JCKkum1Qye}
}
```

## ✉️ Contact

Have questions or want to collaborate? Feel free to reach out to **Yassine Abbahaddou** at <yassine.abbahaddou@polytechnique.edu> 
