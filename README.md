 🌍 Land Cover Classification (EuroSAT)

A deep learning pipeline for classifying **Sentinel-2 satellite images** into land cover types using the **EuroSAT** dataset.

---

## ⚙️ Setup
```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
🛰️ Dataset
Download EuroSAT → place in:

bash
Copy code
data/archive/EuroSAT/
🚀 Commands
1️⃣ Prepare Data
bash
Copy code
python landcover_pipeline_eurosat_full.py prepare_data \
  --data-dir data/archive/EuroSAT \
  --out-dir data/eurosat_split
2️⃣ Train Model
bash
Copy code
python landcover_pipeline_eurosat_full.py train \
  --data-root data/eurosat_split \
  --stats data/mean_std.json \
  --save-dir checkpoints/ \
  --epochs 10 --batch-size 8 --lr 0.0001
3️⃣ Evaluate
bash
Copy code
python landcover_pipeline_eurosat_full.py evaluate \
  --model checkpoints/model_best.pth \
  --val-dir data/eurosat_split/val
4️⃣ Inference
bash
Copy code
python landcover_pipeline_eurosat_full.py infer \
  --model checkpoints/model_best.pth \
  --image sample.png
🧠 Model
Backbone: ResNet-18

Loss: CrossEntropy

Optimizer: AdamW

Accuracy: ~92% (val)

📦 Output
Trained weights → checkpoints/model_best.pth

Split data → data/eurosat_split/

Report → reports/LandCover_Report.docx

Author: D. Karthiga
Tech: PyTorch • EuroSAT • Deep Learning

yaml
Copy code
