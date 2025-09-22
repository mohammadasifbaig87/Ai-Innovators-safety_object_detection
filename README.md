

# 🚀 HackWithHyderabad - Safety Object Detection (YOLOv8)

This repository contains our solution for the **HackWithHyderabad Hackathon – Duality AI’s Space Station Challenge: Safety Object Detection #2**.
We use **YOLOv8** for multi-class object detection, focused on identifying safety-critical objects in simulated space station environments.

---

## 📌 Project Structure

```
HACKATHON2_SCRIPTS/
│── ENV_SETUP/             # Virtual environment setup
│── predictions/           # Stores prediction outputs
│── runs/                  # YOLO training logs and checkpoints
│── tfenv/                 # TensorFlow environment files (if used)
│── classes.txt            # List of object classes
│── predict.py             # Script for inference
│── train.py               # Script for training
│── visualize.py           # Script for visualizing results
│── yolo_params.yaml       # YOLOv8 hyperparameter config
│── yolov8s.pt             # Pretrained YOLOv8 weights
│── .gitignore             # Ignored files for git
│── app.py                 # Streamlit app for predictions
│── README.md              # Project documentation
```

---

## ⚡ Problem Statement

Duality AI’s challenge:
Develop an **AI model for safety object detection** that can accurately classify multiple safety-critical objects in space station environments.

Key Deliverables:

* ✅ Trained object detection model
* ✅ Performance report with metrics
* ✅ Documentation & usage guide
* ⚡ (Bonus) Novel use-case proposal

---

## 🛠️ Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/HACKATHON2_SCRIPTS.git
   cd HACKATHON2_SCRIPTS
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Training

Run training with custom dataset and parameters:

```bash
python train.py --data yolo_params.yaml --epochs 50 --img 640 --batch 16 --weights yolov8s.pt
```

* `yolo_params.yaml` defines dataset paths and hyperparameters.
* Training logs and checkpoints are saved in `runs/`.

---

## 🔍 Inference (Predictions)

To run predictions on new images/videos using the command line:

```bash
python predict.py --weights runs/train/exp/weights/best.pt --source data/test/images
```

* Results are saved inside `predictions/`.

---

## 🎨 Visualization

To visualize bounding boxes and detection results:

```bash
python visualize.py --input predictions/ --show
```

---

## 🌐 Streamlit App for Predictions

To run the Streamlit app for making predictions through a user-friendly interface:

1. Run the app with:

   ```bash
   streamlit run app.py
   ```

2. The app will open in your browser, allowing you to upload images or videos for object detection using the trained model.

---

## 📈 Evaluation Metrics

We evaluate the model using:

* **mAP (mean Average Precision)**
* **Precision / Recall**
* **F1-Score**
* **Inference Speed (FPS)**

Benchmark reference values are provided in the [Hackathon docs](https://falcon.duality.ai/secure/documentation/ex3-objdetection-multiclass?sidebarMode=learn&utm_source=hackathon&utm_medium=instructions&utm_campaign=hyderabad).

---

## 📂 Deliverables

For final submission:

* ✅ Codebase (`.py`, `.yaml`, `.txt`)
* ✅ Trained weights (`best.pt`)
* ✅ Performance report (`report.pdf`)
* ✅ README.md (this file)
* ⚡ (Optional) Bonus use-case proposal

---

## ⚡ Bonus Use Case Proposal

Beyond hackathon evaluation, this object detection model can be adapted for:

* **Space station monitoring** (detecting missing safety gear)
* **Industrial safety compliance** (detecting helmets, vests, tools)
* **Disaster response robotics** (identifying hazards in debris)

---

## 🙌 Team & Acknowledgments

* Developed for **HackWithHyderabad 2025**
* Powered by [Duality AI](https://www.duality.ai/)
* Based on [YOLOv8](https://github.com/ultralytics/ultralytics)

---

## 📝 License

This project is for **hackathon & educational purposes only**.

