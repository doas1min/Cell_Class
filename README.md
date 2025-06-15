# Cell-Image Detection & Classification Pipeline 
This repository provides a complete workflow for detecting individual cells in microscopy images and classifying them into three categories (N, L, M).  
The core script (`src/Cell_class.py`) relies on a pre-trained scikit-learn model and a feature scaler saved with **Joblib**.

# 3. Quick Start  
# 1) clone
git clone https://github.com/doas1min/Cell_Class.git

# 2) install dependencies
python -m venv venv       # optional
source venv/bin/activate  # optional
pip install -r requirements.txt

# 3) put images into ./target2
# 4) run
python src/Cell_class.py

