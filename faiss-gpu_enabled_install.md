# Create fresh with all dependencies at once
conda create -n nlp_env python=3.10 -y
conda activate nlp_env

# Install everything in one command (resolves dependencies better)
conda install -c pytorch -c nvidia -c conda-forge \
  pytorch=2.1.0 torchvision=0.16.0 cudatoolkit=11.8 \
  faiss-gpu=1.7.4 mkl=2023.1.0 numpy=1.24.4 -y

# Install other dependencies
pip install -r requirements.txt