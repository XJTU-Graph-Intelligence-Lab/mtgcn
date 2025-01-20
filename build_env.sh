conda create -n MTGCN python==3.9

wget https://download.pytorch.org/whl/cu121/torch-2.1.0%2Bcu121-cp39-cp39-linux_x86_64.whl
pip install torch-2.1.0+cu121-cp39-cp39-linux_x86_64.whl

pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch_geometric==2.2.0

pip install einops
pip install pandas==2.0.3