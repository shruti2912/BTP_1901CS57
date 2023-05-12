# BTP_1901CS57
Enhancing Social Media Event Detection with Knowledge Preservation using Heterogeneous GNNs: An Empirical Study
To run the model you can use different arguments.</br>

Commands:
1. cd /KPGNN
2. run main.py ( python3 main.py )
3. If you want to use pruning, set --prune=True
   (python3 main.py --prune=True)
4. If you want to use DGI model, set --use_dgi=True, default is GAT model <br>
   (python3 main.py --prune=True --use_dgi=True ) <br>
   (python3 main.py --prune=False --use_dgi=True )  etc.
5. For experimenting different dataset:<br>
    set --data_path=./data/, for full dataset with spacy embeddings<br>
    set --data_path=./data_filtered/, for filtered dataset with spacy embeddings<br>
    set --data_path=./data_lstm/, for full dataset with LSTM embeddings<br>
    set --data_path=./data_lstm_filtered/, for filtered dataset with LSTM embeddings<br>
    <br>
    (python3 main.py --prune=False --use_dgi=True --data_path=./data_filtered/ ) <br>
    (python3 main.py --prune=True --use_dgi=True --data_path=./data/ ) <br>
    (python3 main.py --prune=False --use_dgi=True --data_path=./data_lstm/ ) <br>
    (python3 main.py --prune=False --use_dgi=True --data_path=./data_lstm_filtered/ ) <br>
6. You can go through each possible arrangement with changing these parameters
7. We calculated the results with 30 epochs set --n_epochs=30

You can find the dataset folders here : "https://drive.google.com/drive/folders/15s68LxBU-3ZVnfZ5a2SVKhyGPj6WpNzo?usp=share_link"
