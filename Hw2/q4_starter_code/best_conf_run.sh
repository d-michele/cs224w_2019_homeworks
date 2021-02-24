### GraphSAGE ###
## Cora
# val accuracy: 0.79
# test accuracy: 0.777
python train.py --model_type GraphSage --hidden_dim 64 --dropout 0.1 --weight_decay 1e-5 --lr 0.001 --epochs 200  

### GAT ###
## Cora
# val accuracy: 0.802
# test accuracy: 0.814
python train.py --model_type GAT --hidden_dim 8 --dropout 0.6 --weight_decay 5e-4 --lr 0.001 --epochs 200
