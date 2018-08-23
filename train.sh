srun -p GTX1080 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=lstm \
python main.py --epochs 10000 --optim adam --lr 0.05 | tee logs/lstm_rbl_v2.log
