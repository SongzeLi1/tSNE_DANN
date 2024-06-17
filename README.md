# DANN implementation reference from:  
https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA#citation

## Usage

1. During the training and inference stages, the selection of data paths, source and target domains needs to be replaced by corresponding paths and names on their own.
2. Training - Do not use DANN:  
`python train.py --config scripts/source-only.yaml --data_dir E:\\数据集\\office31 --src_domain webcam --tgt_domain amazon | tee DANN_W2A.log`
3. Training - Use DANN:  
`python train.py --config scripts/DANN.yaml --data_dir E:\\数据集\\office31 --src_domain webcam --tgt_domain amazon | tee DANN_W2A.log`
4. Infer, drawing tSNE visualization:  
`python infer_tSNE.py --config scripts/DANN.yaml --data_dir E:\\数据集\\office31 --src_domain webcam --tgt_domain amazon`


## References

Ganin, Yaroslav, and Victor Lempitsky. "Unsupervised domain adaptation by backpropagation." International conference on machine learning. PMLR, 2015.


