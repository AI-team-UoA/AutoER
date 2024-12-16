CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task AutoER/D2  --batch_size 16 --max_len 256 --lr 3e-5 --n_epochs 5 --lm roberta --fp16 --da del --dk product --summarize


docker cp ./autoer/ 54b7408b1edd:/workspace/ditto/data/

docker cp ./configs.json 54b7408b1edd:/workspace/ditto   

sudo docker run -it --gpus all --entrypoint=/bin/bash ditto       

cd ../../../../workspace/ditto/


CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task AutoER/D1  --batch_size 16 --max_len 256 --lr 3e-5 --n_epochs 5 --lm roberta --fp16 --da del --dk product --summarize


docker cp ./configs.json acc70a93a256:/workspace/ditto     
docker cp ./ready_for_ditto_input/ acc70a93a256:/workspace/ditto/data/./ready_for_ditto_input/  
docker cp ./train_ditto.py acc70a93a256:/workspace/ditto
docker cp ./run_all_inside.sh 54d79d32d83d:/workspace/ditto


INSIDE DOCKER:
cd /workspace/ditto
mkdir logs
nohup ./run_all_inside.sh > nohup.out 2>&1 & 