FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime
RUN apt-get update && apt-get install -y git build-essential
RUN conda install -c conda-forge nvidia-apex
RUN git clone https://github.com/nishadi/ditto.git
WORKDIR ditto
RUN sed -i '8d' requirements.txt
RUN pip install -r requirements.txt
RUN pip install typer --upgrade
RUN python -m spacy download en_core_web_lg
RUN pip install tensorboardX nltk
RUN python -m nltk.downloader stopwords
WORKDIR /home/remote/u6852937/projects


# FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

# # Install dependencies
# RUN apt-get update && apt-get install -y git build-essential

# # Install Nvidia Apex
# RUN conda install -c conda-forge nvidia-apex

# # Clone the Ditto repository and navigate to its directory
# RUN git clone https://github.com/nishadi/ditto.git /workspace/ditto
# WORKDIR /workspace/ditto

# # Remove problematic line from requirements.txt and install dependencies
# RUN sed -i '8d' requirements.txt
# RUN pip install -r requirements.txt
# RUN pip install typer --upgrade

# # Install spaCy language model and other Python dependencies
# RUN python -m spacy download en_core_web_lg
# RUN pip install tensorboardX nltk
# RUN python -m nltk.downloader stopwords

# # Copy the AutoER data and configurations to the container
# COPY ./ready_for_ditto_input/ /workspace/ditto/data/ready_for_ditto_input/
# COPY ./configs.json /workspace/ditto/

# # Expose the directory for any other tasks
# WORKDIR /home/remote/u6852937/projects

# # Run training for all D1 to D10 tasks
# WORKDIR /workspace/ditto

# RUN CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task AutoER/D1  --batch_size 16 --max_len 256 --lr 3e-5 --n_epochs 5 --lm roberta --fp16 --da del --dk product --summarize \
#     && CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task AutoER/D2  --batch_size 16 --max_len 256 --lr 3e-5 --n_epochs 5 --lm roberta --fp16 --da del --dk product --summarize \
#     && CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task AutoER/D3  --batch_size 16 --max_len 256 --lr 3e-5 --n_epochs 5 --lm roberta --fp16 --da del --dk product --summarize \
#     && CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task AutoER/D4  --batch_size 16 --max_len 256 --lr 3e-5 --n_epochs 5 --lm roberta --fp16 --da del --dk product --summarize \
#     && CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task AutoER/D5  --batch_size 16 --max_len 256 --lr 3e-5 --n_epochs 5 --lm roberta --fp16 --da del --dk product --summarize \
#     && CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task AutoER/D6  --batch_size 16 --max_len 256 --lr 3e-5 --n_epochs 5 --lm roberta --fp16 --da del --dk product --summarize \
#     && CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task AutoER/D7  --batch_size 16 --max_len 256 --lr 3e-5 --n_epochs 5 --lm roberta --fp16 --da del --dk product --summarize \
#     && CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task AutoER/D8  --batch_size 16 --max_len 256 --lr 3e-5 --n_epochs 5 --lm roberta --fp16 --da del --dk product --summarize \
#     && CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task AutoER/D9  --batch_size 16 --max_len 256 --lr 3e-5 --n_epochs 5 --lm roberta --fp16 --da del --dk product --summarize \
#     && CUDA_VISIBLE_DEVICES=0 python train_ditto.py --task AutoER/D10  --batch_size 16 --max_len 256 --lr 3e-5 --n_epochs 5 --lm roberta --fp16 --da del --dk product --summarize
