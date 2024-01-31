conda create --name finetune python=3.11 -y

conda activate finetune

conda install pytorch==2.1.1 -c pytorch -c nvidia -y

pip install -U "huggingface_hub[cli]"

pip install -r requirements.txt


# sudo apt update

# sudo apt install nvidia-cuda-toolkit -y
# sudo apt install nvidia-utils-535
