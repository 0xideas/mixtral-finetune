To finetune Mistral-7B:

1. adapt the finetuning script finetune.py to the hyperparameters, huggingface user name, etc that are relevant to your project
2. create an AWS EC2 p2.8xlarge instance with the image ami-0fcbae1ba9e2e9a07 "Deep Learning Proprietary Nvidia Driver AMI GPU PyTorch 2.1.0 (Ubuntu 20.04) 20231226"
3. in the file setup.sh set env variables
    - IP_ADDRESS to your instance IP address-
    - CORPUS_PATH to the path on your file system to the corpus you want to finetune on (a jsonl file)
    - KEY_PAIR_PATH to the path to the key used to authenticate with the aws instance
4. create a file "set_auth_token.sh" in setup, which sets the env variable HUGGINGFACE_TOKEN to your huggingface token
5. run bash setup.sh
