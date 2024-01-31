#

# mkdir setup

# scp -i keypairs/test-keypair.pem -r setup/* ubuntu@3.70.99.48:~/setup

#sudo sed -i "/#\$nrconf{restart} = 'i';/s/.*/\$nrconf{restart} = 'a';/" /etc/needrestart/needrestart.conf

cd setup

source dependencies.sh

conda activate finetune

source set_auth_token.sh
huggingface-cli login --token "$HUGGINGFACE_TOKEN"


cd ../src

python finetune.py
