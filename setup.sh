export IP_ADDRESS=""
export CORPUS_PATH=""
export KEY_PAIR_PATH=""

echo "$IP_ADDRESS"
echo "$CORPUS_PATH"
echo "$KEY_PAIR_PATH"

ssh -i "$KEY_PAIR_PATH" "ubuntu@${IP_ADDRESS}" -t 'mkdir setup; mkdir src; mkdir data'

scp -i "$KEY_PAIR_PATH" -r setup/* "ubuntu@${IP_ADDRESS}:~/setup"
scp -i "$KEY_PAIR_PATH" -r src/* "ubuntu@${IP_ADDRESS}:~/src"
scp -i "$KEY_PAIR_PATH" run.sh "ubuntu@${IP_ADDRESS}:~/run.sh"
scp -i "$KEY_PAIR_PATH" "$CORPUS_PATH" "ubuntu@${IP_ADDRESS}:~/data/rust-corpus.jsonl"

ssh -i "$KEY_PAIR_PATH" "ubuntu@${IP_ADDRESS}" -t 'cd setup; source conda.sh'

ssh -i "$KEY_PAIR_PATH" "ubuntu@${IP_ADDRESS}" -t 'bash run.sh'

