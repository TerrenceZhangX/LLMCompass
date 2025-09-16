
sudo docker build -t llmcompass-backend .

sudo docker run --rm -it -w /app --name llmcompass llmcompass-backend /bin/bash