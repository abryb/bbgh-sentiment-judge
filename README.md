
Posting new job to google cloud platform
```shell script
gcloud ai-platform jobs submit training bbgh_sentiment_judge_(date -u +%y%m%d_%H%M%S) \
  --package-path=$PWD/trainer \
  --module-name=trainer.task \
  --region=europe-west4 \
  --staging-bucket=gs://bbgh-sentiment-judge-bucket \
  --scale-tier=BASIC_GPU  \
  --python-version 3.7 \
  --runtime-version 2.2 \
  -- \
  --data_file gs://bbgh-sentiment-judge-bucket/Data/Dataset.csv \
  --embeddings_file gs://bbgh-sentiment-judge-bucket/Data/nkjp+wiki-forms-all-300-cbow-hs-50.txt \
  --output_dir gs://bbgh-sentiment-judge-bucket/Models


gcloud ai-platform jobs submit training bbgh_sentiment_judge_(date -u +%y%m%d_%H%M%S) \
  --package-path=$PWD/trainer \
  --config config.yaml
```


#### Development

If using PyCharm follow https://www.jetbrains.com/help/pycharm/using-docker-compose-as-a-remote-interpreter.html#tw 