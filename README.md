### Process

##### 0. Build docker container
```shell script
docker-compose build
```

##### 1. First step. Download pretrained word2vec and train it on articles and comments texts

```shell script
docker compose run app python -m trainer prepare_word2vec
```
This operation creates 4 files in `Data` directory:
- `repository.articles.pickle` - File with downloaded articles
- `repository.comments.pickle` - File with downloaded comments
- `word2vector.model.pretrained.pickle` - Pretrained model downloaded from http://dsmodels.nlp.ipipan.waw.pl/.
- `word2vector.model.trained_on_articles_and_comments` - Model trained on articles and comments.

##### 2. Download mentions.

```shell script
docker compose run app  python -m trainer download_mentions
```

This operations creates 2 cache files in `Data` directory:  
- `repository.mentions.pickle` - file with downloaded mentions
- `word2vector.model.trained_on_mentions.pickle` - Saved Word2Vec model.
- `worker.dataset.pickle` - Mentions split to train, test and val sets.

##### 3. Train model 

```shell script
docker compose run app  python -m trainer train --maxlen=32 --epochs=22 --save
```

This command trains our keras model and saves it in `Models` directory. 

##### 4. Predict mentions sentiments
```shell script
docker compose run app  python -m trainer predict
``` 

This command takes our model from step 3 and runs model.predict on all mentions 
without checked sentiment.

This operations creates 1 file:
- `repository.predictions.pickle` - file with all our guesses about mentions sentiments


##### 5. Publish predicted sentiments
```shell script
docker compose run app  python -m trainer publish
```

This command published our predicted sentiments. 


##### 6. Run worker
```shell script
docker compose up
```