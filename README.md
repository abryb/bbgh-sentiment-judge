### Usage

Single command usage:
```shell script
docker-compose up --build
```

### Process

##### 1. First step. Download pretrained word2vec and training it on articles and comments texts

```shell script
python -m trainer train_word2vec
```

This can take up to 3 hours. 

This operation creates 1 file in `Data` directory:
- `word2vector.trained_model.pickle` - Saved Word2Vec model.

##### 2. Download mentions.

```
python -m trainer download_mentions
```

This operations creates 2 cache files in `Data` directory:  
- `word2vector.dictionary.pickle` - a pickle file of dictionary for words which appear in mentions  
- `repository.mentions.pickle` - file with all mentions

##### 3. Train model 

```shell script
python -m trainer train --maxlen=32 --epochs=44 --save
```

This command trains our keras model and saves it in `Models` directory. 

##### 4. Predict mentions sentiments
```shell script
python -m trainer predict
``` 

This command takes our model from step 3 and runs model.predict on all mentions 
without checked sentiment.

This operations creates 1 file:
- `repository.predictions.pickle` - file with all our guesses about mentions sentiments



##### 5. Publish predicted sentiments
```shell script
python -m trainer publish
```

This command published our predicted sentiments. 