# Emojify Sentences

#### Predict cool emojis for the sentences you type.
![In-use Animation](https://github.com/soum-sr/EmojifySentences/blob/master/emojify_webapp.gif?raw=true "In-use Animation")

#
 
Download the <b>glove.6B.50d.txt</b> file from https://www.kaggle.com/watts2/glove6b50dtxt


 
Train the model before you run using:
```
python train.py
```
It will create a <b>emojify.h5</b> weights file. Once you have both files(glove.6B.50d.txt and emojify.h5),


Run it in the terminal:
```
python run.py
```
<b>OR</b>

Run the flask app:
```
python app.py
```

