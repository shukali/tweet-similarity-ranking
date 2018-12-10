# Tweet Similarity Ranking

This is an implementation of a **basic ranked Information Retrieval system**. The tool allows to **rank tweets from a given dataset** based on cosine-similarity and TF-IDF weights.
You can search for a given tweet and you will be shown the top n relevant tweets, according to the query. Ranked retrieval based on cosine-similarity and TF-IDF weights is a popular alogrithm for IR. Please note: this code is made only for the given dataset.

## Prerequisites

You need to have installed: **[numpy](http://www.numpy.org/)**, **[pandas](https://pandas.pydata.org/)** and **[scikit-learn](https://scikit-learn.org/stable/)**. The code was tested with Python 3.6.6. A dataset is already provided.

## How to start it?

Just start the script `python tweetranking.py`. After a few seconds, the top 10 matching tweets for a predefined tweet will be shown. To show the most similar results for a different tweet, open the dataset `data/tweets.csv`, pick a tweet of your choice, take it's ID and paste it in the console window. Alternatively, you can call the method `printTopSimilarTweets(tweetID='xy')` with the new tweet's ID.

Note: the whole tweet dataset consists of 50.000 tweets, some of them duplicates. Due to limited resources, you are strongly advised to limit the number of tweets to some thousands. The parameter `n_tweets_to_read` in the beginning defines the number of tweets to be read, default value is 5000.

The algorithm ignores duplicate tweets, thats why the actual number of tweets used can be smaller than the number set with `n_tweets_to_read`.

## Implementation
The algorithm behind this similarity search is based on cosine similarity and a TF-IDF weighting of term-document pairs. The TF-IDF calculation itself is done explicitly, as well as the cosine similarity calculation. For the counting of terms in a document, the scikit-learn [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) is used.

## Authors
* **Marcus Rottschäfer** - [GitHub profile](https://github.com/shukali)

If you have any ideas or questions regarding the code, feel free to contact me.


## License
This project is licensed under the MIT License. See the [LICENSE.md](LICENSE) file for details.
