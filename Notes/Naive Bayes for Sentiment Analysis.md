## Naive Bayes
Useful for
- Spam filtering
- Basic Sentiment Analysis
- Document Classification

### Sentiment Analysis
> Is the process of analyzing the emotional tone of a piece of text. Whether the text provided is positive, negative or neutral in tone.

**Gathering Data** - Can be done in a structured (with ratings equating to sentiment) or unstructured manner (plain text). Aggregation of data is possible in structured format.
Naive Bayes is typically used for processing and automatic labelling of unstructured data to improve performance on higher level models.

#### Stages
- **Data transformation** - Transforming categorical labels into numerical values
- **Data Cleanup** - Removing punctuation & special characters, converting to lowercase, removing all stop words
	words that do not assist in understanding sentiments "the, was is, are ...".
- **"Stem" words** - convert all forms of a word to its root: "running" -> "run"
- **Bag of words representation** - binary array of size `len(vocabulary)` of all words appearing in the example.

Then for all examples and all words in your vocabulary:
computer $p(word|+ve)$, $p(word|-ve)$ , $p(word|neutral)$ and 
