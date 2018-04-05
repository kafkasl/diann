# diann
Diann competition repo


Current notes:

Negated expressions do not coincide with negex ( 'not have' vs 'not' ). Will have to modify ngx input probably. 

Notes 1 Lluis meeting


* Keep parenthesis (may help identify acronyms) keep them from
* Upper / Lower (same as before), add it as feature,
* Embeddings to use (relate to parenthesis)
* Redo embeddings without lower threshold (maybe rare diseases disabilities wouldn't appear but have similar embeddings)
* Do embeddings over wikipedia?, only over the dataset, download some of them?

# Clear things

Tokenize everything with NLTK
Features:
* Decide first features (as strings)
* Create translation table with: feature_string, feature_id, feature_frequency
* Prune features with few occurrences (try different thresholds)
* Ideas: word, lemma, isCapitalized, isInsideParenthesis, isInFileX (provided lists), previous/next word, maybe
something about relation with previous for acronyms, isAcronym (probability), etc...

For LSTM too few examples, try to train embeddings before hand and then check.
