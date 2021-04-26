# CLab21
CL team lab course repository at 21SS

## List of (mostly used) classification model

**Classical**

- SVM
- K-Nearest Neighbors
- Naïve Bayes
- Hidden Markov model
- Decision Trees

**Deep Learning**

**pre-processing procedure**

1. tokenization (create separate unit)
2. numericalization
3. replace the tokens with id (the position) 
- WARNING: to prevent the case where weight matrix gets too huge, since every word has its own weight matrix, better to limit maximum vocabulary size by default (e.g. 60000) + least frequency
4. add unknown token(not common enough word to appear in our vocab), separator tag (when new document starts)
