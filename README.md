# A Text Classifier based on Log-Linear Model

A simple model from scratch.

## Dataset
[AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

## Implementation
### Preprocess
[code](preprocess.py)

- Sample data to reduce dataset size
- Merge the Title and Content
- With the help of nltk:
    + Remove punctuations and numbers
    + Remove URLs
    + Split the Content into words
    + Filter stopwords
    + Stem and Lemmarize

### Feature Extraction
[code](tfidf.py)

Use TF-IDF as the Feature
- Keep only the most frequent words for a reasonable feature size 
- Calculate TF-IDF
$$TF(t,d) = \frac{\text{count}(t, d)}{\sum_k \text{count}(k, d)}$$
, where count($t$, $d$) means the count of term $t$ in document $d$.
$$IDF(t, D) = \log \frac{N + 1}{num(t, D) + 1} + 1$$ 
, where num($t$, $D$) means the number of documents in $D$ that contains term $t$, and $D$ is the set of all documents.
$$TF-IDF = TF \times IDF$$
Note that L2 normalisation is applied to the final TF-IDF for a better performance.

### Log-Linear Model
[code](loglinear.py)

- Logistic Regression Model
$$\hat{y} = \text{softmax}\big( X_{N,F}W_{F,C}+b_{C} \big) $$
, where $N$ is the size of train data, $F$ is the number of features and $C$ is the number of classes.

- Cross Entropy Loss

$$\text{loss} = -\frac{1}{N} \sum_{i=1}^N \sum_{j=1}^{C} y_{i,j} \log \hat{y}_{i,j}$$

, where $y_{i,j}=1$ if train text $i$ belongs to class $j$, and $0$ otherwise.

- Gradients
$$dW = \frac{1}{N} X^T \cdot\big(\hat{y} - y\big) \\
db =\frac{1}{N} \sum_{i=1}^{N} \big(\hat{y} - y\big) 
$$

### Update Algorithm
[code](loglinear.py)

Gradient Descend with a shrinking learning rate.

### Evalutaion
[code](evaluation.py)

- Accuracy

$$ \text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}$$
- F1 Score (macro)

$$ \text{Precision} = \frac{TP}{TP+FP} $$
$$ \text{Recall} = \frac{TP}{TP+FN} $$
$$ \text{F1 Score} = 2 \times \frac{PR}{P+R}$$
$$ \text{Macro F1 Score} = \frac{1}{C}\sum_{i=1}^{C}\text{F1 Score}
$$ 

