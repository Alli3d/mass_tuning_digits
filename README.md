# mass_tuning_digits
Uses ML model and parameter optimisation to best classify numbers from the sklearn handwritten digits dataset.

Uses PCA before training to limit the dminesion of the training set while not jeopardising the success of the model.

## Usage
`main.py` prepares the data and trains the algorithm.
It outputs a database of the models (SVM, decision trees, log reg, gaussian and multinomial naive bayes, random forest), as well as their best score, and their optimal parameters.
