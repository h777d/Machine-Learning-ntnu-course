# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # TTT4185 Machine learning for Speech technology
# ## Computer assignment 2: Classification using the Bayes Decision Rule and Support Vector Machines
# ## Suggested Solution
#

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib notebook

import scipy.stats
from sklearn import svm
from sklearn.metrics import confusion_matrix


# -

# Before starting on the solution we define some useful helper functions.
#
# The following nice helper-function is stolen from scikit learn. Very useful for visualizing confusion matrices.

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.subplots(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # solving a plot cutoff bug
    plt.ylim(len(cm)-0.5, -0.5)
    plt.show()


# ### Problem 1
#
# (a) Just by eyeing the plots, it seems that "ae" should be easy to separate from "ux". As for the two other classes it is not so clear.

# +
# Load data
train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")

# Extract vowels
aes = train[train["Phoneme"] == 'ae']
eys = train[train["Phoneme"] == 'ey']
uxs = train[train["Phoneme"] == 'ux']

# Plotting here
fig, ax = plt.subplots(1,3, figsize=(12,4))
ax[0].scatter(aes["F1"],aes["F2"],color='r',marker='.',label="aes")
ax[0].scatter(eys["F1"],eys["F2"],color='b',marker='.',label="eys")
ax[0].set_xlim([0,1])
ax[0].set_ylim([1,3])
ax[0].legend()

ax[1].scatter(aes["F1"],aes["F2"],color='r',marker='.',label="aes")
ax[1].scatter(uxs["F1"],uxs["F2"],color='g',marker='.',label="uxs")
ax[1].set_xlim([0,1])
ax[1].set_ylim([1,3])
ax[1].legend()

ax[2].scatter(eys["F1"],eys["F2"],color='b',marker='.',label="eys")
ax[2].scatter(uxs["F1"],uxs["F2"],color='g',marker='.',label="uxs")
ax[2].set_xlim([0,1])
ax[2].set_ylim([1,3])
ax[2].legend()

fig.tight_layout()


# -

# (b) Here we create a Gaussian Naive Bayes classifier. First create a few functions that will make our solution a bit tidier.

# +
# Function that computes the ML estimate given feature vectors in the rows of X
def MLGaussian(X):
    mean = np.mean(X)
    cov = np.cov(X, rowvar=False)
    return (mean, cov)

# Function that computes the a posteriori class probability for a set of observations X
# (models contains Gaussian models and priors for classes 1 through C)
def gaussClassifier(X, models):
    # Compute all probabilites
    numclasses = len(models)
    numobservations = X.shape[0]
    Px = np.zeros((numobservations, numclasses))
    for c, model in enumerate(models):
        mvn = scipy.stats.multivariate_normal(model["mean"], model["cov"])
        Px[:,c] = mvn.pdf(X)*model["prior"]
    # Normalize
    Pnorm = np.sum(Px, axis=1)

    return (Px.T/Pnorm).T

# function to plot countours for each Gaussian distribution in the model (that is P(x|c_i)
# assuming 2 dimentional input space.
# models contains Gaussian models and priors for classes 1 through C, each element in the
# list is a dictionary with the keys: "means", "cov" and "prior" containing mean vector,
# covariance matrix and prior probability for each class.
def plotGaussians(models, colors, ax=plt.gca()):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x, y = np.mgrid[xlim[0]:xlim[1]:(xlim[1]-xlim[0])/500.0, ylim[0]:ylim[1]:(ylim[1]-ylim[0])/500.0]
    xy = np.dstack((x, y))
    for idx, model in enumerate(models):
        mvn = scipy.stats.multivariate_normal(model["mean"], model["cov"])
        lik = mvn.pdf(xy)
        ax.contour(x,y,lik, colors=colors[idx])

# plots decision regions for a Bayesian classifier, assuming 2 dimensional input space
# models as in plotGaussians
def plotDecisionRegions(models, colors, ax=plt.gca()):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x, y = np.mgrid[xlim[0]:xlim[1]:(xlim[1]-xlim[0])/500.0, ylim[0]:ylim[1]:(ylim[1]-ylim[0])/500.0]
    xy = np.dstack((x, y))
    K = len(models)
    N, M = x.shape
    scaled_posteriors = np.zeros((N,M,K))
    for idx, model in enumerate(models):
        mvn = scipy.stats.multivariate_normal(model["mean"], model["cov"])
        scaled_posteriors[:,:,idx] = mvn.pdf(xy) * model["prior"]
    regions = scaled_posteriors.argmax(axis=2)
    ax.contourf(x, y, regions, levels=np.arange(-0.5,K), colors=colors, alpha=0.2)
    return regions


# -

# Now extract the data and compute the ML (maximum likelihood) estimates for each class, as well as the prior probability of each class. The prior probability of a class is simply the fraction seen in the training set.

# +
# Extract the feature vectors for each of the three classes and compute the ML estimates
aeTrain = aes[["F1","F2"]]
aeMean, aeCov = MLGaussian(aeTrain)

eyTrain = eys[["F1","F2"]]
eyMean, eyCov = MLGaussian(eyTrain)

uxTrain = uxs[["F1","F2"]]
uxMean, uxCov = MLGaussian(uxTrain)

# Compute the prior estimates using counting
trainSize = aeTrain.shape[0]+eyTrain.shape[0]+uxTrain.shape[0]
aePrior = aeTrain.shape[0]/trainSize
eyPrior = eyTrain.shape[0]/trainSize
uxPrior = uxTrain.shape[0]/trainSize

# Put the models in a list
models = list()
models.append({"mean":aeMean, "cov":aeCov, "prior":aePrior})
models.append({"mean":eyMean, "cov":eyCov, "prior":eyPrior})
models.append({"mean":uxMean, "cov":uxCov, "prior":uxPrior})
# -

# plot contours of the class conditional likelihoods and decision regions
fig, ax = plt.subplots(1, 2, figsize=(12,6))
ax[0].scatter(aes["F1"],aes["F2"],color='r',marker='.',label="aes")
ax[0].scatter(eys["F1"],eys["F2"],color='b',marker='.',label="eys")
ax[0].scatter(uxs["F1"],uxs["F2"],color='g',marker='.',label="uxs")
ax[0].set_xlim([0,1])
ax[0].set_ylim([1,3])
plotGaussians(models, ['r', 'b', 'g'], ax[0])
ax[0].legend()
ax[1].scatter(aes["F1"],aes["F2"],color='r',marker='.',label="aes")
ax[1].scatter(eys["F1"],eys["F2"],color='b',marker='.',label="eys")
ax[1].scatter(uxs["F1"],uxs["F2"],color='g',marker='.',label="uxs")
ax[1].set_xlim([0,1])
ax[1].set_ylim([1,3])
plotDecisionRegions(models, ['r', 'b', 'g'], ax[1])
ax[1].legend()


# The above figure shows that the class conditional distributions $p(x|c)$ are highly overlapping (because the data is not easily separable in two dimensions. The corresponding decision regions are computed by evaluating the maximu

# Prepare the test set. We use the phoneme names as labels.

# +
# A little bit of Pandas magic to extract both the phoneme and features in one swoop
aeTest = test[test["Phoneme"] == 'ae'].loc[:,["F1","F2"]]
eyTest = test[test["Phoneme"] == 'ey'].loc[:,["F1","F2"]]
uxTest = test[test["Phoneme"] == 'ux'].loc[:,["F1","F2"]]

# Stack the rows from the three matrixes on top of each other to create on test matrix
Test = np.vstack((aeTest, eyTest, uxTest))
Testlabels = np.array(aeTest.shape[0]*["ae"] + eyTest.shape[0]*["ey"] + uxTest.shape[0]*["ux"])
phonemes = np.array(["ae","ey","ux"])
# -

# Perform classification, compute the confusion matrix and plot the results.

# +
# gaussClassifier function will compute P(c|x) or all classes c and observations x
P = gaussClassifier(Test, models)

# Each row in P now holds p(c|x) for all c. We find the maximum for each row
# and its corresponding label
labels = phonemes[np.argmax(P, axis=1)]

# Compute the confusion matrix
cm = confusion_matrix(Testlabels, labels)
plot_confusion_matrix(cm, phonemes)
errors = np.sum(Testlabels != labels)
print("Error rate:", round(100*errors/len(labels),2), "%")
# -

# As we see from this result there is very little confusion between "ae" and "ux", as we expected in (a). We got 80+97+23=200 correct out of 271 examples, making the error rate 26.2%

# (c) First we include the features "F1"-"F4" instead of only "F1" and "F2".

# +
# Cut and paste from prev problem
aeTrain = aes[["F1","F2", "F3", "F4"]]
aeMean, aeCov = MLGaussian(aeTrain)
eyTrain = eys[["F1","F2", "F3", "F4"]]
eyMean, eyCov = MLGaussian(eyTrain)
uxTrain = uxs[["F1","F2", "F3", "F4"]]
uxMean, uxCov = MLGaussian(uxTrain)

models = list()
models.append({"mean":aeMean, "cov":aeCov, "prior":aePrior})
models.append({"mean":eyMean, "cov":eyCov, "prior":eyPrior})
models.append({"mean":uxMean, "cov":uxCov, "prior":uxPrior})

aeTest = test[test["Phoneme"] == 'ae'].loc[:,["F1","F2","F3","F4"]]
eyTest = test[test["Phoneme"] == 'ey'].loc[:,["F1","F2","F3","F4"]]
uxTest = test[test["Phoneme"] == 'ux'].loc[:,["F1","F2","F3","F4"]]

Test = np.vstack((aeTest, eyTest, uxTest))

P = gaussClassifier(Test, models)

labels = phonemes[np.argmax(P, axis=1)]

cm = confusion_matrix(Testlabels, labels)
plot_confusion_matrix(cm, phonemes)
errors = np.sum(Testlabels != labels)
print("Error rate:", round(100*errors/len(labels),2), "%")
# -

# Again we see from this result there is very little confusion between "ae" and "ux". We got 82+98+25=205 correct out of 271 examples, making the error rate 24.4%. This is about 1.8% better than when using only 2 features, although the improvement is minor.
#
# Finally, we include all the continous features "F1"-"F4" and "B1"-"B4" by repeating the same steps.

# +
# Cut and paste from prev problem
aeTrain = aes[["F1","F2","F3","F4","B1","B2","B3","B4"]]
aeMean, aeCov = MLGaussian(aeTrain)
eyTrain = eys[["F1","F2","F3","F4","B1","B2","B3","B4"]]
eyMean, eyCov = MLGaussian(eyTrain)
uxTrain = uxs[["F1","F2","F3","F4","B1","B2","B3","B4"]]
uxMean, uxCov = MLGaussian(uxTrain)

models = list()
models.append({"mean":aeMean, "cov":aeCov, "prior":aePrior})
models.append({"mean":eyMean, "cov":eyCov, "prior":eyPrior})
models.append({"mean":uxMean, "cov":uxCov, "prior":uxPrior})

aeTest = test[test["Phoneme"] == 'ae'].loc[:,["F1","F2","F3","F4","B1","B2","B3","B4"]]
eyTest = test[test["Phoneme"] == 'ey'].loc[:,["F1","F2","F3","F4","B1","B2","B3","B4"]]
uxTest = test[test["Phoneme"] == 'ux'].loc[:,["F1","F2","F3","F4","B1","B2","B3","B4"]]

Test = np.vstack((aeTest, eyTest, uxTest))

P = gaussClassifier(Test, models)

labels = phonemes[np.argmax(P, axis=1)]

cm = confusion_matrix(Testlabels, labels)
plot_confusion_matrix(cm, phonemes)
errors = np.sum(Testlabels != labels)
print("Error rate:", round(100*errors/len(labels),2), "%")
# -

# By including "B1"-"B4", the error rate increase from around 24.5% to 28.8%. So in this case, using less features, "F1"-"F4", resulted in better classification performance.

# (d) If we have models $p(x|g,c)$ for all classes $c$, we can compute
#     \begin{equation}
#     p(x|c) = \sum_g p(x|g,c)p(g)
#     \end{equation}
#     where we assume $p(g)=0.5$ (50/50 change of male or female).
#     Hopefully this model should be more precise.

# +
# Estimate gender specific models
aesM = train[(train["Phoneme"] == 'ae') & (train["Gender"] == 'M')]
eysM = train[(train["Phoneme"] == 'ey') & (train["Gender"] == 'M')]
uxsM = train[(train["Phoneme"] == 'ux') & (train["Gender"] == 'M')]
aesF = train[(train["Phoneme"] == 'ae') & (train["Gender"] == 'F')]
eysF = train[(train["Phoneme"] == 'ey') & (train["Gender"] == 'F')]
uxsF = train[(train["Phoneme"] == 'ux') & (train["Gender"] == 'F')]
aeTrainM = aesM[["F1","F2", "F3", "F4"]]
aeMeanM, aeCovM = MLGaussian(aeTrainM)
eyTrainM = eysM[["F1","F2", "F3", "F4"]]
eyMeanM, eyCovM = MLGaussian(eyTrainM)
uxTrainM = uxsM[["F1","F2", "F3", "F4"]]
uxMeanM, uxCovM = MLGaussian(uxTrainM)
aeTrainF = aesF[["F1","F2", "F3", "F4"]]
aeMeanF, aeCovF = MLGaussian(aeTrainF)
eyTrainF = eysF[["F1","F2", "F3", "F4"]]
eyMeanF, eyCovF = MLGaussian(eyTrainF)
uxTrainF = uxsF[["F1","F2", "F3", "F4"]]
uxMeanF, uxCovF = MLGaussian(uxTrainF)

uxPrior = uxTrain.shape[0]/trainSize

# Put the models in a list
modelsG = list()
modelsG.append({"meanM":aeMeanM, "covM":aeCovM, "meanF":aeMeanF, "covF":aeCovF, "prior":aePrior})
modelsG.append({"meanM":eyMeanM, "covM":eyCovM, "meanF":eyMeanF, "covF":eyCovF, "prior":eyPrior})
modelsG.append({"meanM":uxMeanM, "covM":uxCovM, "meanF":uxMeanF, "covF":uxCovF, "prior":uxPrior})


# -

# Define a new function that computes the posterior probability of a class given the gender-dependent models.

# Function that computes the a posteriori class probability for a set of observations X
# (models contains two Gaussian models, one male and one female, and priors for classes 1 through C)
def gaussClassifierG(X, models):
    # Compute all probabilites
    numclasses = len(models)
    numobservations = X.shape[0]
    Px = np.zeros((numobservations, numclasses))
    for c, model in enumerate(models):
        mvnM = scipy.stats.multivariate_normal(model["meanM"], model["covM"])
        mvnF = scipy.stats.multivariate_normal(model["meanF"], model["covF"])
        Px[:,c] = 0.5*mvnM.pdf(X)*model["prior"]
        Px[:,c] += 0.5*mvnF.pdf(X)*model["prior"]
    # Normalize
    Pnorm = np.sum(Px, axis=1)

    return (Px.T/Pnorm).T


# Classify the test set and visualize the results.

# +
aeTest = test[test["Phoneme"] == 'ae'].loc[:,["F1","F2","F3","F4"]]
eyTest = test[test["Phoneme"] == 'ey'].loc[:,["F1","F2","F3","F4"]]
uxTest = test[test["Phoneme"] == 'ux'].loc[:,["F1","F2","F3","F4"]]

Test = np.vstack((aeTest, eyTest, uxTest))

P = gaussClassifierG(Test, modelsG)

labels = phonemes[np.argmax(P, axis=1)]

cm = confusion_matrix(Testlabels, labels)
plot_confusion_matrix(cm, phonemes)
errors = np.sum(Testlabels != labels)
print("Error rate:", round(100*errors/len(labels),2), "%")
# -

# Even better performance, but only very slightly. All in all the error rate is 81+99+26=206 correct out of 271 examples, making the error rate 23.98%. This is about 1.8% better than when using only 2 features, although the improvement is minor.

# (e) To use only diagonal covariance matrices, we simply set the off diagonal of the covariance matrices in the previous problem to zero, and repeat the classification. We do this by a elementwise multiplication with the identity matrix.
#
# Looking at the confusion matrix, the new results with the diagonal matrices are a bit different, but the total number of errors are the same.

# +
I = np.eye(4)
modelsGD = list()
modelsGD.append({"meanM":aeMeanM, "covM":aeCovM*I, "meanF":aeMeanF, "covF":aeCovF*I, "prior":aePrior})
modelsGD.append({"meanM":eyMeanM, "covM":eyCovM*I, "meanF":eyMeanF, "covF":eyCovF*I, "prior":eyPrior})
modelsGD.append({"meanM":uxMeanM, "covM":uxCovM*I, "meanF":uxMeanF, "covF":uxCovF*I, "prior":uxPrior})

P = gaussClassifierG(Test, modelsGD)

labels = phonemes[np.argmax(P, axis=1)]

cm = confusion_matrix(Testlabels, labels)
plot_confusion_matrix(cm, phonemes)
errors = np.sum(Testlabels != labels)
print("Error rate:", round(100*errors/len(labels),2), "%")
# -

# ### Problem 2

# (a) First prepare test and training data (this is done earlier in the script for Problem 1, but we repeat it here so that the problems are a bit independent).
#
# Then setup a linear SVM and train the model for the penalty term $C \in \{0.1, 1, 10\} $
#
# We see that the error rate goes down when increasing $C$. This is natural as increasing $C$ trades separation margin for fewer errors.

# +
# Get the training and test data
aeTrain = train[train["Phoneme"] == 'ae'].loc[:,["F1","F2","F3","F4"]]
eyTrain = train[train["Phoneme"] == 'ey'].loc[:,["F1","F2","F3","F4"]]
uxTrain = train[train["Phoneme"] == 'ux'].loc[:,["F1","F2","F3","F4"]]
Train = np.vstack((aeTrain, eyTrain, uxTrain))

aeTest = test[test["Phoneme"] == 'ae'].loc[:,["F1","F2","F3","F4"]]
eyTest = test[test["Phoneme"] == 'ey'].loc[:,["F1","F2","F3","F4"]]
uxTest = test[test["Phoneme"] == 'ux'].loc[:,["F1","F2","F3","F4"]]
Test = np.vstack((aeTest, eyTest, uxTest))

# Create training and test labels
Trainlabels = np.array(aeTrain.shape[0]*["ae"] + eyTrain.shape[0]*["ey"] + uxTrain.shape[0]*["ux"])
Testlabels = np.array(aeTest.shape[0]*["ae"] + eyTest.shape[0]*["ey"] + uxTest.shape[0]*["ux"])
phonemes = np.array(["ae","ey","ux"])
# -



for C in [0.1, 1, 10]:
    print("C = ", C)
    clf = svm.SVC(C=C, kernel='linear')
    clf.fit(Train, Trainlabels)
    labelsTr = clf.predict(Train)
    labelsTe = clf.predict(Test)

    cmTe = confusion_matrix(Testlabels, labelsTe)
    cmTr = confusion_matrix(Trainlabels, labelsTr)

    errors = np.sum(Trainlabels != labelsTr)
    print("Training set error rate:", round(100*errors/len(labelsTr),2), "%")
    errors = np.sum(Testlabels != labelsTe)
    print("Test set error rate:", round(100*errors/len(labelsTe),2), "%")

    plot_confusion_matrix(cmTe, phonemes)
    # plot support vectors
    fig, ax = plt.subplots(1, figsize=(8,8))
    ax.scatter(aes["F1"],aes["F2"],color='r',marker='.',label="aes")
    ax.scatter(eys["F1"],eys["F2"],color='b',marker='.',label="eys")
    ax.scatter(uxs["F1"],uxs["F2"],color='g',marker='.',label="uxs")
    ax.set_xlim([0,1])
    ax.set_ylim([1,3])
    ax.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], alpha=0.4)
    print('# support vectors: ', clf.n_support_)

# (b) For the linear and RBF kernels, we note a relevant decay on the error rates when increasing the value of the penalty term. While for the polynomial kernel, there is only a slight decrease. The sigmoid kernel achieved high error rates for all values of $C$, showing to be inadequate for this problem.
#
# In all the cases, the test set error rates followed the same trend as for the train set. As expected, the test set presented higher error rates than the train set.

Cs = [0.1, 0.5, 1, 2, 5, 7, 10]
for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
    errors_Train,errors_Test = [],[]
    for C in Cs:
        clf = svm.SVC(C=C, kernel=kernel, gamma="auto")
        clf.fit(Train, Trainlabels)
        labelsTr = clf.predict(Train)
        labelsTe = clf.predict(Test)

        cmTe = confusion_matrix(Testlabels, labelsTe)
        cmTr = confusion_matrix(Trainlabels, labelsTr)

        # plot_confusion_matrix(cmTe, phonemes)
        errors = np.sum(Trainlabels != labelsTr)
        errors_Train.append(errors/len(labelsTr))
        errors = np.sum(Testlabels != labelsTe)
        errors_Test.append(errors/len(labelsTe))

    plt.plot(Cs,errors_Train,label="Train")
    plt.plot(Cs,errors_Test,label="Test")
    plt.ylim(0,0.6)
    plt.title(kernel)
    plt.legend()
    plt.show()
