import numpy as np
import matplotlib.pyplot as plt

# [Section 1.] Loading the data
X_train = np.loadtxt("iris_train.data", delimiter=" ")  # train data
X_test = np.loadtxt("iris_test.data", delimiter=" ")  # test data
y_train = np.loadtxt("iris_train.labels", delimiter=" ")  # train labels
y_test = np.loadtxt("iris_test.labels", delimiter=" ")  # test labels

print("\n Shapes of data:")
for N, _ in zip(['X_train', 'X_test ', 'y_train', 'y_test '], [X_train, X_test, y_train, y_test]):
    print("\t" + N + " : ", _.shape)

# [Section 2.] Classifying the data

""" Based on the plots from Step 3. ("look at the noteboooks"), it appears that Feature 2 and 3 do the best job of \
classification. Basically, this is * Dimensionality Reduction (Feature Selection) *. We are mainly interested in \
 using the features which can best classify the flower into the 3 classes. """


def classify(x):
    if x[-2] < 2.8 and x[-1] < 0.8:  # class 1
        return 0

    elif x[-2] > 4.5 and x[-1] > 1.75:  # class 3
        return 2

    else:  # class 2
        return 1


# [Section 3.] Testing the data
y_hat = np.array([classify(_) for _ in X_test])


# Calculating the test error
errors = 0
error_ID = []
for y_true, y_pred, ID in zip(y_test, y_hat, range(50)):
    if y_true != y_pred:
        errors += 1
        error_ID.append(ID)
        
test_error = errors / len(X_test) * 100
print('\nTest error is %g percent.' % test_error)

# plotting:
def plot_that():
    fig = plt.figure(1, figsize=(12, 8))
    ax = fig.subplots(1, 2, sharey=True)
    c = ["b", "r", "g"]
    for jj in [0, 1, 2]:
        ax[0].scatter(X_test[np.where(y_test == jj)[0]][:, 3], X_test[np.where(y_test == jj)[0]][:, 2], c=c[jj])
        ax[1].scatter(X_test[np.where(y_hat == jj)[0]][:, 3], X_test[np.where(y_hat == jj)[0]][:, 2], c=c[jj])

    ax[1].scatter( X_test[error_ID][:,3], X_test[error_ID][:,2], marker="x", s = 25, c="green")
    # ax[1].axvline(1.7, c="green", linestyle=":")
    # ax[1].axhline(4.5, c="g", linestyle=":")
    # ax[1].axvline(x = 0.8, ymin=0, ymax =2.8, c="b", linestyle="--")
    # ax[1].axhline(2.8, c="b", linestyle="--")

    ax[0].set_title("Train Data Classification")
    ax[0].grid(True)
    ax[0].legend(["Class-1", "Class-2", "Class-3"], loc=2)
    ax[1].legend(["Class-1", "Class-2", "Class-3", 'errors'], loc=2)
    ax[1].set_title("Test Data Classification")
    ax[1].grid(True)




    axN = fig.add_subplot(111, frameon=False)
    plt.xlabel("Feature 3", labelpad=30)
    plt.ylabel("Feature 2", labelpad=25)
    plt.title("Performance of Naive Classifier", pad=40, FontSize=35)
    plt.text(0.40, 1.05, "( Classification Error : "+str(test_error)+"% )")
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(wspace=.05)
    plt.show()


plot_that()
# fin.
