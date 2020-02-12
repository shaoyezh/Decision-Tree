from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import graphviz
import os
import math
import matplotlib.pyplot as plt


def load_data(files):
    """load data
    """
    x = []
    y = []
    f = open(files[0], "r")
    lines = f.readlines()
    for line in lines:
        x.append(line.strip('\n'))
        y.append(0)   # when it is fake
    f.close()
    f = open(files[1], "r")
    lines = f.readlines()
    for line in lines:
        x.append(line.strip('\n'))
        y.append(1)    # when it is real
    f.close()
    vectorizer_x = CountVectorizer()
    vector_x = vectorizer_x.fit_transform(x)
    print(vector_x.__class__)
    splitter_name = vectorizer_x.get_feature_names()
    vector_x = vector_x.toarray()
    print(vector_x.__class__)
    #print(y)
    x_train, x_test_validation, y_train, y_test_validation = train_test_split(vector_x, y, test_size=0.3, shuffle=True)
    x_test, x_validation, y_test, y_validation = train_test_split(x_test_validation, y_test_validation, test_size=0.5)

    return [x_train, y_train, x_test, y_test,  x_validation, y_validation], splitter_name


def select_tree_model(data, splitter_name):
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]
    y_test = data[3]
    x_validation = data[4]
    y_validation = data[5]
    # entropy
    entropy1 = DecisionTreeClassifier(criterion="entropy", max_depth=4)
    entropy2 = DecisionTreeClassifier(criterion="entropy", max_depth=8)
    entropy3 = DecisionTreeClassifier(criterion="entropy", max_depth=16)
    entropy4 = DecisionTreeClassifier(criterion="entropy", max_depth=32)
    entropy5 = DecisionTreeClassifier(criterion="entropy", max_depth=64)
    entropy_lst = [entropy1, entropy2, entropy3, entropy4, entropy5]
    # gini
    gini1 = DecisionTreeClassifier(criterion="gini", max_depth=4)
    gini2 = DecisionTreeClassifier(criterion="gini", max_depth=8)
    gini3 = DecisionTreeClassifier(criterion="gini", max_depth=16)
    gini4 = DecisionTreeClassifier(criterion="gini", max_depth=32)
    gini5 = DecisionTreeClassifier(criterion="gini", max_depth=64)
    gini_lst = [gini1, gini2, gini3, gini4, gini5]
    for i in range(5):
        trained_tree = entropy_lst[i].fit(x_train, y_train)
        predict_resault = trained_tree.predict(x_validation)
        num = 0
        for j in range(len(predict_resault)):
            if predict_resault[j] == y_validation[j]:
                num += 1
        score = num / len(predict_resault)
        print("Entropy Accuracy:" + str(score) + '\n')
    for i in range(5):
        trained_tree = gini_lst[i].fit(x_train, y_train)
        predict_resault = trained_tree.predict(x_validation)
        num = 0
        for j in range(len(predict_resault)):
            if predict_resault[j] == y_validation[j]:
                num += 1
        score = num / len(predict_resault)
        print("Gini Accuracy:" + str(score) + '\n')
    trained_tree = entropy_lst[4].fit(x_train, y_train)
    predict_resault = trained_tree .predict(x_test)
    num = 0
    # Test accuracy for entropy and gini
    for j in range(len(predict_resault)):
        if predict_resault[j] == y_test[j]:
            num += 1
    score = num / len(predict_resault)
    print("Entropy Test Accuracy:" + str(score) + '\n')
    trained_tree = gini_lst[4].fit(x_train, y_train)
    predict_resault = trained_tree .predict(x_test)
    num = 0
    for j in range(len(predict_resault)):
        if predict_resault[j] == y_test[j]:
            num += 1
    score = num / len(predict_resault)
    print("gini Test Accuracy:" + str(score) + '\n')

    trained_classifier_plot = gini_lst[4].fit(x_train, y_train)
    dot_data = tree.export_graphviz(trained_classifier_plot, out_file=None, feature_names=splitter_name, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.format = "png"
    graph.render("CSC311_A1_Q3_C_gini")

    trained_classifier_plot_2 = entropy_lst[4].fit(x_train, y_train)
    dot_data_2 = tree.export_graphviz(trained_classifier_plot_2, out_file=None, feature_names=splitter_name,
                                      rounded=True)
    graph_2 = graphviz.Source(dot_data_2)
    graph.format = "png"
    graph_2.render("CSC311_A1_Q3_C_entropy")


def compute_information_gain(samples, labels, splitters, split):
    """compute the information gain of this split"""
    split_index = splitters.index(split)
    # calculate the total fake and real
    total_real = 0
    total_fake = 0

    for j in labels:
        if j == 0:
            total_fake += 1
        else:
            total_real += 1
    total = total_fake + total_real
    left = 0
    left_fake = 0
    left_real = 0
    right = 0
    right_fake = 0
    right_real = 0
    for i in range(len(samples)):
        if samples[i][split_index] > 0:
            left += 1
            if labels[i] == 1:  # when it is real
                left_real += 1
            else:
                left_fake += 1
        else:
            right += 1
            if labels[i] == 1:  # when it is real
                right_real += 1
            else:
                right_fake += 1

    # compute the IG of the split
    root_entropy = -(total_real / total) * math.log2((total_real / total)) - (total_fake / total) * \
                   math.log2((total_fake / total))
    left_entropy = -(left_real / left) * math.log2(left_real / left) - (left_fake / left) * math.log2(left_fake / left)
    right_entropy = -(right_fake / right) * math.log2(right_fake / right) - (right_real / right) * math.log2(
        right_real / right)

    return root_entropy - (left / (left + right)) * left_entropy - (right / (left + right)) * right_entropy


def knn_load_data(files):
    """load knn data"""
    x = []
    y = []
    f = open(files[0], "r")
    lines = f.readlines()
    for line in lines:
        x.append(line.strip('\n'))
        y.append(0)   # when it is fake
    f.close()
    f = open(files[1], "r")
    lines = f.readlines()
    for line in lines:
        x.append(line.strip('\n'))
        y.append(1)    # when it is real
    f.close()
    vectorizer_x = CountVectorizer()
    vector_x = vectorizer_x.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(vector_x, y, test_size=0.3)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)
    return [x_train, y_train, x_test, y_test, x_val, y_val]


def select_knn_model(x_train, y_train, x_test, y_test, x_val, y_val):
    """KNN classifer"""
    val_errors = []
    knn = []
    train_errors = []
    for i in range(1, 21):
        clf = KNeighborsClassifier(n_neighbors=i)
        clf.fit(x_train, y_train)
        accuracy = clf.score(x_val, y_val)
        train_error = 1 - clf.score(x_train, y_train)
        val_errors.append(1-accuracy)
        knn.append(i)
        train_errors.append(train_error)
    n = val_errors.index(min(val_errors)) + 1
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    print("the best n = " + str(n) + ". The validation accuracy is "+ str(accuracy))
    plt.plot(knn, val_errors, color='orange')
    plt.plot(knn, train_errors, color='blue')
    plt.xlabel("k - Number of Nearest Neighbors")
    plt.ylabel("Test error")
    plt.show()


if __name__ == '__main__':
    os.chdir(os.getcwd() + '/HW1')
    file_list = ["clean_fake.txt", "clean_real.txt"]
    # data_set, splitter = load_data(file_list)
    # select_tree_model(data_set, splitter)
    # donald = compute_information_gain(data_set[0], data_set[1], splitter, "donald")
    # trumps = compute_information_gain(data_set[0], data_set[1], splitter, "trumps")
    # the = compute_information_gain(data_set[0], data_set[1], splitter, "the")
    # print("The I.G. of 'the' is: " + str(the))
    # print("The I.G. of 'trumps' is: " + str(trumps))
    # print("The I.G. of 'donald' is: " + str(donald))
    data = knn_load_data(file_list)
    select_knn_model(data[0], data[1], data[2], data[3], data[4], data[5])
