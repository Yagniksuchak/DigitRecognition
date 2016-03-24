import numpy
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

COMPONENT_NUM = 150

print('Read training data...')
with open('mnist_train_100.csv', 'r') as reader:
    # reader.readline()
    train_label = []
    train_data = []
    for line in reader.readlines():
        data = map(int, line.rstrip().split(','))
        train_label.append(data[0])
        train_data.append(data[1:])
print('Loaded ' + str(len(train_label)))

print('Reduction...')
train_label = numpy.array(train_label)
train_data = numpy.array(train_data)
pca = PCA(n_components=COMPONENT_NUM, whiten=True)
pca.fit(train_data)
train_data = pca.transform(train_data)

print('Train SVM...')
svc = SVC()
svc.fit(train_data, train_label)

print('Read testing data...')
with open('mnist_test_10.csv', 'r') as reader:
    # reader.readline()
    test_data = []
    test_label= []
    for line in reader.readlines():
        label_pixels = map(int, line.rstrip().split(','))
        label=label_pixels[0]
        pixels=label_pixels[1:]
        test_data.append(pixels)
        test_label.append(label)
print('Loaded ' + str(len(test_data)))

print('Predicting...')
test_data = numpy.array(test_data)
test_data = pca.transform(test_data)
predict = svc.predict(test_data)
# print predict
# print test_label

print('Saving...')
with open('predict.csv', 'w') as writer:
    writer.write('"ImageId","Label"\n')
    count = 0
    for p in predict:
        count += 1
        writer.write(str(count) + ',"' + str(p) + '"\n')


print('Accuracy...')
print accuracy_score(test_label,predict)