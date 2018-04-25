import os
import tarfile
import urllib.request
import numpy as np

training_data = None
testing_data = None
batch_index = 0


def download_and_extract():
    """Download the data and extract to folder."""
    dest = './cifar10_data/'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    if not os.path.exists(dest):
        filename = url.split('/')[-1]
        filepath = os.path.join(dest, filename)
        if not os.path.exists(filepath):
            os.makedirs(dest)

            def progress(count, block_size, total_size):
                perc = float(count * block_size) / float(total_size)
                width = 80
                print(
                    '\r>> Downloading {} [{}] {:.1f}%'.format(
                        filename, ('=' * int(perc * width)) + '>' + (' ' * int(
                            (1.0 - perc) * width)), perc * 100.0),
                    end='')

            filepath, _ = urllib.request.urlretrieve(url, filepath, progress)
            print()
            statinfo = os.stat(filepath)
            print("   Successfully downloaded", filename, statinfo.st_size,
                  'bytes.')
        extract_dir = os.path.join(dest, 'cifar-10-batches')
        if not os.path.exists(extract_dir):
            print(">> Extracting {}".format(filepath))
            tarfile.open(filepath, 'r:gz').extractall(dest)
            print("   Successfully extracted {}".format(filepath))


def unpickle(file):
    import pickle
    with open(file, 'rb') as file:
        vals = pickle.load(file, encoding='bytes')
    return vals


def get_data(file):
    download_and_extract()
    if isinstance(file, int):
        absFile = os.path.abspath("cifar10_data/cifar-10-batches-py/data_batch_{}".format(file))
    else:
        absFile = os.path.abspath("cifar10_data/cifar-10-batches-py/{}".format(file))
    dict = unpickle(absFile)
    X = np.asarray(dict[b'data'].T).astype("uint8")
    Yraw = np.asarray(dict[b'labels'])
    Y = np.zeros((10, 10000))
    for i in range(10000):
        Y[Yraw[i], i] = 1
    names = np.asarray(dict[b'filenames'])
    return X, Y, names

def load_batch(fpath):
    import pickle
    with open(fpath, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    data = np.asarray(d[b'data'].T).astype('uint8')
    labels = np.asarray(d[b'labels'])
    labels_raw = np.asarray(d[b'labels'])
    labels = np.zeros((10, 10000))
    for i in range(10000):
        labels[labels_raw[i], i] = 1
    names = np.asarray(d[b'filenames'])

    return data, labels, names

def load_data(batch=1):
    download_and_extract()
    dir_name = 'cifar10_data/cifar-10-batches-py/'
    if isinstance(batch, int):
        x_data, y_data, names= load_batch(dir_name + "data_batch_{}".format(batch))
    else:
        x_data = None
        y_data = None
        names = None
        for bat in batch:
            x, y, na= load_batch(dir_name + "data_batch_{}".format(bat))
            if x_data is not None:
                x_data = np.concatenate((x_data, x), axis=1)
                y_data = np.concatenate((y_data, y), axis=1)
                names = np.concatenate((names, na))
            else:
                x_data = x
                y_data = y
                names = na
    return x_data.T, y_data.T, names

def import_data(training=1, testing='test'):
    global training_data
    global testing_data
    training_data = load_data(training)
    testing_data = load_data(testing)

def data_size():
    return len(training_data[0])

def next_batch(batch_size=50):
    global training_data
    global testing_data
    global batch_index
    X = training_data[0][batch_index:batch_index + batch_size]
    Y = training_data[1][batch_index:batch_index + batch_size]
    batch_index += batch_size
    if batch_index >= len(training_data[0]):
        batch_index = 0
    return X, Y


if __name__ == "__main__":
    download_and_extract()
