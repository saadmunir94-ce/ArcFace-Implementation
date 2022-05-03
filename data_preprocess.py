from skimage.io import imread
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import os

# arrange the labels
label_dict = {
    "Patty_Schnyder": 0,
    "Paul_Bremer": 1,
    "Paul_Burrell": 2,
    "Paul_Martin": 3,
    "Paul_McCartney": 4,
    "Paul_ONeill": 5,
    "Paul_Wolfowitz": 6,
    "Pedro_Almodovar": 7,
    "Pervez_Musharraf": 8,
    "Pete_Sampras": 9,
    "Peter_Struck": 10,
    "Pierce_Brosnan": 11,
    "Queen_Elizabeth_II": 12,
    "Rachel_Hunter": 13,
    "Ralf_Schumacher": 14,
    "Ralph_Lauren": 15,
    "Raquel_Welch": 16,
    "Ray_Nagin": 17
}
# Reverse keys and values for the dictionary
label_dict = {v: k for k, v in label_dict.items()}


def get_data(id):
    """
    Gets images and labels for category defined by index id
    Args:
        id: class index

    Returns:
        images: list
            list of NumPy images
        labels: list
            list of labels [id, ..., id]
        images.shape[0]: int
            no of images for label id
    """
    person = label_dict[id]
    images = [imread(file) for file in glob.glob('./arcFace_dataset/' + person + '/*.jpg')]
    images = np.stack(images)
    images = images.astype(np.float64)
    labels = np.full(shape=images.shape[0], fill_value=id)
    return images, labels, images.shape[0]


def get_count_dict(arr):
    """
    Returns Count of Unique categories in a Numpy Array as a Python Dictionary
    Args:
        arr: Numpy Array of class labels

    Returns:
        Count of Unique categories in a Numpy Array as a Python Dictionary
    """
    unique, counts = np.unique(arr, return_counts=True)
    return dict(zip(unique, counts))


keys = list(label_dict.keys())
sizes = []
total_images, total_labels, size = get_data(0)
sizes.append(size)
for key in keys[1:]:
    images, labels, size = get_data(key)
    sizes.append(size)
    total_images = np.concatenate((total_images, images))
    total_labels = np.concatenate((total_labels, labels))

# Scale images to [0,1] range
print("Before scaling, min: {}".format(np.min(total_images)))
print("Before scaling, max: {}".format(np.max(total_images)))
total_images /= np.max(total_images)
print("After scaling, min: {}".format(np.min(total_images)))
print("After scaling, max: {}".format(np.max(total_images)))
# Do the train test split
X_train, X_test, y_train, y_test = train_test_split(total_images, total_labels, test_size=0.30, stratify=total_labels,
                                                    random_state=1234)
print(get_count_dict(y_test))
# Set the directories
train_dir = "./train_data"
test_dir = "./test_data"
if not os.path.isdir(train_dir):
    print("train dir doesnt exist")
    os.makedirs(train_dir)
else:
    print("train dir exists")
np.save(file=os.path.join(train_dir, "train_images.npy"), arr=X_train)
np.save(file=os.path.join(train_dir, "train_labels.npy"), arr=y_train)
if not os.path.isdir(test_dir):
    print("test dir doesnt exist")
    os.makedirs(test_dir)
else:
    print("test dir exists")
np.save(file=os.path.join(test_dir, "test_images.npy"), arr=X_test)
np.save(file=os.path.join(test_dir, "test_labels.npy"), arr=y_test)
