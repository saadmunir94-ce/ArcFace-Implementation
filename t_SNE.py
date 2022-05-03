from sklearn import manifold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
label_dict = {v: k for k, v in label_dict.items()}
def computeTSNEProjectionOfLatentSpace(X_encoded, label, path_of_visualizations, epoch):
    """
    Computes T-SNE non-linear manifold of the last weight layer
    Args:
        X_encoded: torch.tensor
            Logits before activation function
        label: torch.tensor
            Corresponding class labels
        path_of_visualizations: str
            Directory to store plots into
        epoch: int
            the epoch in consideration


    Returns: None

    """
    model = manifold.TSNE(n_components=3, random_state=0, learning_rate=10, perplexity=20)
    tsne_data = model.fit_transform(X_encoded).T
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'tab:blue', 'tab:orange', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan', 'yellow', 'tomato']

    for i in range(len(label)):
        color = colors[label[i].item()]
        ax.scatter(tsne_data[0, i], tsne_data[1, i], tsne_data[2, i], color=color, label=label_dict[label[i].item()])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title("T-SNE 3D Projection")
    plt.grid(b=True)
    plt.savefig(path_of_visualizations + '/tsne_epoch' + str(epoch) + '.jpg')
    plt.close()
    return None