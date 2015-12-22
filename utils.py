import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def visualize_representation(repr, labels, title='', color_map=('red', 'green', 'blue', 'black')):
    classes = sorted(set(labels))
    print(classes)
    class_to_int = {c:i for i, c in enumerate(classes)}

    lda = LDA(n_components=2)
    proj = lda.fit(repr, labels).transform(repr)

    fig = plt.figure()
    plt.title(title)
    points = []
    for i,c in enumerate(classes):
        current_class = proj[labels == c, :]
        p = plt.scatter(current_class[:,0], current_class[:,1], color=color_map[i], alpha=0.4)
        points.append(p)
    plt.legend(points, classes, scatterpoints=1, loc='upper right', ncol=2, fontsize=8)

    return fig
