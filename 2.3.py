import numpy as np
import matplotlib.pyplot as plt
 
 
WINDOW_OPEN = False
 
 
def plot(data, classes, centers):
    global WINDOW_OPEN
    if classes.max():
        colors = classes / classes.max()
    else:
        colors = classes
    plt.clf()
    plt.scatter(data[:, 0], data[:, 1], c=colors, s=100)
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=1000)
    if WINDOW_OPEN:
        plt.draw()
    else:
        WINDOW_OPEN = True
        plt.show(block=False)
 
 
def get_distances(points, centers):
    return np.sum(np.absolute(points[:, :, None] - centers.T[None, :, :]),
                  axis=1)
 
 
def get_centers(points, assignment, k):
    centers = np.zeros((k, 2))
    for index, _ in enumerate(centers):
        mask = (assignment == index)
        centers[index] = np.mean(points[mask, :], axis=0)
    return centers
 
 
def kmeans(data, k):
    centers = np.random.rand(k, 2)
    previous_assignment = np.zeros(len(data))
    while True:
        distances = get_distances(data, centers)
        assignment = np.argmin(distances, axis=1)
        plot(data, assignment, centers)
        input('Press any key to continue')
        if np.array_equal(previous_assignment, assignment):
            break
        centers = get_centers(data, assignment, k)
        previous_assignment = assignment
    return assignment
 
 
if __name__ == '__main__':
    # data = np.array([
    #     [3, 8], [4, 7], [3, 6], [3, 4],
    #     [4, 5], [5, 5], [5, 2], [8, 4], [9, 4], [9, 1]])
    data = np.random.rand(100, 2)
 
    classes = kmeans(data, k=3)
    input('Press any key to exit')
