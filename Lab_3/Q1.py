import numpy as np
from scipy.spatial.distance import mahalanobis


def compute_dist(poly1_inp, poly2_inp):
    poly1 = np.array(poly1_inp)
    poly2 = np.array(poly2_inp)

    centroid1 = np.mean(poly1, axis=0)
    centroid2 = np.mean(poly2, axis=0)

    combined = np.vstack((poly1, poly2))
    cov = np.cov(combined.T)

    try:
        inv_cov = np.linalg.inv(cov)
    except Exception as e:
        print(f"Error: {e}")

    dist = mahalanobis(centroid1, centroid2, inv_cov)

    return dist


def get_poly():
    print("Seperate x and y with a ' ' and different sets with a ','")
    inp = input("Enter the vertices of the polygon: ").strip().split(",")
    coords = [(x.split()) for x in inp]

    filtered = []
    for c in coords:
        if len(c) == 2:
            filtered.append(tuple(map(float, c)))
        else:
            print(f"Skipping {c} due to length mismatch")

    return filtered


p1 = get_poly()
p2 = get_poly()

print(f"Distance: {compute_dist(p1, p2)}")
