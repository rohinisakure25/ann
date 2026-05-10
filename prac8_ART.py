import numpy as np
inputs = np.array([
    [1, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 0]
])
vigilance = 0.6

# List to store clusters
clusters = []
for pattern in inputs:

    matched = False

    for i, cluster in enumerate(clusters):
        similarity = np.sum(pattern == cluster) / len(pattern)
        if similarity >= vigilance:
            print(f"Pattern {pattern} belongs to Cluster {i+1}")

            # Update cluster
            clusters[i] = np.logical_and(cluster, pattern).astype(int)

            matched = True
            break

    # Create new cluster
    if not matched:

        clusters.append(pattern)
        print(f"Pattern {pattern} creates new Cluster {len(clusters)}")
        
print("\nFinal Clusters:")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {cluster}")