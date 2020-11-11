from matplotlib import pyplot as plt

heights = []
distances = []
with open('height-distance-webcam.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        height, distacne = tuple(map(int, line.split()))
        heights.append(height)
        distances.append(distacne)

plt.scatter(heights, [1/distacne for distacne in distances])
plt.savefig("TEMP.jpg")