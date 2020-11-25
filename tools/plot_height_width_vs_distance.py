from matplotlib import pyplot as plt

heights = []
widths = []
distances = []
with open('height-width-distance-webcam.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        height, width, distacne = tuple(map(int, line.split()))
        heights.append(height)
        widths.append(width)
        distances.append(distacne)

plt.scatter(heights, [1/distacne for distacne in distances])
plt.scatter(widths, [1/distacne for distacne in distances])
plt.show()
plt.savefig("height-width-distance.jpg")