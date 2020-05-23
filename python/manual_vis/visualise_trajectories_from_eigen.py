import matplotlib.pyplot as plt

ROOM_H = 10
ROOM_W = 10

trajectory_pts = """
        1
        1
  3.99437
  1.80234
  6.98874
  2.60468
  9.98311
  3.40702
  7.02252
  4.20936
  4.02815
   5.0117
  1.03378
  5.81403
  1.96059
  6.61637
  4.95496
  7.41871
  7.94933
  8.22105
   9.0563
  9.02339
  6.06193
  9.82573
  3.06756
  9.37193
0.0731892
  8.56959
  2.92118
  7.76725
"""

# remove first and last empty line
trajectory_pts = trajectory_pts[1:-1].split("\n")
trajectory_pts = [float(pt.strip()) for pt in trajectory_pts]
x = [i for i in trajectory_pts[::2]]
y = [i for i in trajectory_pts[1::2]]

impact_pts ="""
     10
3.41154
      0
6.09103
     10
8.77053
5.41154
     10
      0
8.54998
"""

# remove first and last empty line
impact_pts = impact_pts[1:-1].split("\n")
impact_pts = [float(pt.strip()) for pt in impact_pts]
x_impact = [i for i in impact_pts[::2]]
y_impact = [i for i in impact_pts[1::2]]

for i, j in zip(x_impact, y_impact):
    plt.ylim([ROOM_H , 0])
    plt.xlim([0, ROOM_W])
    plt.scatter(i, j, c="black")
    plt.pause(1)

for i, j in zip(x, y):
    plt.ylim([ROOM_H , 0])
    plt.xlim([0, ROOM_W])
    plt.scatter(i, j, c="blue")
    plt.pause(0.8)

plt.show()