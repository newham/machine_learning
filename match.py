from matplotlib import pyplot as plt
x_values = []
y_values = []
start = -100
end = 100
x = start
step = 1
while x >= start and x <= end:
    if x == 0:
        x = x+step
        continue

    y = x**3-2*x**-4
    y_values.append(y)
    x = x+step
    x_values.append(x)

plt.scatter(x_values, y_values, s=0.5)
plt.show()
