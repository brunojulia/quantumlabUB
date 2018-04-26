import numpy as np
import matplotlib.pyplot as plt
plt.tight_layout()

ax = plt.gca()
ax.ticklabel_format(useOffset=False)
fig = plt.gcf()

def sigma(t, s = 0.5):
    return np.sqrt(s**2 + t**2/(4*s**2))

psit = np.load("psit2d.npy")
t = np.loadtxt("times2d.dat")
x = np.loadtxt("x2d.dat")
y = np.loadtxt("y2d.dat")

dx = y[1][0]-y[0][0]

N = [np.sum(np.real(np.conjugate(psit[i])*psit[i]))*dx*dx for i in range(psit.shape[0])]

with open("N.dat", "w") as outfile:
    for i in range(len(N)):
        print(N[i])
        outfile.write(str(t[i]) + "\t"+ str(N[i]) + "\n")

plt.xlabel("time")
plt.ylabel("$<\Psi | \Psi >$")
plt.xlim(t.min(), t.max())
#plt.ylim(0, 1)
plt.plot(t, N)
plt.show()

fig.set_size_inches(7,7)
fig.savefig("N2D.png")
