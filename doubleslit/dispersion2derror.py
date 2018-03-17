import numpy as np
import matplotlib.pyplot as plt
#plt.rc('text', usetex=True)
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

r2 = x**2 + y**2
r = np.sqrt(r2)

P = [np.real(np.conjugate(psit[i])*psit[i])*dx**2 for i in range(psit.shape[0])]

r2_mean = np.array([np.sum(r2*P[i]) for i in range(psit.shape[0])])
r_mean = np.array([np.sum(r*P[i]) for i in range(psit.shape[0])])

print(r_mean.min()**2, r2_mean.min())
plt.xlabel("time")
plt.xlim(t.min(), t.max())

#<r>:
plt.plot(t, np.abs(sigma(t)*np.sqrt(np.pi/2)-r_mean), label = "$\sigma(t)\sqrt{\pi/2} - <r>$")
#<r^2>:
plt.plot(t, np.abs(2*sigma(t)**2 - r2_mean), label = "$ 2\sigma (t) ^2-<r^2>$")

#sigma
plt.plot(t, np.abs(sigma(t)*np.sqrt(2-np.pi/2) - np.sqrt(r2_mean - r_mean**2)), label = "$ \sigma (t) \sqrt{2-\pi/2}-(\sqrt{<r^2>-<r>^2})$")

plt.legend()
plt.show()

fig.set_size_inches(7,7)
fig.savefig("errordispersio.png")
