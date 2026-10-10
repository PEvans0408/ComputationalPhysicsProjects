import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#1D Burgers' Equation
#Physical Parameters
Lx = 1.0
T = 1.0
nu = 0.1  # viscosity coefficient

#Number of grid points and timesteps
Nx = 100
Nt = 2000

#Initial State
x = numpy.linspace(0, Lx, Nx + 1)  # Create our horizontal track coordinates
t = numpy.linspace(0, T, Nt + 1)

#Space and time steps
dx = x[1] - x[0]
#Solution array (x, t)
u = numpy.zeros((Nt + 1, Nx + 1))

# Create our initial condition; we have a Gaussian bump at t=0
#u[0, :] = 1.0 + numpy.exp(-((x - 0.5) / 0.25)**2)
u[0, :] = 5*numpy.sin(x*numpy.pi) + numpy.cos(x*numpy.pi)

#Diffusion Stability Condition
dt_diffusion = 0.25 * dx**2 / nu
#Advection CFL limiting Condition
umax = numpy.max(numpy.abs(u[0, :]))
if umax > 0:
    dt_advection = 0.5 * dx / umax
else:
    dt_advection = numpy.inf
#Choose the more restrictive condition
dt = min(dt_diffusion, dt_advection)

# Number of timesteps required to reach T
Nt = int(numpy.ceil(T / dt))
# Recalculate dt so that Nt steps lands exactly at T
dt = T / Nt
t = numpy.linspace(0, T, Nt + 1)

# Boundary Conditions @ the borders
# In this case both ends are kept at 1
u[:, 0] = 1
u[:, -1] = 1

plt.ion()  # turning interactive mode on
fig, ax = plt.subplots()
ax.set_xlim(0, Lx)
ax.set_ylim(numpy.min(u[0, :]), numpy.max(u[0, :]))
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.grid('both')
line, = ax.plot(x, u[0, :])

for n in range(Nt):
    for i in range(1, Nx):
        u[n+1, i] = (
            u[n, i] -
            dt * u[n, i] * ((u[n, i] - u[n, i-1]) / dx)
            + nu * dt * (
                (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / dx**2
            )
        )

    #Restore Boundary Conditions
    u[n+1, 0] = 1
    u[n+1, -1] = 1

    print(u[n+1].min(), u[n+1].max())
    #Update the plot with the new data
    if n % 10 == 0:   
        line.set_ydata(u[n+1, :])
        #Redraw the plot
        fig.canvas.draw()
        fig.canvas.flush_events()
        #Small pause so you can see it
        plt.pause(0.001)
plt.ioff()
plt.show()

#PINN for 1D Burgers' Equation goes here