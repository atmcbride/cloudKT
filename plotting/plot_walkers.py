import matplotlib.pyplot as plt

def plot_walkers(chain, idx,  sightline = None,):
        fig, ax = plt.subplots()
        ax.plot(chain[:, :, idx], color = 'k', alpha = 0.4)
        ax.set_xlabel('Walker Steps')
        label = ""
        if sightline is not None:
            if idx < sightline.ndim:
                label = "Velocity (km/s)"
            else:
                 label = r"$\Delta A(V)"
        ax.set_ylabel(label)
        return fig, ax