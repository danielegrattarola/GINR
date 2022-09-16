import matplotlib.pyplot as plt


def scatter_eig(*us, axes):
    assert len(axes) == 2, "must have 2 axes"
    for i, u in enumerate(us):
        x, y = u[:, axes].T
        plt.scatter(x, y, marker=".", linewidth=0, label=i)
        plt.xlabel(f"u{axes[0]}")
        plt.ylabel(f"u{axes[1]}")
    plt.legend()


def hist_eig(us, axes):
    plt.hist(us[0][:, axes[0]], bins=100, alpha=0.5, density=True)
    plt.hist(us[1][:, axes[1]], bins=100, alpha=0.5, density=True)
