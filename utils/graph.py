import matplotlib.pyplot as plt

def draw_loss(it_loss, filepath):

    for i in range(len(it_loss)):
        plt.plot(i, it_loss[i].detach().numpy(), marker='o', color='red', linestyle='-', linewidth=2, markersize=1)     

    plt.grid(b=True) # Add grid
    plt.savefig(filepath)
    plt.show()