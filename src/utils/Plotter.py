import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots()

    def plot_threads_single_method(self, est, color, label=None):
        label_added = False
        for e in est:
            if not label_added and label is not None:
                self.ax.plot(e, label=label, color=color)
                label_added = True
            else:
                self.ax.plot(e, color=color)
                
    def plot_convergence_band_single_method(self, est, delta, color, label=None):
        if len(est) == 0:
            raise ValueError("The estimates list is empty.")
        # delta = 97.5 for example
        upper_quantile = np.percentile(est, delta, axis=0)
        x_vals = np.arange(1, len(upper_quantile) + 1)
        self.ax.fill_between(x_vals, 
                             0, 
                             upper_quantile, 
                             color=color, 
                             alpha=0.2, 
                             label=label)

    def plot_single_line(self, val, label=None):
        plt.axhline(val, color='black', linewidth=0.5, label=label)

    def set_labels(self, title=None, xlabel=None, ylabel=None):
        if title:
            self.ax.set_title(title)
        if xlabel:
            self.ax.set_xlabel(xlabel)
        if ylabel:
            self.ax.set_ylabel(ylabel)

    def set_ylim(self, ylim):
        if ylim:
            self.ax.set_ylim(ylim)
    
    def set_xlim(self, xlim):
        if xlim:
            self.ax.set_xlim(xlim)

    def set_tick_labels(self, total_sample_used, step_size):
        num_ticks = 10  # Number of ticks to display
        tick_spacing = total_sample_used //step_size // num_ticks
        
        tick_positions = np.arange(0, total_sample_used //step_size+1, tick_spacing)
        
        tick_labels = [f'{(i * tick_spacing * step_size):.0e}' for i in range(len(tick_positions))]
        plt.xticks(tick_positions, tick_labels, rotation=45)

    def show_plot(self, legend=True):
        if legend:
            self.ax.legend()
        plt.show()

    def save_plot(self, save_path, legend=True):
        if legend:
            self.ax.legend()
        plt.savefig(save_path, bbox_inches='tight')

