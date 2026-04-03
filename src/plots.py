import matplotlib.pyplot as plt
import math

from typing import List, Union
from matplotlib.ticker import MaxNLocator


class Plots:
    def __init__(self):
        self.plots = []

    def add_plot(self, values: List[Union[int, float]], title: str = "", x_label: str = "", y_label: str = "") -> None:
        self.plots.append({
            'values': values,
            'title': title,
            'x_label': x_label,
            'y_label': y_label
        })

    def _draw_plots(self) -> None:
        num_plots = len(self.plots)

        if num_plots == 0:
            return

        cols = math.ceil(math.sqrt(num_plots))
        rows = math.ceil(num_plots / cols)

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axs = axs.flatten() if num_plots > 1 else [axs]

        i = -1
        for i, plot in enumerate(self.plots):
            axs[i].plot(plot['values'])
            axs[i].set_title(plot['title'])
            axs[i].set(xlabel=plot['x_label'], ylabel=plot['y_label'])
            axs[i].xaxis.set_major_locator(MaxNLocator(integer=True))

        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        fig.tight_layout()

    def show(self) -> None:
        self._draw_plots()
        plt.show()
