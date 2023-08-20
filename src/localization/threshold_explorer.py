import typing as tp
from tkinter import IntVar, Label, Scale, StringVar, Tk

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from metrics import compute_precision_recall, create_precision_recall_fig


class ThresholdExplorerApp(Tk):
    def __init__(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        scores: np.ndarray,
        iou_thresholds: tp.Sequence[float],
    ):
        # Initialize main window
        super().__init__()
        self.title("Threshold explorer")
        self.geometry("700x500")

        # Data to plot
        self.precision = precision
        self.recall = recall
        self.scores = scores
        self.iou_thresholds = iou_thresholds

        # Initialize scale
        self.idx = IntVar(value=0)
        self.scale = Scale(
            self,
            orient="vertical",
            length=200,
            from_=0,
            to=len(scores) - 1,
            variable=self.idx,
            command=self._on_update_thr,
            resolution=1,
            showvalue=False,
        )
        self.scale.grid(column=2, row=0)

        # Initialize threshold label
        self.thr_text = StringVar(value=f"Threshold: {self.scores[self.idx.get()]:.3f}")
        self.thr_label = Label(self, textvariable=self.thr_text)
        self.thr_label.grid(column=1, row=0)

        # Initialize and plot precision-recall curve
        self.fig, self.marked_pr_points = create_precision_recall_fig(
            precision,
            recall,
            iou_thresholds,
            idx=self.idx.get(),
        )
        self._display_fig()

        # To handle closing of the window
        self.protocol("WM_DELETE_WINDOW", self._quit)

    def _quit(self):
        self.quit()
        self.destroy()

    def _display_fig(self):
        canvas = FigureCanvasTkAgg(self.fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().grid(column=0, row=0)

    def _on_update_thr(self, value):
        for i, pr_point in enumerate(self.marked_pr_points):
            pr_point.set_xdata([self.recall[i, self.idx.get()]])
            pr_point.set_ydata([self.precision[i, self.idx.get()]])

        plt.draw()
        self._display_fig()
        self.thr_text.set(f"Threshold: {self.scores[self.idx.get()]:.3f}")


@click.command(context_settings={"show_default": True})
@click.option("--gt_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option(
    "--preds_path", type=click.Path(exists=True, dir_okay=False), required=True
)
@click.option(
    "-t",
    "--iou_threshold",
    "iou_thresholds",
    type=click.FloatRange(min=0, max=1),
    multiple=True,
    default=[0.5, 0.75],
)
def main(gt_path: str, preds_path: str, iou_thresholds: tp.Sequence[float]):
    precision, recall, scores = compute_precision_recall(
        gt_path, preds_path, similarity_thresholds=iou_thresholds, return_scores=True
    )
    app = ThresholdExplorerApp(precision, recall, scores, iou_thresholds)
    app.mainloop()
    return


if __name__ == "__main__":
    main()
