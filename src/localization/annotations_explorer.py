#!/usr/bin/env python3
"""
A simple GUI application to visualize predicted bounding boxes on images.
Use the left and right arrow keys to move from one image to the other.

Usage:
    annotations_explorer.py [OPTIONS]

Options:
--annotations_path: path to the json file with the annotation data. Both ground truths
    (without scores) and predictions (with scores) are accepted.
--img_dir, optional: path to the directory with the images. If not given, images are
    expected to be in the same directory of the json file.
"""

import json
import os
import typing as tp
from tkinter import DoubleVar, Entry, Label, Scale, StringVar, Tk, ttk

import click
from PIL import Image, ImageDraw, ImageTk


class AnnotationsExplorer:
    def __init__(self, annotations_path: str, img_dir: tp.Optional[str]):
        # Load data and initialize
        if os.path.splitext(annotations_path)[1] != ".json":
            raise Exception("Given file is not json.")
        if img_dir is None:
            img_dir = os.path.dirname(annotations_path)
        with open(annotations_path) as f:
            annotations = json.load(f)

        self.img_dir = img_dir
        self.annotations = annotations
        self.annotated_img_names = list(annotations.keys())
        self.is_ground_truth = len(annotations[self.annotated_img_names[0]]) == 2
        self.nb_images = len(self.annotated_img_names)
        self.idx = 0

        # Main window
        self.root = Tk()
        self.root.title(f"Annotations explorer from file: {annotations_path}")
        self.root.geometry("1150x580")

        # Frame inside main window
        mainframe = ttk.Frame(self.root, padding="3")
        mainframe.grid(column=0, row=0)

        if self.is_ground_truth is False:
            self.score_thr = DoubleVar(value=0.5)

            # Label widget for score threshold
            score_thr_label = Label(mainframe, text="Score threshold")
            score_thr_label.grid(column=0, row=0)

            # Scale widget for score threshold
            scale = Scale(
                mainframe,
                orient="vertical",
                length=200,
                from_=0.0,
                to=1.0,
                variable=self.score_thr,
                command=self._display_image_wrapper,
                resolution=0.01,
            )
            scale.grid(column=0, row=1, sticky="n")

        # Label widget for image counter
        self.img_counter = Label(mainframe, text="")
        self.img_counter.grid(column=3, row=2, sticky="e")

        # Increment label and entry
        Label(mainframe, text="Increment=").grid(column=1, row=2, sticky="e")
        self.increment = StringVar(value="1")
        Entry(mainframe, textvariable=self.increment, width=10).grid(
            column=2, row=2, sticky="w"
        )

        # Label widget for image (and image title)
        self.img_label = Label(mainframe, compound="bottom")
        self.img_label.grid(column=1, row=0, columnspan=3, rowspan=2)
        self._display_image()

        # Bind left and right arrows to command
        self.root.bind("<Left>", self._swipe)
        self.root.bind("<Right>", self._swipe)

        # Handle closing of the window
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    def _quit(self):
        self.root.quit()
        self.root.destroy()

    def _display_image(self):
        """Displays the image at the current idx (and its bounding boxes)"""
        img = Image.open(os.path.join(self.img_dir, self.annotated_img_names[self.idx]))
        draw = ImageDraw.Draw(img)
        if self.is_ground_truth:
            for bbox in self.annotations[self.annotated_img_names[self.idx]][0]:
                draw.rectangle(bbox, outline="red", width=3)
        else:
            for bbox, score in zip(
                self.annotations[self.annotated_img_names[self.idx]][0],
                self.annotations[self.annotated_img_names[self.idx]][2],
            ):
                if score >= self.score_thr.get():
                    draw.rectangle(bbox, outline="red", width=3)

        img = ImageTk.PhotoImage(img.resize((1000, 530), Image.Resampling.LANCZOS))
        self.img_label.configure(image=img, text=self.annotated_img_names[self.idx])
        self.img_label.image = img
        self.img_counter.configure(text=f"{self.idx+1}/{self.nb_images}")

    def _display_image_wrapper(self, value):
        self._display_image()

    def _swipe(self, e):
        """Swipe to the next image, and display it"""

        if self.increment.get().isdigit():
            delta = int(self.increment.get())
        else:
            delta = 1

        if e.keysym == "Left":
            self.idx = (self.idx - delta) % self.nb_images
        elif e.keysym == "Right":
            self.idx = (self.idx + delta) % self.nb_images
        self._display_image()

    def mainloop(self):
        self.root.mainloop()


@click.command()
@click.option(
    "--annotations_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--img_dir",
    type=click.Path(exists=True, file_okay=False),
    required=False,
    default=None,
)
def main(annotations_path: str, img_dir: tp.Optional[str]):
    app = AnnotationsExplorer(annotations_path, img_dir)
    app.mainloop()

    return


if __name__ == "__main__":
    main()
