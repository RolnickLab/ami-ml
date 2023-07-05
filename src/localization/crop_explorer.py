#!/usr/bin/env python3
"""
A simple GUI application to visualize segmented moths, and discard bad segmentations
from the dataset.
Use the left and right arrow keys to move from one image to the other.
Use the 'o' key to activate/deactivate the overlay.
Use the delete or backspace key to remove bad segmentations from the dataset.

Usage:
    crop_explorer.py [OPTIONS]

Options:
--crops_path: path to the .npz file with the cropped moths and their mask.
"""

import os
import re
from tkinter import *
from tkinter import filedialog, messagebox

import click
import cv2
import numpy as np
from PIL import Image, ImageTk


class CropsExplorer:
    def __init__(self, crops_path: str):
        # Load data and initialize
        self.crops_path = crops_path
        if os.path.splitext(crops_path)[1] != ".npz":
            raise Exception("Given file is not npz.")
        with np.load(crops_path) as data:
            self.data = dict(data)
        self.crop_keys = [s for s in list(self.data.keys()) if re.search(r"_crop_", s)]
        self.nb_images = len(self.crop_keys)
        self.idx = 0
        self.display_overlay = True
        self.overlay_color = (255, 105, 180)  # RGB pink

        # Main window
        self.root = Tk()
        self.root.title(f"Crops explorer from file: {crops_path}")
        self.root.geometry("700x500")

        # Label widget for image counter_label
        self.counter_label = Label(self.root, text="")

        # Scale widget for overlay
        self.alpha = DoubleVar(value=0.5)
        overlay_scale = Scale(
            self.root,
            orient="vertical",
            length=200,
            from_=0.0,
            to=1.0,
            variable=self.alpha,
            command=self._display_image_wrapper,
            resolution=0.1,
        )
        overlay_label = Label(self.root, text="Overlay:")

        # Buttons for save and discard actions
        self.save_button = Button(
            self.root, text="Save as", command=self._save_changes, state=DISABLED
        )
        self.discard_button = Button(
            self.root, text="Discard", command=self._discard_changes, state=DISABLED
        )

        # Label widgets for image and image title
        self.img_title_label = Label(self.root)
        self.img_label = Label(self.root)
        self._display_image()

        # Monitor changes from last save/discard
        self.changes_occured = BooleanVar(value=False)
        self.changes_occured.trace("w", self._update_buttons_state)

        # Bindings to keyboard events
        self.root.bind("<Left>", self._swipe)
        self.root.bind("<Right>", self._swipe)
        self.root.bind("o", self._overlay_switch)
        self.root.bind("<Delete>", self._remove_from_dataset)
        self.root.bind("<BackSpace>", self._remove_from_dataset)

        # Widgets arrangement and resize handling
        self.img_title_label.grid(column=1, row=0, pady=5)
        self.img_label.grid(column=1, row=1, rowspan=2, sticky=(N, S, E, W))
        self.save_button.grid(column=0, row=1, sticky=(S, E, W), padx=5, pady=5)
        self.discard_button.grid(column=0, row=2, sticky=(N, E, W), padx=5, pady=5)
        overlay_label.grid(column=2, row=1, sticky=(S, E, W), padx=5, pady=5)
        overlay_scale.grid(column=2, row=2, sticky=(N, E, W), padx=5)
        self.counter_label.grid(column=1, row=3, pady=5)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)

        # Window closure handling
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    def _quit(self):
        if self.changes_occured.get() and messagebox.askyesno(
            message="Save changes before leaving?"
        ):
            self._save_changes()

        self.root.quit()
        self.root.destroy()

    def _display_image(self):
        """Displays the image at the current idx"""
        img = self.data[self.crop_keys[self.idx]]
        if self.display_overlay:
            mask = self.data[re.sub(r"crop", r"mask", self.crop_keys[self.idx])]
            overlay = (
                np.full_like(img, self.overlay_color, dtype=np.uint8)
                * mask[:, :, np.newaxis]
            )
            img = cv2.addWeighted(img, 1, overlay, self.alpha.get(), 0)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        self.img_label.configure(image=img)
        self.img_label.image = img
        self.img_title_label.configure(text=self.crop_keys[self.idx])
        self.counter_label.configure(text=f"{self.idx+1}/{self.nb_images}")

    def _display_image_wrapper(self, value):
        self._display_image()

    def _remove_from_dataset(self, value):
        del self.data[re.sub(r"crop", r"mask", self.crop_keys[self.idx])]
        del self.data[self.crop_keys[self.idx]]
        del self.crop_keys[self.idx]
        if self.idx == self.nb_images:
            self.idx = 0
        self.nb_images = len(self.crop_keys)
        self.changes_occured.set(True)
        self._display_image()

    def _swipe(self, e):
        """Swipe to the next image, and display it"""

        if e.keysym == "Left":
            self.idx = (self.idx - 1) % self.nb_images
        elif e.keysym == "Right":
            self.idx = (self.idx + 1) % self.nb_images
        self._display_image()

    def _overlay_switch(self, value):
        self.display_overlay = not self.display_overlay
        self._display_image()

    def _save_changes(self):
        file = filedialog.asksaveasfile(
            mode="wb",
            defaultextension=".npz",
            initialdir=os.path.dirname(self.crops_path),
        )
        if file is not None:
            np.savez(file, **self.data)
            file.close()
            self.changes_occured.set(False)

    def _discard_changes(self):
        # Start over, re-load data from .npz file
        with np.load(self.crops_path) as data:
            self.data = dict(data)
        self.crop_keys = [s for s in list(self.data.keys()) if re.search(r"_crop_", s)]
        self.nb_images = len(self.crop_keys)
        self.idx = 0
        self.changes_occured.set(False)
        self._display_image()

    def _update_buttons_state(self, *args):
        if self.changes_occured.get():
            self.save_button.config(state=ACTIVE)
            self.discard_button.config(state=ACTIVE)
        else:
            self.save_button.config(state=DISABLED)
            self.discard_button.config(state=DISABLED)

    def mainloop(self):
        self.root.mainloop()


@click.command()
@click.option(
    "--crops_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
def main(crops_path: str):
    app = CropsExplorer(crops_path)
    app.mainloop()
    return


if __name__ == "__main__":
    main()
