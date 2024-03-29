#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os

#################################################################
# NOTE: These two lines are VERY IMPORTANT -- they ensure qt
# uses its own path to the graphics plugins and not the cv2 path
import cv2
import numpy as np


# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
#################################################################

from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import (
    QAction,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    qApp,
)

from jumpstreet.utils import BaseClass, TimeMonitor


class Display(BaseClass):
    NAME = "display"

    def __init__(self, identifier, runnable=None, verbose=False, debug=False) -> None:
        super().__init__(self.NAME, identifier, verbose=verbose, debug=debug)
        self.runnable = runnable

    def start(self):
        """Start up display process"""
        raise NotImplementedError

    def update(self):
        """Update the display with image batch"""
        raise NotImplementedError


class ConfirmationDisplay(Display):
    def __init__(self, identifier=0, verbose=False, debug=False) -> None:
        super().__init__(identifier, main_loop=None, verbose=verbose, debug=debug)

    def start(self):
        self.print("started display process")

    def update(self, images):
        ks = images.keys()
        lvs = [len(imgs) for imgs in images.values()]
        if self.debug:
            self.print(
                f"received image batch with keys {ks}" + f"and value lens {lvs} images"
            )


class StreamThrough(Display):
    def __init__(
        self, main_loop, width=800, height=800, identifier=0, verbose=False, debug=False
    ) -> None:
        super().__init__(identifier, verbose=verbose, debug=debug)
        self.imageViewer = QImageViewer(width, height)
        self.thread = QThread()
        self.worker = main_loop
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.update.connect(self.update)
        self.thread.finished.connect(self.thread.deleteLater)

    def start(self):
        self.imageViewer.show()
        self.thread.start()

    def update(self, image_buffer):
        if self.debug:
            print("received image from worker")
        assert len(image_buffer) == 1, "For now only 1 image at a time"
        self.imageViewer.show_image_from_array(image_buffer[0])

    def resize(self, width, height):
        self.imageViewer.resize(width, height)


class QImageViewer(QMainWindow):
    def __init__(self, width: int=800, height: int=800):
        super().__init__()
        self.printer = QPrinter()
        self.scaleFactor = 0.0
        self.time_monitor = TimeMonitor()

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setVisible(False)

        self.setCentralWidget(self.scrollArea)

        self.createActions()
        self.createMenus()

        self.setWindowTitle("Image Viewer")
        self.resize(int(width), int(height))

    def open(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()",
            "",
            "Images (*.png *.jpeg *.jpg *.bmp *.gif)",
            options=options,
        )
        if fileName:
            image = QImage(fileName)
            if image.isNull():
                QMessageBox.information(
                    self, "Image Viewer", "Cannot load %s." % fileName
                )
                return
            self.show_image(image)

    def show_image_from_array(self, image, channel_order="bgr"):
        """Input is a numpy array"""
        h, w, _ = image.shape
        if channel_order == "bgr":
            try:
                img_form = QImage.Format_BGR888
            except AttributeError:
                img_form = QImage.Format_RGB888
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                channel_order = "rgb"
        else:
            raise NotImplementedError
        qimage = QImage(image.data, w, h, 3 * w, img_form)
        self.show_image(qimage)

    def show_image(self, image):
        self.imageLabel.setPixmap(QPixmap.fromImage(image))
        self.scaleFactor = 1.0

        self.scrollArea.setVisible(True)
        self.printAct.setEnabled(True)
        self.fitToWindowAct.setEnabled(True)
        self.updateActions()
        self.time_monitor.trigger()

        if not self.fitToWindowAct.isChecked():
            self.imageLabel.adjustSize()

    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel.pixmap())

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def about(self):
        QMessageBox.about(
            self,
            "About Image Viewer",
            "<p>The <b>Image Viewer</b> example shows how to combine "
            "QLabel and QScrollArea to display an image. QLabel is "
            "typically used for displaying text, but it can also display "
            "an image. QScrollArea provides a scrolling view around "
            "another widget. If the child widget exceeds the size of the "
            "frame, QScrollArea automatically provides scroll bars.</p>"
            "<p>The example demonstrates how QLabel's ability to scale "
            "its contents (QLabel.scaledContents), and QScrollArea's "
            "ability to automatically resize its contents "
            "(QScrollArea.widgetResizable), can be used to implement "
            "zooming and scaling features.</p>"
            "<p>In addition the example shows how to use QPainter to "
            "print an image.</p>",
        )

    def createActions(self):
        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O", triggered=self.open)
        self.printAct = QAction(
            "&Print...", self, shortcut="Ctrl+P", enabled=False, triggered=self.print_
        )
        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.zoomInAct = QAction(
            "Zoom &In (25%)",
            self,
            shortcut="Ctrl++",
            enabled=False,
            triggered=self.zoomIn,
        )
        self.zoomOutAct = QAction(
            "Zoom &Out (25%)",
            self,
            shortcut="Ctrl+-",
            enabled=False,
            triggered=self.zoomOut,
        )
        self.normalSizeAct = QAction(
            "&Normal Size",
            self,
            shortcut="Ctrl+S",
            enabled=False,
            triggered=self.normalSize,
        )
        self.fitToWindowAct = QAction(
            "&Fit to Window",
            self,
            enabled=False,
            checkable=True,
            shortcut="Ctrl+F",
            triggered=self.fitToWindow,
        )
        self.aboutAct = QAction("&About", self, triggered=self.about)
        self.aboutQtAct = QAction("About &Qt", self, triggered=qApp.aboutQt)

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)
        # self.zoomInAct.setEnabled(True) # new, doesn't seem to work
        # self.zoomOutAct.setEnabled(True) # new, doesn't seem to work

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(
            int(factor * scrollBar.value() + ((factor - 1) * scrollBar.pageStep() / 2))
        )


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    imageViewer = QImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())
