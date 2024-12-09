from PySide6.QtWidgets import (
    QWidget,
    QMainWindow,
    QApplication,
    QStyle,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QLineEdit
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import (
    QMouseEvent,
    QPaintEvent,
    QPen,
    QImage,
    QPainter,
    QPixmap,
)
import numpy as np
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from service import Service

service = Service()

class MplCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_image(self, vector):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(vector.T, interpolation='nearest', cmap='gray', origin='upper')
        ax.axis('off')
        self.canvas.draw()

class PainterWidget(QWidget):
    """A widget where user can draw with their mouse

    The user draws on a QPixmap which is itself paint from paintEvent()

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFixedSize(512, 512)
        self.pixmap = QPixmap(self.size())  
        self.pixmap.fill(Qt.black)

        self.previous_pos = None
        self.painter = QPainter()
        self.pen = QPen()
        self.pen.setWidth(35)
        self.pen.setCapStyle(Qt.RoundCap)
        self.pen.setJoinStyle(Qt.RoundJoin)
        self.pen.setColor(Qt.white)

    def paintEvent(self, event: QPaintEvent):
        """Override method from QWidget

        Paint the Pixmap into the widget

        """
        with QPainter(self) as painter:
            painter.drawPixmap(0, 0, self.pixmap)

    def mousePressEvent(self, event: QMouseEvent):
        """Override from QWidget

        Called when user clicks on the mouse

        """

        current_pos = event.position().toPoint()
        self.painter.begin(self.pixmap)
        self.painter.setRenderHints(QPainter.RenderHint.Antialiasing, True)
        self.painter.setPen(self.pen)
        self.painter.drawPoint(current_pos)
        self.painter.end()
        self.update()

        self.previous_pos = current_pos
        QWidget.mousePressEvent(self, event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Override method from QWidget

        Called when user moves and clicks on the mouse

        """
        current_pos = event.position().toPoint()
        self.painter.begin(self.pixmap)
        self.painter.setRenderHints(QPainter.RenderHint.Antialiasing, True)
        self.painter.setPen(self.pen)
        self.painter.drawLine(self.previous_pos, current_pos)
        self.painter.end()
        self.update()

        self.previous_pos = current_pos

        QWidget.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Override method from QWidget

        Called when user releases the mouse

        """
        self.previous_pos = None
        QWidget.mouseReleaseEvent(self, event)

    def verify(self):
        # downscale pixmap and convert to image
        qImage = self.pixmap.scaled(28, 28, mode=Qt.SmoothTransformation).toImage().convertToFormat(QImage.Format_Grayscale8)
        # qImage = self.pixmap.scaled(28, 28, mode=Qt.FastTransformation).toImage().convertToFormat(QImage.Format_Grayscale8)

        # show newly created image
        # resized_pixmap = QPixmap(qImage)
        # resized_pixmap_qLabel = QLabel()
        # resized_pixmap_qLabel.setPixmap(resized_pixmap)
        # self.image_widget = QWidget()
        # image_widget_layout = QGridLayout()
        # image_widget_layout.addWidget(resized_pixmap_qLabel)
        # self.image_widget.setLayout(image_widget_layout)
        # self.image_widget.show()

        input_vector = np.zeros((28, 28))

        # create numpy array
        for y in range(qImage.height()):
            for x in range(qImage.width()):
                input_vector[x][y] = qImage.pixel(x, y) & 0xFF
                # if x == 0:
                #     print()
                # print(input_vector[x][y], end=' ')
                input_vector[x][y] /= 255  # normalize

        input_vector = input_vector.reshape(784, 1)

        self.plot_widget = MplCanvas()
        self.plot_widget.plot_image((input_vector * 255).reshape(28, 28))
        self.plot_widget.show()

        return service.get_prediction_confidence(input_vector)


    def clear(self):
        """ Clear the pixmap """
        self.pixmap.fill(Qt.black)
        self.update()


class MainWindow(QMainWindow):
    """An Application example to draw using a pen """

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)

        self.root = QWidget()
        self.service = Service()

        self.prediction = QLineEdit("7")
        self.prediction.setReadOnly(True)
        self.prediction.setAlignment(Qt.AlignCenter)
        self.confidence = QLineEdit("97 %")
        self.confidence.setReadOnly(True)
        self.confidence.setAlignment(Qt.AlignCenter)

        self.grid_layout = QGridLayout()
        self.grid_layout.addWidget(QLabel("Prediction"), 0, 0)
        self.grid_layout.addWidget(self.prediction, 0, 1)
        self.grid_layout.addWidget(QLabel("Confidence"), 1, 0)
        self.grid_layout.addWidget(self.confidence, 1, 1)

        self.results = QWidget()
        self.results.setLayout(self.grid_layout)

        self.painter_widget = PainterWidget(self.root)
        # self.root.setFixedHeight(self.painter_widget.height())
        # self.root.setFixedWidth(self.painter_widget.width() * 1.5)

        self.main_layout = QHBoxLayout()
        self.main_layout.addWidget(self.painter_widget)
        self.main_layout.addWidget(self.results)

        self.root.setLayout(self.main_layout)

        self.bar = self.addToolBar("Menu")
        self.bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        self._verify_action = self.bar.addAction(
            qApp.style().standardIcon(QStyle.SP_DialogSaveButton),  # noqa: F821
            "Verify", self.on_verify
        )

        self.bar.addAction(
            qApp.style().standardIcon(QStyle.SP_DialogResetButton),  # noqa: F821
            "Clear",
            self.painter_widget.clear,
        )
        self.bar.addSeparator()

        self.setCentralWidget(self.root)

    @Slot()
    def on_verify(self):
        prediction, confidence = self.painter_widget.verify()
        self.prediction.setText(str(prediction))
        self.confidence.setText(str(confidence))


if __name__ == "__main__":

    app = QApplication(sys.argv)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())
