from manim import *
import numpy as np


class SineWaveScene(Scene):
    def construct(self):
        axes = Axes(x_range=[-3, 3], y_range=[-1, 1])
        graph = axes.plot(lambda x: np.sin(x), color=BLUE)
        self.play(Create(axes), Create(graph))
        self.wait()


class RotatingSquareScene(Scene):
    def construct(self):
        square = Square()
        self.play(Rotate(square))
