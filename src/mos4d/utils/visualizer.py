# MIT License
#
# Copyright (c) 2024 Benedikt Mersch, Luca Lobefaro, Ignazio Vizzo, Tiziano Guadagnino
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import datetime
import importlib
import os
import numpy as np
from abc import ABC

# Button names
START_BUTTON = " START\n[SPACE]"
PAUSE_BUTTON = " PAUSE\n[SPACE]"
NEXT_FRAME_BUTTON = "NEXT FRAME\n\t\t [N]"
SCREENSHOT_BUTTON = "SCREENSHOT\n\t\t  [S]"
LOCAL_VIEW_BUTTON = "LOCAL VIEW\n\t\t [G]"
GLOBAL_VIEW_BUTTON = "GLOBAL VIEW\n\t\t  [G]"
CENTER_VIEWPOINT_BUTTON = "CENTER VIEWPOINT\n\t\t\t\t[C]"
QUIT_BUTTON = "QUIT\n  [Q]"

# Colors
BACKGROUND_COLOR = [0.0, 0.0, 0.0]
POINT_SIZE = 0.05


class StubVisualizer(ABC):
    def __init__(self):
        pass

    def update(self, points, labels):
        pass


class MOS4DVisualizer(StubVisualizer):
    # Public Interface ----------------------------------------------------------------------------
    def __init__(self):
        try:
            self._ps = importlib.import_module("polyscope")
            self._gui = self._ps.imgui
        except ModuleNotFoundError as err:
            print(f'polyscope is not installed on your system, run "pip install polyscope"')
            exit(1)

        # Initialize GUI controls
        self._background_color = BACKGROUND_COLOR
        self._point_size = POINT_SIZE
        self._block_execution = True
        self._play_mode = False

        # Initialize Visualizer
        self._initialize_visualizer()

    def update(self, points, labels):
        self._update_geometries(points, labels)
        while self._block_execution:
            self._ps.frame_tick()
            if self._play_mode:
                break
        self._block_execution = not self._block_execution

    # Private Interface ---------------------------------------------------------------------------
    def _initialize_visualizer(self):
        self._ps.set_program_name("MapMOS Visualizer")
        self._ps.init()
        self._ps.set_ground_plane_mode("none")
        self._ps.set_background_color(BACKGROUND_COLOR)
        self._ps.set_verbosity(0)
        self._ps.set_user_callback(self._main_gui_callback)
        self._ps.set_build_default_gui_panels(False)

    def _update_geometries(self, points, labels):
        steps = points[:, -1] - np.min(points[:, -1])
        colors = np.zeros((len(points), 3))
        colors = np.array([27, 45, 72]) + (steps / len(np.unique(steps))).reshape(-1, 1) @ np.array(
            [[104, 130, 168]]
        )
        colors /= 256
        colors[labels == 1] = [1, 0, 0]
        point_cloud = self._ps.register_point_cloud(
            "point_cloud",
            points[:, :3],
            point_render_mode="quad",
        )
        point_cloud.set_radius(self._point_size, relative=False)
        point_cloud.add_color_quantity("colors", colors, enabled=True)

    # GUI Callbacks ---------------------------------------------------------------------------
    def _start_pause_callback(self):
        button_name = PAUSE_BUTTON if self._play_mode else START_BUTTON
        if self._gui.Button(button_name) or self._gui.IsKeyPressed(self._gui.ImGuiKey_Space):
            self._play_mode = not self._play_mode

    def _next_frame_callback(self):
        if self._gui.Button(NEXT_FRAME_BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_N):
            self._block_execution = not self._block_execution

    def _screenshot_callback(self):
        if self._gui.Button(SCREENSHOT_BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_S):
            image_filename = "kisshot_" + (
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
            )
            self._ps.screenshot(image_filename)

    def _center_viewpoint_callback(self):
        if self._gui.Button(CENTER_VIEWPOINT_BUTTON) or self._gui.IsKeyPressed(
            self._gui.ImGuiKey_C
        ):
            self._ps.reset_camera_to_home_view()

    def _toggle_buttons_andslides_callback(self):
        changed, self._point_size = self._gui.SliderFloat(
            "##point_size", self._point_size, v_min=0.01, v_max=0.6
        )
        if changed:
            self._ps.get_point_cloud("point_cloud").set_radius(self._point_size, relative=False)

    def _background_color_callback(self):
        changed, self._background_color = self._gui.ColorEdit3(
            "Background Color",
            self._background_color,
        )
        if changed:
            self._ps.set_background_color(self._background_color)

    def _quit_callback(self):
        self._gui.SetCursorPosX(
            self._gui.GetCursorPosX() + self._gui.GetContentRegionAvail()[0] - 50
        )
        if (
            self._gui.Button(QUIT_BUTTON)
            or self._gui.IsKeyPressed(self._gui.ImGuiKey_Escape)
            or self._gui.IsKeyPressed(self._gui.ImGuiKey_Q)
        ):
            print("Destroying Visualizer")
            self._ps.unshow()
            os._exit(0)

    def _main_gui_callback(self):
        # GUI callbacks
        self._start_pause_callback()
        if not self._play_mode:
            self._gui.SameLine()
            self._next_frame_callback()
        self._gui.SameLine()
        self._screenshot_callback()
        self._gui.Separator()
        self._toggle_buttons_andslides_callback()
        self._background_color_callback()
        self._center_viewpoint_callback()
        self._gui.SameLine()
        self._quit_callback()
