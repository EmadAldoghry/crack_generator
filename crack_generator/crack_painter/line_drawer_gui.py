import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

class LineDrawer:
    def __init__(self, image_to_draw_on):
        self.image = image_to_draw_on
        self.fig, self.ax = plt.subplots(figsize=(10,10))
        self.ax.imshow(self.image, cmap='gray' if len(self.image.shape)==2 else None)
        self.title_artist = self.ax.set_title("Press 'D' to enter draw mode. Then click start & end points.\n'Enter' when done. 'C' to clear last. 'Esc' to cancel.")
        
        self.lines_coords = []
        self.permanent_line_artists = [] # Keep track of permanent lines for easy removal
        self.temp_line_artist = None
        self.first_click_point = None
        
        self.drawing_mode_active = False
        self._d_key_pressed_recently = False # Flag to handle single 'd' press

        # Connect all event handlers
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_key_press = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.cid_key_release = self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

        self.done_drawing = False

    def update_title(self, message):
        self.title_artist.set_text(message)
        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        if self.done_drawing: return
        
        key = event.key.lower()

        if key == 'd':
            if not self._d_key_pressed_recently:
                self._d_key_pressed_recently = True
                if not self.drawing_mode_active:
                    self.drawing_mode_active = True
                    self.first_click_point = None
                    self.update_title("Draw Mode: Click for line start point.\n'Esc' to exit draw mode.")
                    print("Draw mode activated. Click to set start point.")

        elif key == 'enter':
            if self.drawing_mode_active:
                self.drawing_mode_active = False
                self.first_click_point = None
                if self.temp_line_artist:
                    self.temp_line_artist.remove()
                    self.temp_line_artist = None
                print("Exited draw mode via 'Enter'. All drawn lines are kept. Press 'D' to re-enter.")
                self.update_title("Exited draw mode. Press 'D' to draw, or 'Enter' again to finish.")
            else:
                print("Finished drawing lines.")
                self.done_drawing = True
                plt.close(self.fig)

        elif key == 'escape':
            if self.drawing_mode_active:
                self.drawing_mode_active = False
                self.first_click_point = None
                if self.temp_line_artist:
                    self.temp_line_artist.remove()
                    self.temp_line_artist = None
                self.update_title("Draw mode cancelled. Press 'D' to draw. 'Enter' to finish.")
                print("Draw mode cancelled by Escape key.")

        elif key == 'c':
            if self.drawing_mode_active:
                if self.first_click_point:
                    self.first_click_point = None
                    if self.temp_line_artist:
                        self.temp_line_artist.remove()
                        self.temp_line_artist = None
                    self.update_title("Draw Mode: Start point cleared. Click for a new start point.")
                    print("Current line start point cleared.")
            elif self.lines_coords:
                self.lines_coords.pop()
                last_line_artist = self.permanent_line_artists.pop()
                last_line_artist.remove()
                self.update_title("Last line cleared. Press 'D' to draw. 'Enter' to finish.")
                print("Last confirmed line cleared.")
            self.fig.canvas.draw_idle()

    def on_key_release(self, event):
        key = event.key.lower()
        if key == 'd':
            self._d_key_pressed_recently = False

    def on_click(self, event):
        if event.inaxes != self.ax or self.done_drawing:
            return
        
        if not self.drawing_mode_active:
            print("Click ignored: Not in draw mode. Press 'D' to activate.")
            return

        if self.first_click_point is None:
            self.first_click_point = (event.xdata, event.ydata)
            self.temp_line_artist = Line2D([self.first_click_point[0]], [self.first_click_point[1]],
                                           color='red', linestyle='--')
            self.ax.add_line(self.temp_line_artist)
            self.update_title("Draw Mode: Click for line end point.")
            print(f"Line start set at: ({event.xdata:.2f}, {event.ydata:.2f})")
        else:
            end_point = (event.xdata, event.ydata)
            self.lines_coords.append((self.first_click_point, end_point))
            
            perm_line = Line2D([self.first_click_point[0], end_point[0]],
                               [self.first_click_point[1], end_point[1]],
                               color='blue', linewidth=1)
            
            # --- THIS IS THE FIX ---
            self.ax.add_line(perm_line)
            self.permanent_line_artists.append(perm_line)

            if self.temp_line_artist:
                self.temp_line_artist.remove()
                self.temp_line_artist = None
            
            self.first_click_point = None
            
            self.update_title(f"Line added. Click for next start point or 'Esc' to exit draw mode.")
            print(f"Line end set at: ({event.xdata:.2f}, {event.ydata:.2f}). Line confirmed.")
        
        self.fig.canvas.draw_idle()

    def on_motion(self, event):
        if not self.drawing_mode_active or event.inaxes != self.ax or self.first_click_point is None:
            return
        
        if self.temp_line_artist:
            self.temp_line_artist.set_data([self.first_click_point[0], event.xdata],
                                           [self.first_click_point[1], event.ydata])
            self.fig.canvas.draw_idle()

    def get_lines(self):
        try:
            plt.show() 
        except Exception as e:
            print(f"Matplotlib display error: {e}")
        return self.lines_coords