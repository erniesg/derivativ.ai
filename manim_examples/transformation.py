from manim import *
import numpy as np

class RecreateTransformationImage(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        fg_color = BLACK # Define a foreground color for clarity

        # Question Number
        q_num = Text("18", font_size=36, color=fg_color)
        q_num.to_corner(UL, buff=0.5)

        # Data ranges from the image
        x_min_data, x_max_data, x_step_data = -6, 8, 1
        y_min_data, y_max_data, y_step_data = -6, 7, 1

        # Screen dimensions for the axes
        x_axis_length_screen = 7.0
        y_axis_length_screen = x_axis_length_screen * ((y_max_data - y_min_data) / (x_max_data - x_min_data))

        # --- Explicitly define the numbers to be displayed on axes ---
        x_numbers_for_labels = np.arange(x_min_data, x_max_data + x_step_data, x_step_data)
        y_numbers_for_labels = np.arange(y_min_data, y_max_data + y_step_data, y_step_data)

        axis_tick_numbers_fs = 24 # Font size for the axis numbers

        axes = Axes(
            # x_range/y_range define the line extent and default ticks.
            x_range=[x_min_data, x_max_data, x_step_data],
            y_range=[y_min_data, y_max_data, y_step_data],
            x_length=x_axis_length_screen,
            y_length=y_axis_length_screen,
            axis_config={
                "color": fg_color,                 # Axis line and tip color
                "stroke_width": 2,
                "include_tip": True,            # Show arrow tips
                "tip_width": 0.2,
                "tip_height": 0.2,
                "include_ticks": True,          # Ensure ticks are drawn based on step in x_range/y_range
                "tick_size": 0.1,               # Default tick size is 0.1
                # "include_numbers" is not strictly needed here if using add_coordinates,
                # but doesn't hurt and can be a fallback.
                # Number styling will be more directly controlled by x_axis_config/y_axis_config
                # when using add_coordinates.
            },
            # --- Per-axis configuration, especially for number styling with add_coordinates ---
            x_axis_config={
                "font_size": axis_tick_numbers_fs,
                "decimal_number_config": {
                    "num_decimal_places": 0,
                    "color": fg_color,
                },
                # "line_to_number_buff": 0.25 # or specific value like SMALL_BUFF
            },
            y_axis_config={
                "font_size": axis_tick_numbers_fs,
                "decimal_number_config": {
                    "num_decimal_places": 0,
                    "color": fg_color,
                },
                "label_direction": LEFT,
                # "line_to_number_buff": 0.25
            }
        )

        # --- Explicitly add the number labels to the axes ---
        axes.add_coordinates(
            x_numbers_for_labels,
            y_numbers_for_labels
        )

        # Optional: Re-apply styles to ensure they take effect after add_coordinates
        # This can be useful for debugging or if add_coordinates doesn't fully respect initial configs
        if hasattr(axes.x_axis, 'numbers'):
            for number_mob in axes.x_axis.numbers:
                number_mob.set_color(fg_color).set_font_size(axis_tick_numbers_fs)
        if hasattr(axes.y_axis, 'numbers'):
            for number_mob in axes.y_axis.numbers:
                number_mob.set_color(fg_color).set_font_size(axis_tick_numbers_fs)

        # Axis labels "x" and "y"
        x_label = axes.get_x_axis_label("x", edge=RIGHT, direction=RIGHT, buff=0.2).set_color(fg_color)
        y_label = axes.get_y_axis_label("y", edge=UP, direction=UP, buff=0.2).set_color(fg_color)

        # Dashed Grid Lines
        grid_lines = VGroup()
        grid_line_style = {
            "stroke_color": fg_color, "stroke_width": 1, "dash_length": 0.05,
            "dashed_ratio": 0.5, "stroke_opacity": 0.5
        }
        for x_val in x_numbers_for_labels: # Align grid with the numbers being displayed
            grid_lines.add(DashedLine(axes.c2p(x_val, y_min_data), axes.c2p(x_val, y_max_data), **grid_line_style))
        for y_val in y_numbers_for_labels: # Align grid with the numbers being displayed
            grid_lines.add(DashedLine(axes.c2p(x_min_data, y_val), axes.c2p(x_max_data, y_val), **grid_line_style))

        # Shapes A and B
        shape_A_vertices_data = [(2,1), (3,1), (3,4), (2,3)]
        shape_B_vertices_data = [(-4,3), (-1,3), (-1,2), (-3,2)]
        shape_A_points = [axes.c2p(x,y) for x,y in shape_A_vertices_data]
        shape_B_points = [axes.c2p(x,y) for x,y in shape_B_vertices_data]
        shape_A = Polygon(*shape_A_points, stroke_color=fg_color, stroke_width=2, fill_color=LIGHT_GRAY, fill_opacity=0.75)
        shape_B = Polygon(*shape_B_points, stroke_color=fg_color, stroke_width=2, fill_opacity=0)

        label_A = Text("A", color=fg_color, font_size=24).move_to(axes.c2p(2.5, 2.0))
        label_B = Text("B", color=fg_color, font_size=24).move_to(axes.c2p(-2.5, 2.5))

        q_text_str = "(a) Describe fully the <span weight='bold'>single</span> transformation that maps shape A onto shape B."
        q_text = MarkupText(q_text_str, font_size=24, color=fg_color).to_edge(DOWN, buff=0.3)

        self.add(q_num, grid_lines, axes, x_label, y_label, shape_A, shape_B, label_A, label_B, q_text)
