from manim import *
import numpy as np

class RecreateTransformationImage(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # Question Number
        q_num = Text("18", font_size=36, color=BLACK)
        q_num.to_corner(UL, buff=0.5)

        # Data ranges from the image
        x_min_data, x_max_data, x_step_data = -6, 8, 1
        y_min_data, y_max_data, y_step_data = -6, 7, 1

        # Screen dimensions for the axes
        x_axis_length_screen = 7.0
        y_axis_length_screen = x_axis_length_screen * ((y_max_data - y_min_data) / (x_max_data - x_min_data))

        # Define the exact numbers to be displayed on each axis
        # For x-axis: numbers from -6 to 8 (inclusive) with a step of 1
        x_numbers = np.arange(x_min_data, x_max_data + x_step_data, x_step_data)
        # For y-axis: numbers from -6 to 7 (inclusive) with a step of 1
        y_numbers = np.arange(y_min_data, y_max_data + y_step_data, y_step_data)

        axes = Axes(
            # x_range/y_range define the line extent and default ticks.
            # The step here also influences default number generation if numbers_to_include is not used.
            x_range=[x_min_data, x_max_data, x_step_data],
            y_range=[y_min_data, y_max_data, y_step_data],
            x_length=x_axis_length_screen,
            y_length=y_axis_length_screen,
            axis_config={
                "color": BLACK,                 # Axis line and tick color
                "stroke_width": 2,
                "include_tip": True,            # Show arrow tips
                "tip_width": 0.2,
                "tip_height": 0.2,

                # Number related configurations, applied to both axes by default
                "include_numbers": True,        # CRUCIAL: tells NumberLine to attempt to add numbers
                "font_size": 24,                # Font size for numbers on axes
                "line_to_number_buff": MED_SMALL_BUFF, # Default is 0.25. Controls distance from axis to number.
                "decimal_number_config": {      # Config passed to DecimalNumber mobjects for styling
                    "num_decimal_places": 0,    # Show integers
                    "color": BLACK,             # Number color
                    # IMPORTANT: font_size for numbers is taken from NumberLine's font_size (set above),
                    # not from here, to avoid conflicts.
                },
                "exclude_origin_tick": False,   # If 0 is a tick, ensure it's shown.
            },
            # Override default number generation by explicitly providing the numbers for each axis
            x_axis_config={
                "numbers_to_include": x_numbers,
                # Default label_direction for x-axis is DOWN, which is usually correct.
            },
            y_axis_config={
                "numbers_to_include": y_numbers,
                # Default label_direction for y-axis is LEFT, which is usually correct.
            }
        )

        # Axis labels "x" and "y"
        x_label = axes.get_x_axis_label("x", edge=RIGHT, direction=RIGHT, buff=0.2).set_color(BLACK)
        y_label = axes.get_y_axis_label("y", edge=UP, direction=UP, buff=0.2).set_color(BLACK)

        # Dashed Grid Lines
        grid_lines = VGroup()
        grid_line_style = {
            "stroke_color": BLACK,
            "stroke_width": 1,
            "dash_length": 0.05,
            "dashed_ratio": 0.5,
            "stroke_opacity": 0.5
        }

        # Generate grid lines aligned with the numbers that should be on the axes
        for x_val in x_numbers: # Iterate over the same numbers intended for x-axis labels
            if x_min_data <= x_val <= x_max_data: # Draw grid line only within axis physical range
                grid_lines.add(DashedLine(axes.c2p(x_val, y_min_data), axes.c2p(x_val, y_max_data), **grid_line_style))
        for y_val in y_numbers: # Iterate over the same numbers intended for y-axis labels
            if y_min_data <= y_val <= y_max_data: # Draw grid line only within axis physical range
                grid_lines.add(DashedLine(axes.c2p(x_min_data, y_val), axes.c2p(x_max_data, y_val), **grid_line_style))

        # Shapes A and B
        shape_A_vertices_data = [(2,1), (3,1), (3,4), (2,3)]
        shape_B_vertices_data = [(-4,3), (-1,3), (-1,2), (-3,2)]
        shape_A_points = [axes.c2p(x,y) for x,y in shape_A_vertices_data]
        shape_B_points = [axes.c2p(x,y) for x,y in shape_B_vertices_data]
        shape_A = Polygon(*shape_A_points, stroke_color=BLACK, stroke_width=2, fill_color=LIGHT_GRAY, fill_opacity=0.75)
        shape_B = Polygon(*shape_B_points, stroke_color=BLACK, stroke_width=2, fill_opacity=0)

        # Labels "A" and "B" inside the shapes
        label_A = Text("A", color=BLACK, font_size=24).move_to(axes.c2p(2.5, 2.0))
        label_B = Text("B", color=BLACK, font_size=24).move_to(axes.c2p(-2.5, 2.5))

        # Question Text at the bottom
        q_text_str = "(a) Describe fully the <span weight='bold'>single</span> transformation that maps shape A onto shape B."
        q_text = MarkupText(q_text_str, font_size=24, color=BLACK).to_edge(DOWN, buff=0.4)

        # Add all elements to the scene in desired layering order
        self.add(q_num)
        self.add(grid_lines)
        self.add(axes) # The Axes mobject itself contains the axis lines, ticks, and number labels
        self.add(x_label, y_label)
        self.add(shape_A, shape_B)
        self.add(label_A, label_B)
        self.add(q_text)
