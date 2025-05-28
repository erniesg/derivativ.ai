from manim import *
import numpy as np

class ScatterPlotScene(Scene):
    def construct(self):
        # 0. Configuration
        bg_color = WHITE
        fg_color = BLACK
        grid_color_minor = GREY_C
        grid_color_major = GREY_B
        point_color = BLACK
        new_point_color = RED_C
        self.camera.background_color = bg_color

        # --- Define scales and font sizes centrally ---
        q_number_fs = 38  # Slightly smaller to accommodate wider layout
        desc_text_fs = 22
        axis_label_fs_tex = 20
        axis_tick_numbers_fs = 18
        table_desc_text_fs = 18
        table_element_fs = 16
        q_i_text_fs = 20
        q_i_marks_fs = 20

        # This scale will be critical. We make content wider, then scale to fit.
        final_content_overall_scale = 0.6 # May need adjustment after making ax_width larger

        # 1. Title and descriptive text
        q_number = Text("15 (a)", color=fg_color, weight=BOLD, font="Sans", font_size=q_number_fs)
        desc1_text_str = "As part of a sports competition, 14 athletes run 100m and complete a swimming race."
        desc1 = Text(desc1_text_str, color=fg_color, line_spacing=0.9, font="Sans", font_size=desc_text_fs)
        desc2_text_str = "The scatter diagram shows the times, in seconds, to run 100m and the times, in seconds, to\ncomplete the swimming race, for 11 of these athletes."
        desc2 = Text(desc2_text_str, color=fg_color, line_spacing=0.9, font="Sans", font_size=desc_text_fs)
        header_text = VGroup(q_number, desc1, desc2).arrange(DOWN, aligned_edge=LEFT, buff=0.2)

        # 2. Axes and Grid Setup
        x_min, x_max, x_step_nums = 10.0, 11.4, 0.2
        y_min, y_max, y_step_nums = 23.0, 27.0, 1.0
        x_step_fine_grid = 0.02
        y_step_fine_grid = 0.1

        ax_width = 13.0 # SIGNIFICANTLY INCREASED plot width
        ax_height = 7.0 # Kept height, plot will be much wider aspect

        fine_grid_plane = NumberPlane(
            x_range=(x_min, x_max + x_step_fine_grid, x_step_fine_grid),
            y_range=(y_min, y_max + y_step_fine_grid, y_step_fine_grid),
            x_length=ax_width, y_length=ax_height,
            background_line_style={"stroke_color": grid_color_minor, "stroke_width": 0.4, "stroke_opacity": 0.5}
        )
        major_grid_plane = NumberPlane(
            x_range=(x_min, x_max + x_step_nums, x_step_nums),
            y_range=(y_min, y_max + y_step_nums, y_step_nums),
            x_length=ax_width, y_length=ax_height,
            background_line_style={"stroke_color": grid_color_major, "stroke_width": 0.8, "stroke_opacity": 0.7}
        )
        axes = Axes(
            x_range=(x_min, x_max + 0.001, x_step_nums), y_range=(y_min, y_max + 0.001, y_step_nums),
            x_length=ax_width, y_length=ax_height,
            axis_config={"color": fg_color, "stroke_width": 1.5, "include_tip": True, "tip_width": 0.12, "tip_length": 0.12}, # Smaller tips for wider axes
            x_axis_config={"numbers_to_include": np.arange(x_min, x_max + x_step_nums, x_step_nums), "font_size": axis_tick_numbers_fs,
                           "decimal_number_config": {"num_decimal_places": 1, "color": fg_color}},
            y_axis_config={"numbers_to_include": np.arange(y_min, y_max + y_step_nums, y_step_nums), "font_size": axis_tick_numbers_fs,
                           "decimal_number_config": {"num_decimal_places": 0, "color": fg_color}},
            tips=True
        )
        y_label = axes.get_y_axis_label(
            Tex(r"\textsf{Time to \\ complete the \\ swimming race \\ (s)}", font_size=axis_label_fs_tex, color=fg_color),
            edge=LEFT, direction=LEFT, buff=1.0 # Adjust as needed, was 1.1
        )
        x_label = axes.get_x_axis_label(
            Tex(r"\textsf{Time to run 100m (s)}", font_size=axis_label_fs_tex, color=fg_color),
            edge=DOWN, direction=DOWN, buff=0.5 # Was 0.6
        )
        _graph_elements_no_points = VGroup(fine_grid_plane, major_grid_plane, axes, x_label, y_label)

        # 3. Data Points
        existing_data = [(10.34, 24.1), (10.44, 24.4), (10.56, 24.5), (10.66, 24.1), (10.68, 24.9), (10.70, 24.0), (10.74, 24.4), (10.82, 25.0), (10.88, 25.1), (11.06, 24.6), (11.32, 26.0)]
        existing_points_mobjects = VGroup()
        for x_val, y_val in existing_data:
            dot = MathTex(r"\times", color=point_color).scale(0.9) # Slightly smaller x markers
            dot.move_to(axes.coords_to_point(x_val, y_val))
            existing_points_mobjects.add(dot)
        new_data_points_values = [(10.20, 23.5), (10.86, 25.4), (11.04, 24.9)]
        new_points_mobjects = VGroup()
        for x_val, y_val in new_data_points_values:
            dot = MathTex(r"\times", color=new_point_color).scale(0.9)
            dot.move_to(axes.coords_to_point(x_val, y_val))
            new_points_mobjects.add(dot)
        graph_group = VGroup(_graph_elements_no_points, existing_points_mobjects, new_points_mobjects)

        # 4. Table of new data
        table_desc_text = Text("The table shows the times for the other 3 athletes.", color=fg_color, font="Sans", font_size=table_desc_text_fs)
        table_data_list = [["Time to run 100m (s)", "10.20", "10.86", "11.04"], ["Time to complete the swimming race (s)", "23.5", "25.4", "24.9"]]
        table_manim = Table(
            table_data_list, include_outer_lines=True, line_config={"stroke_width": 1, "color": fg_color},
            element_to_mobject=Text, element_to_mobject_config={"font_size": table_element_fs, "color": fg_color, "font": "Sans"}
        )
        # Make table wider to match graph if possible.
        # table_manim.set_width(ax_width * 0.8) # Or some proportion of ax_width
        table_group = VGroup(table_desc_text, table_manim).arrange(DOWN, buff=0.15, aligned_edge=LEFT)

        # 5. Question (i) instruction
        q_i_text_str = "(i) On the scatter diagram, plot these three points."
        q_i_marks_str = "[2]"
        q_i_text = Text(q_i_text_str, color=fg_color, font="Sans", font_size=q_i_text_fs)
        q_i_marks_text = Text(q_i_marks_str, color=fg_color, font="Sans", font_size=q_i_marks_fs)
        # Align question text to the left of graph_group, and marks to the right.
        # This is tricky with the current arrange. Let's try to align q_i_text with table_group
        # and then place q_i_marks_text far to the right.

        q_i_full_text = VGroup(q_i_text, q_i_marks_text) # Will position manually after scaling

        # --- ASSEMBLE AND SCALE ALL CONTENT ---
        all_content = VGroup(
            header_text,
            graph_group,
            table_group,
            # q_i_full_text # Position this one after scaling all_content
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT) # Adjusted buff

        # Scale the main content
        all_content.scale(final_content_overall_scale)

        # Now add and position q_i_full_text relative to the scaled all_content
        q_i_text.scale(final_content_overall_scale) # Scale it too
        q_i_marks_text.scale(final_content_overall_scale)

        # Position q_i_text below the scaled table_group (which is inside all_content)
        # To do this, we need a reference from the scaled all_content
        scaled_table_group_bottom = all_content[2].get_bottom() # all_content[2] is table_group
        q_i_text.next_to(scaled_table_group_bottom, DOWN, buff=0.2 * final_content_overall_scale, aligned_edge=LEFT)

        # Position q_i_marks_text to the right end of the scene, aligned with q_i_text vertically
        q_i_marks_text.align_to(q_i_text, UP)
        q_i_marks_text.to_edge(RIGHT, buff=0.5 * final_content_overall_scale) # Use a buff relative to overall scale

        # Add q_i_text and q_i_marks_text to a final VGroup to be moved to ORIGIN
        final_layout = VGroup(all_content, q_i_text, q_i_marks_text)
        final_layout.move_to(ORIGIN)

        # --- ANIMATION (or static add) ---
        # self.add(final_layout) # For static render with -s flag

        self.play(Write(header_text)) # header_text is part of all_content, which is part of final_layout
        self.play(Create(fine_grid_plane), Create(major_grid_plane))
        self.play(Create(axes))
        self.play(Write(x_label), Write(y_label))
        self.play(LaggedStart(*[GrowFromCenter(point) for point in existing_points_mobjects], lag_ratio=0.1))
        self.play(Write(table_group)) # table_group is part of all_content
        self.play(Write(q_i_text), Write(q_i_marks_text)) # Animate the manually positioned q_i texts
        self.play(LaggedStart(*[GrowFromCenter(point) for point in new_points_mobjects], lag_ratio=0.2))

        self.wait(1)
