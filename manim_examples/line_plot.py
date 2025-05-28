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
        q_number_fs = 40
        desc_text_fs = 24
        axis_label_fs_tex = 24
        axis_tick_numbers_fs = 20 # Increased from 18
        table_desc_text_fs = 22
        table_element_fs = 20
        q_i_text_fs = 22
        q_i_marks_fs = 22

        final_content_overall_scale = 0.60 # Start here, adjust as needed

        # 1. Title and descriptive text (assuming this part is correct from your last full script)
        q_number = Text("15 (a)", color=fg_color, weight=BOLD, font="Sans", font_size=q_number_fs)
        desc1_text_str = "As part of a sports competition, 14 athletes run 100m and complete a swimming race."
        desc1 = Text(desc1_text_str, color=fg_color, line_spacing=0.9, font="Sans", font_size=desc_text_fs)
        desc2_text_str = "The scatter diagram shows the times, in seconds, to run 100m and the times, in seconds, to\ncomplete the swimming race, for 11 of these athletes."
        desc2 = Text(desc2_text_str, color=fg_color, line_spacing=0.9, font="Sans", font_size=desc_text_fs)
        header_text = VGroup(q_number, desc1, desc2).arrange(DOWN, aligned_edge=LEFT, buff=0.25) # Adjusted buff

        # 2. Axes and Grid Setup
        x_min_val, x_max_val, x_step_val = 10.0, 11.4, 0.2
        y_min_val, y_max_val, y_step_val = 23.0, 27.0, 1.0

        x_axis_numbers_to_show = np.arange(x_min_val, x_max_val + x_step_val * 0.5, x_step_val)
        y_axis_numbers_to_show = np.arange(y_min_val, y_max_val + y_step_val * 0.5, y_step_val)

        x_step_fine_grid = 0.02
        y_step_fine_grid = 0.1

        ax_width_manim = 12.0
        ax_height_manim = 7.5

        fine_grid_plane = NumberPlane(
            x_range=(x_min_val, x_max_val + x_step_fine_grid, x_step_fine_grid),
            y_range=(y_min_val, y_max_val + y_step_fine_grid, y_step_fine_grid),
            x_length=ax_width_manim, y_length=ax_height_manim,
            background_line_style={"stroke_color": grid_color_minor, "stroke_width": 0.5, "stroke_opacity": 0.6}
        )
        major_grid_plane = NumberPlane(
            x_range=(x_min_val, x_max_val + x_step_val, x_step_val),
            y_range=(y_min_val, y_max_val + y_step_val, y_step_val),
            x_length=ax_width_manim, y_length=ax_height_manim,
            background_line_style={"stroke_color": grid_color_major, "stroke_width": 0.9, "stroke_opacity": 0.7}
        )

        axes = Axes(
            x_range=[x_min_val, x_max_val, x_step_val],
            y_range=[y_min_val, y_max_val, y_step_val],
            x_length=ax_width_manim,
            y_length=ax_height_manim,
            axis_config={ # This is the main config for both axes lines and general properties
                "color": fg_color,
                "stroke_width": 1.5,
                "include_tip": True,
                # "tip_shape": ArrowTriangleFilledTip, # Default tip is usually fine for Axes
                "tip_width": 0.15, # Using values from your original Axes
                "tip_height": 0.15, # Using values from your original Axes (or tip_length)
                "include_ticks": True, # General switch for ticks on NumberLine
                "tick_size": 0.08,     # Length of the ticks
            },
            x_axis_config={
                "numbers_to_include": x_axis_numbers_to_show,
                "numbers_with_elongated_ticks": x_axis_numbers_to_show, # Key for ticks at numbers
                "font_size": axis_tick_numbers_fs,
                "decimal_number_config": {"num_decimal_places": 1, "color": fg_color, "group_with_commas":False},
                "line_to_number_buff": MED_SMALL_BUFF,
            },
            y_axis_config={
                "numbers_to_include": y_axis_numbers_to_show,
                "numbers_with_elongated_ticks": y_axis_numbers_to_show, # Key for ticks at numbers
                "font_size": axis_tick_numbers_fs,
                "decimal_number_config": {"num_decimal_places": 0, "color": fg_color, "group_with_commas":False},
                "label_direction": LEFT,
                "line_to_number_buff": MED_SMALL_BUFF,
            },
            tips=True # This is redundant if include_tip is in axis_config, but harmless
        )

        y_label = axes.get_y_axis_label(
            Tex(r"\textsf{Time to \\ complete the \\ swimming race \\ (s)}", font_size=axis_label_fs_tex, color=fg_color),
            edge=LEFT, direction=LEFT, buff=1.0
        )
        x_label = axes.get_x_axis_label(
            Tex(r"\textsf{Time to run 100m (s)}", font_size=axis_label_fs_tex, color=fg_color),
            edge=DOWN, direction=DOWN, buff=0.6
        )
        _graph_elements_no_points = VGroup(fine_grid_plane, major_grid_plane, axes, x_label, y_label)

        # 3. Data Points (using plain \times for now)
        existing_data = [(10.34, 24.1), (10.44, 24.4), (10.56, 24.5), (10.66, 24.1), (10.68, 24.9), (10.70, 24.0), (10.74, 24.4), (10.82, 25.0), (10.88, 25.1), (11.06, 24.6), (11.32, 26.0)]
        existing_points_mobjects = VGroup()
        for x_val, y_val in existing_data:
            dot = MathTex(r"\times", color=point_color).scale(1.0)
            dot.move_to(axes.coords_to_point(x_val, y_val))
            existing_points_mobjects.add(dot)
        new_data_points_values = [(10.20, 23.5), (10.86, 25.4), (11.04, 24.9)]
        new_points_mobjects = VGroup()
        for x_val, y_val in new_data_points_values:
            dot = MathTex(r"\times", color=new_point_color).scale(1.0)
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
        table_group = VGroup(table_desc_text, table_manim).arrange(DOWN, buff=0.2, aligned_edge=LEFT)

        # 5. Question (i) instruction
        q_i_text_str = "(i) On the scatter diagram, plot these three points."
        q_i_marks_str = "[2]"
        q_i_text = Text(q_i_text_str, color=fg_color, font="Sans", font_size=q_i_text_fs)
        q_i_marks_text = Text(q_i_marks_str, color=fg_color, font="Sans", font_size=q_i_marks_fs)

        # --- ASSEMBLE AND SCALE ALL CONTENT ---
        all_content_unscaled = VGroup(
            header_text,
            graph_group,
            table_group,
            q_i_text
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT)

        q_i_marks_text_scaled = q_i_marks_text.copy().scale(final_content_overall_scale)
        all_content_unscaled.scale(final_content_overall_scale)

        scaled_q_i_text_ref = all_content_unscaled[-1]
        scaled_graph_ref = all_content_unscaled[1]

        q_i_marks_text_scaled.align_to(scaled_q_i_text_ref, UP)
        q_i_marks_text_scaled.align_to(scaled_graph_ref, RIGHT)
        q_i_marks_text_scaled.shift(RIGHT * 0.25)

        all_content_final = VGroup(all_content_unscaled, q_i_marks_text_scaled)
        all_content_final.move_to(ORIGIN)

        # --- ANIMATION ---
        self.play(Write(header_text))
        self.play(Create(fine_grid_plane), Create(major_grid_plane), Create(axes))
        self.play(Write(x_label), Write(y_label))
        self.play(LaggedStart(*[GrowFromCenter(point) for point in existing_points_mobjects], lag_ratio=0.1))
        self.play(Write(table_group))
        self.play(Write(all_content_unscaled[-1]))
        self.play(Write(q_i_marks_text_scaled))
        self.play(LaggedStart(*[GrowFromCenter(point) for point in new_points_mobjects], lag_ratio=0.2))

        self.wait(1)
