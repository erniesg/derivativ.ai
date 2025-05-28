from manim import *
import numpy as np

class FullPlotRecreationFinalTouch(Scene): # Renamed
    def construct(self):
        self.camera.background_color = WHITE

        x_min, x_max, x_step_major = -2, 4, 1
        y_min, y_max, y_step_major = -6, 6, 1
        x_step_minor, y_step_minor = 0.2, 0.2

        plane_y_length = 6.0
        plane_x_length = plane_y_length * ((x_max - x_min) / (y_max - y_min))

        axis_numbers_font_size = 20

        plane = NumberPlane(
            x_range=(x_min, x_max, x_step_major),
            y_range=(y_min, y_max, y_step_major),
            x_length=plane_x_length,
            y_length=plane_y_length,
            axis_config={
                "stroke_color": BLACK,
                "include_tip": True,
                "tip_shape": ArrowTriangleFilledTip,
                "tip_width": 0.12,
                "tip_height": 0.12,
                "include_ticks": True,
                "tick_size": 0.08,
            },
            x_axis_config={
                "numbers_to_include": np.arange(x_min, x_max + x_step_major, x_step_major),
                "font_size": axis_numbers_font_size,
                "decimal_number_config": {"num_decimal_places": 0}
            },
            y_axis_config={
                "numbers_to_include": np.arange(y_min, y_max + y_step_major, y_step_major),
                "font_size": axis_numbers_font_size,
                "decimal_number_config": {"num_decimal_places": 0},
                "label_direction": LEFT,
            },
            background_line_style={
                "stroke_color": GRAY,
                "stroke_width": 1.0,
            }
        )

        plane.add_coordinates()

        number_buff = SMALL_BUFF * 0.7

        if hasattr(plane.x_axis, 'numbers') and plane.x_axis.numbers:
            for number_mob in plane.x_axis.numbers:
                number_mob.set_color(BLACK)
                tick = plane.x_axis.get_tick(number_mob.get_value())
                number_mob.next_to(tick, DOWN, buff=number_buff)

        if hasattr(plane.y_axis, 'numbers') and plane.y_axis.numbers:
            for number_mob in plane.y_axis.numbers:
                number_mob.set_color(BLACK)
                tick = plane.y_axis.get_tick(number_mob.get_value())
                number_mob.next_to(tick, LEFT, buff=number_buff)
                if number_mob.get_value() == 0:
                     number_mob.set_opacity(0)


        minor_lines = VGroup()
        for x_val_minor in np.arange(x_min + x_step_minor, x_max, x_step_minor):
            if not np.isclose(x_val_minor % x_step_major, 0.0) and not np.isclose(x_val_minor % x_step_major, x_step_major):
                minor_lines.add(DashedLine(
                    plane.c2p(x_val_minor, y_min), plane.c2p(x_val_minor, y_max),
                    stroke_width=0.5, color=LIGHT_GRAY, dash_length=0.03, dashed_ratio=0.5
                ))
        for y_val_minor in np.arange(y_min + y_step_minor, y_max, y_step_minor):
            if not np.isclose(y_val_minor % y_step_major, 0.0) and not np.isclose(y_val_minor % y_step_major, y_step_major):
                minor_lines.add(DashedLine(
                    plane.c2p(x_min, y_val_minor), plane.c2p(x_max, y_val_minor),
                    stroke_width=0.5, color=LIGHT_GRAY, dash_length=0.03, dashed_ratio=0.5
                ))

        line_L_obj = plane.plot(
            lambda x: 0.5 * x - 1,
            x_range=[x_min - 0.2, x_max + 0.2],
            color=BLACK,
            stroke_width=2.5
        )

        x_axis_label_obj = plane.get_x_axis_label(Tex("x", font_size=28, color=BLACK))
        y_axis_label_obj = plane.get_y_axis_label(Tex("y", font_size=28, color=BLACK))
        x_axis_label_obj.next_to(plane.get_x_axis().get_tip(), RIGHT, buff=0.05)
        y_axis_label_obj.next_to(plane.get_y_axis().get_tip(), UP, buff=0.05)
        axis_labels_group = VGroup(x_axis_label_obj, y_axis_label_obj)

        label_L_text = MathTex("L", font_size=28, color=BLACK).move_to(plane.c2p(3.25, 1.15))

        title_16 = Text("16", font_size=24, color=BLACK)
        title_main = Text("The line L is shown on the grid.", font_size=24, color=BLACK)
        top_title_group = VGroup(title_16, title_main).arrange(RIGHT, buff=0.15)
        top_title_group.to_edge(UP, buff=0.3) # Changed from 0.8

        bottom_question_text = Text("(a) Find the equation of line L in the form ", font_size=24, color=BLACK)
        bottom_question_formula = MathTex("y = mx + c.", font_size=24, color=BLACK)
        bottom_question_group = VGroup(bottom_question_text, bottom_question_formula).arrange(RIGHT, buff=0.05)
        bottom_question_group.to_edge(DOWN, buff=0.3) # Changed from 0.8

        main_plot_elements = VGroup(minor_lines, plane, line_L_obj, label_L_text, axis_labels_group)

        self.add(main_plot_elements)
        self.add(top_title_group)
        self.add(bottom_question_group)
