from manim import *
import numpy as np

class V1_Low_Q10a_Explainer(Scene):
    def construct(self):
        title = Text("Question 10 (a) - Find x", font_size=30).to_edge(UP)
        self.play(Write(title))

        E_coord = np.array([0, 1.0, 0])
        A_coord = np.array([-2.5, 1.0, 0])
        C_coord = np.array([0.8, -1.0, 0])
        CD_line_start_coord = np.array([-2.0, -1.0, 0]) # Point on CD to the left of C for angle ECD

        line_AB_obj = Line(A_coord, np.array([2.5, 1.0, 0]), color=WHITE)
        line_CD_segment = Line(CD_line_start_coord, np.array([2.5, -1.0, 0]), color=WHITE)
        trans_EC = Line(E_coord, C_coord, color=BLUE_C)

        dot_E = Dot(E_coord, color=YELLOW)
        dot_C = Dot(C_coord, color=YELLOW)

        label_A = Text("A", font_size=24).next_to(A_coord, UP, buff=0.1)
        label_E = Text("E", font_size=24).next_to(E_coord, UP, buff=0.1)
        label_C_text = Text("C", font_size=24).next_to(C_coord, DOWN, buff=0.1) # Renamed to avoid conflict
        label_D_marker = Text("D", font_size=24).next_to(line_CD_segment.get_end(), RIGHT, buff=0.1)
        label_C_marker_line_CD = Text("C", font_size=24).next_to(line_CD_segment.get_start(), LEFT, buff=0.1)


        tick_template = Line(UP*0.1+LEFT*0.1, DOWN*0.1+RIGHT*0.1, stroke_width=2.5, color=GRAY)
        pm_AB = VGroup(tick_template.copy().move_to(line_AB_obj.point_from_proportion(0.45)), tick_template.copy().move_to(line_AB_obj.point_from_proportion(0.45)+RIGHT*0.1))
        pm_CD = VGroup(tick_template.copy().move_to(line_CD_segment.point_from_proportion(0.40)), tick_template.copy().move_to(line_CD_segment.point_from_proportion(0.40)+RIGHT*0.1)) # Adjusted position

        diagram_elements = VGroup(line_AB_obj, line_CD_segment, trans_EC, dot_E, dot_C, pm_AB, pm_CD, label_A, label_E, label_C_text, label_D_marker, label_C_marker_line_CD)
        self.play(Create(diagram_elements))
        self.wait(1)

        # Helper lines for angles
        line_EA = Line(E_coord,A_coord)
        line_EC_from_E = Line(E_coord,C_coord) # same as trans_EC

        angle_AEC_arc = Angle(line_EA, line_EC_from_E, radius=0.5, other_angle=False, color=RED)
        label_AEC_val = MathTex("70^\\circ", color=RED).scale(0.7)
        label_AEC_val.move_to(Angle(line_EA, line_EC_from_E, radius=0.7, other_angle=False).point_from_proportion(0.5))
        self.play(Create(angle_AEC_arc), Write(label_AEC_val))
        self.wait(1)

        # For Angle ECD: Line from C to E, and Line from C along CD to the left
        line_CE = Line(C_coord, E_coord)
        line_CD_left = Line(C_coord, CD_line_start_coord)
        angle_ECD_arc = Angle(line_CE, line_CD_left, radius=0.5, color=ORANGE)
        label_ECD_val = MathTex("x", color=ORANGE).scale(0.7)
        label_ECD_val.move_to(Angle(line_CE, line_CD_left, radius=0.7).point_from_proportion(0.5))
        self.play(Create(angle_ECD_arc), Write(label_ECD_val))
        self.wait(1)

        reason_text_group = VGroup(
            Text("AB || CD, EC is a transversal.", font_size=24),
            Text("∠AEC and ∠ECD are alternate angles.", font_size=24),
            Text("Alternate angles are equal.", font_size=24, color=YELLOW)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(title, DOWN, buff=0.3).shift(LEFT*2.5) # Shift for space

        self.play(Write(reason_text_group[0]))
        self.wait(0.5)
        self.play(Write(reason_text_group[1]))
        self.wait(0.5)
        self.play(Write(reason_text_group[2]))
        self.play(Indicate(angle_AEC_arc, scale_factor=1.2, color=RED), Indicate(angle_ECD_arc, scale_factor=1.2, color=ORANGE))
        self.wait(1)

        solution_eq = MathTex("x = 70^\\circ", font_size=36).next_to(reason_text_group, DOWN, buff=0.5)
        self.play(Write(solution_eq))
        self.wait(1)

        final_answer_box = SurroundingRectangle(solution_eq, buff=0.1, color=GREEN)
        final_reason_text = Text("Reason: Alternate angles", font_size=24, color=GREEN).next_to(solution_eq, DOWN, buff=0.2)
        self.play(Create(final_answer_box), Write(final_reason_text))
        self.wait(3)
