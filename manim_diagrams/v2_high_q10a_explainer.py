from manim import *
import numpy as np

class V2_High_Q10a_Explainer(Scene):
    def construct(self):
        title = Text("Q10 (a) - Find x and ∠AEC", font_size=30).to_edge(UP)
        self.play(Write(title))

        # Diagram Elements
        E_coord = np.array([-1, 1.0, 0])
        A_coord = np.array([-3, 1.0, 0])
        C_coord = np.array([0, -1.0, 0])
        CD_line_start_coord = np.array([-2.5, -1.0, 0]) # Point on CD left of C

        line_AB_obj = Line(A_coord, np.array([1,1.0,0]), color=WHITE)
        line_CD_segment = Line(CD_line_start_coord, np.array([2,-1.0,0]), color=WHITE)
        trans_EC_obj = Line(E_coord, C_coord, color=BLUE_C)

        dot_E = Dot(E_coord, color=YELLOW); dot_C = Dot(C_coord, color=YELLOW)
        label_A = Text("A", font_size=20).next_to(A_coord, UP, 0.1)
        label_E = Text("E", font_size=20).next_to(E_coord, UP, 0.1)
        label_C_diag = Text("C", font_size=20).next_to(C_coord, DOWN, 0.1) # Renamed
        label_D_marker = Text("D", font_size=20).next_to(line_CD_segment.get_end(), RIGHT, 0.1)
        label_C_marker_line = Text("C", font_size=20).next_to(line_CD_segment.get_start(), LEFT, 0.1)


        tick_template = Line(UP*0.1+LEFT*0.1, DOWN*0.1+RIGHT*0.1, stroke_width=2.5, color=GRAY)
        pm_AB = VGroup(tick_template.copy().move_to(line_AB_obj.point_from_proportion(0.45)), tick_template.copy().move_to(line_AB_obj.point_from_proportion(0.45)+RIGHT*0.1))
        pm_CD = VGroup(tick_template.copy().move_to(line_CD_segment.point_from_proportion(0.40)), tick_template.copy().move_to(line_CD_segment.point_from_proportion(0.40)+RIGHT*0.1)) # Adjusted

        diagram = VGroup(line_AB_obj, line_CD_segment, trans_EC_obj, dot_E, dot_C, pm_AB, pm_CD, label_A, label_E, label_C_diag, label_D_marker, label_C_marker_line)
        diagram.shift(RIGHT*1.5 + UP*0.5)
        self.play(Create(diagram))
        self.wait(1)

        # Part (a)(i): Find x
        q_ai_text = Text("(i) Find x", font_size=28).next_to(title, DOWN, buff=0.3, aligned_edge=LEFT).shift(LEFT*3)
        self.play(Write(q_ai_text))

        # Angle AEC = (2x + 20)°
        line_EA = Line(E_coord, A_coord)
        angle_AEC_arc = Angle(line_EA, trans_EC_obj, radius=0.4, other_angle=False, color=RED)
        label_AEC_val = MathTex("(2x+20)^\\circ", color=RED).scale(0.5)
        label_AEC_val.move_to(Angle(line_EA, trans_EC_obj, radius=0.6, other_angle=False).point_from_proportion(0.5))
        self.play(Create(angle_AEC_arc), Write(label_AEC_val))
        self.wait(0.5)

        # Angle ECD = (3x - 10)°
        line_CE = Line(C_coord, E_coord)
        line_CD_left = Line(C_coord, CD_line_start_coord)
        angle_ECD_arc = Angle(line_CE, line_CD_left, radius=0.4, color=ORANGE)
        label_ECD_val = MathTex("(3x-10)^\\circ", color=ORANGE).scale(0.5)
        label_ECD_val.move_to(Angle(line_CE, line_CD_left, radius=0.6).point_from_proportion(0.5))
        self.play(Create(angle_ECD_arc), Write(label_ECD_val))
        self.wait(1)

        reason_text = Text("AB || CD, so ∠AEC = ∠ECD (alternate angles).", font_size=22).next_to(q_ai_text, DOWN, buff=0.3, aligned_edge=LEFT)
        self.play(Write(reason_text))
        self.play(Indicate(angle_AEC_arc, color=RED), Indicate(angle_ECD_arc, color=ORANGE))
        self.wait(1)

        # Calculations for x
        calc_x_group = VGroup(
            MathTex("2x + 20 = 3x - 10", font_size=28),
            MathTex("20 + 10 = 3x - 2x", font_size=28),
            MathTex("30 = x \\implies x = 30", font_size=32)
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).next_to(reason_text, DOWN, buff=0.3)

        self.play(Write(calc_x_group[0]))
        self.wait(1)
        self.play(TransformMatchingTex(calc_x_group[0].copy(), calc_x_group[1], path_arc=PI/2))
        self.wait(1)
        self.play(TransformMatchingTex(calc_x_group[1].copy(), calc_x_group[2], path_arc=PI/2))
        self.wait(1)
        box_x = SurroundingRectangle(calc_x_group[2], buff=0.1, color=GREEN)
        self.play(Create(box_x))
        self.wait(1)

        # Part (a)(ii): Find ∠AEC
        q_aii_text = Text("(ii) Find ∠AEC", font_size=28).move_to(q_ai_text)
        self.play(FadeOut(reason_text), FadeOut(calc_x_group[0]), FadeOut(calc_x_group[1]), # Keep x=30
                  Transform(q_ai_text, q_aii_text))
        self.wait(0.5)

        # Calculations for ∠AEC
        calc_AEC_group = VGroup(
            MathTex("\\text{∠AEC} = 2x + 20", font_size=28),
            MathTex("\\text{∠AEC} = 2(30) + 20", font_size=28),
            MathTex("\\text{∠AEC} = 60 + 20", font_size=28),
            MathTex("\\text{∠AEC} = 80^\\circ", font_size=32)
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).next_to(q_aii_text, DOWN, buff=0.3)

        self.play(Write(calc_AEC_group[0]))
        self.wait(0.5)
        self.play(TransformMatchingTex(calc_AEC_group[0].copy(), calc_AEC_group[1], path_arc=PI/2))
        self.wait(1)
        self.play(TransformMatchingTex(calc_AEC_group[1].copy(), calc_AEC_group[2], path_arc=PI/2))
        self.wait(1)
        self.play(TransformMatchingTex(calc_AEC_group[2].copy(), calc_AEC_group[3], path_arc=PI/2))
        self.wait(1)
        box_AEC = SurroundingRectangle(calc_AEC_group[3], buff=0.1, color=GREEN)
        self.play(Create(box_AEC))
        self.wait(3)
