from manim import *
import numpy as np

class V2_High_Q10b_Explainer(Scene):
    def construct(self):
        title = Text("Question 10 (b) - Find 'a'", font_size=30).to_edge(UP)
        self.play(Write(title))

        # Diagram setup
        E_coord = np.array([-0.5, 1.0, 0])
        A_coord = np.array([-2.5, 1.0, 0])
        C_coord = np.array([0.2, -1.0, 0])
        G_coord = np.array([1.5, -1.0, 0])
        CD_line_start_coord = np.array([-2.0, -1.0, 0])
        CD_line_end_coord = np.array([2.5, -1.0, 0])

        line_AB_obj = Line(A_coord, np.array([1.5,1.0,0]), color=WHITE)
        line_CD_obj = Line(CD_line_start_coord, CD_line_end_coord, color=WHITE)
        trans_EC_obj = Line(E_coord, C_coord, color=BLUE_A)
        trans_EG_obj = Line(E_coord, G_coord, color=GREEN_A)

        dot_E = Dot(E_coord); dot_C = Dot(C_coord); dot_G = Dot(G_coord)
        label_E = Text("E", font_size=20).next_to(E_coord, UP, 0.1)
        label_C_diag = Text("C", font_size=20).next_to(C_coord, DOWN, 0.1) # Renamed
        label_G_diag = Text("G", font_size=20).next_to(G_coord, DOWN, 0.1) # Renamed
        label_A = Text("A", font_size=20).next_to(A_coord, UP, 0.1)
        label_D_marker = Text("D", font_size=20).next_to(line_CD_obj.get_end(), RIGHT, 0.1)
        label_C_marker_line = Text("C", font_size=20).next_to(line_CD_obj.get_start(), LEFT, 0.1)


        tick_template = Line(UP*0.1+LEFT*0.1, DOWN*0.1+RIGHT*0.1, stroke_width=2.5, color=GRAY)
        pm_AB = VGroup(tick_template.copy().move_to(line_AB_obj.point_from_proportion(0.45)), tick_template.copy().move_to(line_AB_obj.point_from_proportion(0.45)+RIGHT*0.1))
        pm_CD = VGroup(tick_template.copy().move_to(line_CD_obj.point_from_proportion(0.40)), tick_template.copy().move_to(line_CD_obj.point_from_proportion(0.40)+RIGHT*0.1))

        diagram = VGroup(line_AB_obj, line_CD_obj, trans_EC_obj, trans_EG_obj, dot_E, dot_C, dot_G, label_E, label_C_diag, label_G_diag, label_A, label_D_marker, label_C_marker_line, pm_AB, pm_CD)
        diagram.shift(RIGHT*1 + UP*0.2)
        self.play(Create(diagram))

        prev_info_group = VGroup(
            MathTex("x = 30", font_size=24),
            MathTex("\\text{∠AEC} = 80^\\circ", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT).to_corner(UL, buff=0.5)
        self.play(Write(prev_info_group))
        self.wait(1)

        # Text and calculations on the left
        calc_group_left = VGroup().move_to(LEFT*3.5 + DOWN*0.5)

        # Calculate ∠CEG
        calc_CEG_title = MathTex("\\text{1. Find ∠CEG:}", font_size=26).next_to(title, DOWN, buff=0.4, aligned_edge=LEFT).shift(LEFT*3.5)
        calc_CEG_expr = MathTex("\\text{∠CEG} = x/3 + 30^\\circ", font_size=24).next_to(calc_CEG_title, DOWN, aligned_edge=LEFT)
        calc_CEG_sub = MathTex("\\text{∠CEG} = 30/3 + 30^\\circ", font_size=24).next_to(calc_CEG_expr, DOWN, aligned_edge=LEFT)
        calc_CEG_val = MathTex("\\text{∠CEG} = 10^\\circ + 30^\\circ = 40^\\circ", font_size=24).next_to(calc_CEG_sub, DOWN, aligned_edge=LEFT)
        calc_group_left.add(calc_CEG_title, calc_CEG_expr, calc_CEG_sub, calc_CEG_val)

        self.play(Write(calc_CEG_title))
        self.play(Write(calc_CEG_expr))
        self.play(TransformMatchingTex(calc_CEG_expr.copy(), calc_CEG_sub, path_arc=PI/2))
        self.play(TransformMatchingTex(calc_CEG_sub.copy(), calc_CEG_val, path_arc=PI/2))

        angle_CEG_arc = Angle(trans_EC_obj, trans_EG_obj, radius=0.5, color=ORANGE)
        label_CEG_val_diag = MathTex("40^\\circ", color=ORANGE).scale(0.5).move_to(Angle(trans_EC_obj, trans_EG_obj, radius=0.7).point_from_proportion(0.5))
        self.play(Create(angle_CEG_arc), Write(label_CEG_val_diag))
        self.wait(1)

        line_EA = Line(E_coord, A_coord)
        angle_AEC_arc_disp = Angle(line_EA, trans_EC_obj, radius=0.7, other_angle=False, color=RED)
        label_AEC_val_diag = MathTex("80^\\circ", color=RED).scale(0.5).move_to(Angle(line_EA, trans_EC_obj, radius=0.9, other_angle=False).point_from_proportion(0.5))
        self.play(FadeIn(angle_AEC_arc_disp), FadeIn(label_AEC_val_diag)) # Show existing AEC

        # Calculate ∠AEG
        calc_AEG_title = MathTex("\\text{2. Find ∠AEG:}", font_size=26).next_to(calc_CEG_val, DOWN, buff=0.4, aligned_edge=LEFT)
        calc_AEG_expr = MathTex("\\text{∠AEG} = \\text{∠AEC} - \\text{∠CEG}", font_size=24).next_to(calc_AEG_title, DOWN, aligned_edge=LEFT)
        calc_AEG_sub = MathTex("\\text{∠AEG} = 80^\\circ - 40^\\circ", font_size=24).next_to(calc_AEG_expr, DOWN, aligned_edge=LEFT)
        calc_AEG_val = MathTex("\\text{∠AEG} = 40^\\circ", font_size=24).next_to(calc_AEG_sub, DOWN, aligned_edge=LEFT)
        calc_group_left.add(calc_AEG_title, calc_AEG_expr, calc_AEG_sub, calc_AEG_val)

        self.play(Write(calc_AEG_title))
        self.play(Write(calc_AEG_expr))
        self.play(TransformMatchingTex(calc_AEG_expr.copy(), calc_AEG_sub, path_arc=PI/2))
        self.play(TransformMatchingTex(calc_AEG_sub.copy(), calc_AEG_val, path_arc=PI/2))

        angle_AEG_arc = Angle(line_EA, trans_EG_obj, radius=0.4, other_angle=False, color=YELLOW) # Might need other_angle=True
        label_AEG_val_diag = MathTex("40^\\circ", color=YELLOW).scale(0.5).move_to(Angle(line_EA, trans_EG_obj, radius=0.6, other_angle=False).point_from_proportion(0.5))
        self.play(Create(angle_AEG_arc), Write(label_AEG_val_diag))
        self.wait(1)

        # Angle EGC = a
        line_GE = Line(G_coord, E_coord)
        line_GC = Line(G_coord, C_coord) # Or G to a point on CD left/right of G
        angle_EGC_arc = Angle(line_GE, line_GC, radius=0.5, color=PURPLE_A)
        label_EGC_val_diag = MathTex("a", color=PURPLE_A).scale(0.6).move_to(Angle(line_GE, line_GC, radius=0.7).point_from_proportion(0.5))
        self.play(Create(angle_EGC_arc), Write(label_EGC_val_diag))
        self.wait(1)

        # Reason and final answer for 'a'
        reason_a_title = MathTex("\\text{3. Find 'a':}", font_size=26).next_to(calc_AEG_val, DOWN, buff=0.4, aligned_edge=LEFT)
        reason_a = Text("∠AEG = ∠EGC ('a') (alternate angles, AB || CD)", font_size=20).next_to(reason_a_title, DOWN, aligned_edge=LEFT)
        final_a = MathTex("a = 40^\\circ", font_size=28).next_to(reason_a, DOWN, buff=0.2, aligned_edge=LEFT)
        calc_group_left.add(reason_a_title, reason_a, final_a)

        self.play(Write(reason_a_title))
        self.play(Write(reason_a))
        self.play(Indicate(angle_AEG_arc, color=YELLOW), Indicate(angle_EGC_arc, color=PURPLE_A))
        self.wait(1)
        self.play(Write(final_a))
        box_a = SurroundingRectangle(final_a, buff=0.1, color=GREEN)
        self.play(Create(box_a))
        self.wait(3)
