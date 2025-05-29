from manim import *
import numpy as np

class V2_High_StaticDiagram(Scene):
    def construct(self):
        angle_label_scale = 0.55 # Slightly smaller for algebraic expressions
        angle_radius_factor = 1.4

        # Points definition
        E_coord = np.array([-0.5, 1.0, 0])
        A_coord = np.array([-2.8, 1.0, 0])
        B_coord = np.array([1.8, 1.0, 0])
        C_coord = np.array([0.3, -1.0, 0])
        G_coord = np.array([1.8, -1.0, 0]) # Adjusted G for clarity
        CD_line_start_coord = np.array([-2.8, -1.0, 0])
        CD_line_end_coord = np.array([2.8, -1.0, 0])

        # Lines
        line_AB_obj = Line(A_coord, B_coord, color=WHITE)
        line_CD_obj = Line(CD_line_start_coord, CD_line_end_coord, color=WHITE)
        trans_EC_obj = Line(E_coord, C_coord, color=BLUE_C)
        trans_EG_obj = Line(E_coord, G_coord, color=GREEN_C)

        # Dots
        dot_E = Dot(E_coord, color=YELLOW)
        dot_C = Dot(C_coord, color=YELLOW)
        dot_G = Dot(G_coord, color=YELLOW)

        # Point Labels
        label_A_text = Text("A", font_size=24).next_to(A_coord, UP, buff=0.1)
        label_B_text = Text("B", font_size=24).next_to(B_coord, UP, buff=0.1)
        label_E_text = Text("E", font_size=24).next_to(E_coord, UP, buff=0.1)
        label_C_text = Text("C", font_size=24).next_to(C_coord, DOWN, buff=0.1)
        label_G_text = Text("G", font_size=24).next_to(G_coord, DOWN, buff=0.1)
        label_D_marker = Text("D", font_size=24).next_to(CD_line_end_coord, RIGHT, buff=0.1)
        label_C_marker_line = Text("C", font_size=24).next_to(CD_line_start_coord, LEFT, buff=0.1)


        # Parallel marks
        tick_template = Line(UP*0.1+LEFT*0.1, DOWN*0.1+RIGHT*0.1, stroke_width=2.5, color=GRAY)
        pm_AB = VGroup(tick_template.copy().move_to(line_AB_obj.point_from_proportion(0.45)), tick_template.copy().move_to(line_AB_obj.point_from_proportion(0.45)+RIGHT*0.1))
        pm_CD = VGroup(tick_template.copy().move_to(line_CD_obj.point_from_proportion(0.55)), tick_template.copy().move_to(line_CD_obj.point_from_proportion(0.55)+RIGHT*0.1))

        # Helper lines for angles
        line_EA = Line(E_coord, A_coord)
        line_CE = Line(C_coord, E_coord)
        line_CD_left = Line(C_coord, CD_line_start_coord)
        line_GE = Line(G_coord, E_coord)
        line_GC = Line(G_coord, C_coord)
        line_CG = Line(C_coord, G_coord)


        # Angle AEC = (2x + 20)°
        angle_AEC_arc = Angle(line_EA, trans_EC_obj, radius=0.5, other_angle=False, color=RED)
        label_AEC_val = MathTex("(2x+20)^\\circ", color=RED).scale(angle_label_scale)
        label_AEC_val.move_to(Angle(line_EA, trans_EC_obj, radius=0.5*angle_radius_factor + 0.2, other_angle=False).point_from_proportion(0.5))

        # Angle ECD = (3x - 10)°
        angle_ECD_arc = Angle(line_CE, line_CD_left, radius=0.5, color=ORANGE)
        label_ECD_val = MathTex("(3x-10)^\\circ", color=ORANGE).scale(angle_label_scale)
        label_ECD_val.move_to(Angle(line_CE, line_CD_left, radius=0.5*angle_radius_factor + 0.2).point_from_proportion(0.5))

        # Angle CEG = (x/3 + 30)°
        angle_CEG_arc = Angle(trans_EC_obj, trans_EG_obj, radius=0.6, color=GREEN_A)
        label_CEG_val = MathTex("(\\frac{x}{3}+30)^\\circ", color=GREEN_A).scale(angle_label_scale*0.9) # slightly smaller for fraction
        label_CEG_val.move_to(Angle(trans_EC_obj, trans_EG_obj, radius=0.6*angle_radius_factor).point_from_proportion(0.5))

        # Angle EGC = a
        angle_EGC_arc = Angle(line_GE, line_GC, radius=0.5, color=PURPLE_A)
        label_EGC_val = MathTex("a", color=PURPLE_A).scale(angle_label_scale)
        label_EGC_val.move_to(Angle(line_GE, line_GC, radius=0.5*angle_radius_factor + 0.15).point_from_proportion(0.5))

        # Angle ECG = b
        angle_ECG_arc = Angle(line_CG, line_CE, radius=0.5, color=TEAL_A) # CG then CE
        label_ECG_val = MathTex("b", color=TEAL_A).scale(angle_label_scale)
        label_ECG_val.move_to(Angle(line_CG, line_CE, radius=0.5*angle_radius_factor + 0.15).point_from_proportion(0.5))

        self.add(
            line_AB_obj, line_CD_obj, trans_EC_obj, trans_EG_obj,
            dot_E, dot_C, dot_G,
            label_A_text, label_B_text, label_E_text, label_C_text, label_G_text, label_D_marker, label_C_marker_line,
            pm_AB, pm_CD,
            angle_AEC_arc, label_AEC_val,
            angle_ECD_arc, label_ECD_val,
            angle_CEG_arc, label_CEG_val,
            angle_EGC_arc, label_EGC_val,
            angle_ECG_arc, label_ECG_val
        )
