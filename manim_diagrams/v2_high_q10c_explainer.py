from manim import *
import numpy as np

class V2_High_Q10c_Explainer(Scene):
    def construct(self):
        title = Text("Question 10 (c) - Find 'b'", font_size=30).to_edge(UP)
        self.play(Write(title))

        # Diagram: Triangle ECG
        E_coord = np.array([-1, 0.5, 0]) # Adjusted for layout
        C_coord = np.array([-2, -1.0, 0])
        G_coord = np.array([0, -1.0, 0])

        line_EC_obj = Line(E_coord, C_coord, color=BLUE_A)
        line_EG_obj = Line(E_coord, G_coord, color=GREEN_A)
        line_CG_obj = Line(C_coord, G_coord, color=WHITE)
        dot_E = Dot(E_coord); dot_C = Dot(C_coord); dot_G = Dot(G_coord)
        label_E = Text("E", font_size=20).next_to(E_coord, UP, 0.1)
        label_C_diag = Text("C", font_size=20).next_to(C_coord, DOWN, 0.1) # Renamed
        label_G_diag = Text("G", font_size=20).next_to(G_coord, DOWN, 0.1) # Renamed

        triangle_group = VGroup(line_EC_obj, line_EG_obj, line_CG_obj, dot_E, dot_C, dot_G, label_E, label_C_diag, label_G_diag)
        triangle_group.shift(RIGHT*2 + UP*0.5)
        self.play(Create(triangle_group))

        # Previous answers needed
        prev_ans_group = VGroup(
            MathTex("\\text{∠CEG} = 40^\\circ", font_size=24),
            MathTex("a = \\text{∠EGC} = 40^\\circ", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT).to_corner(UL, buff=0.5)
        self.play(Write(prev_ans_group))
        self.wait(1)

        # Show angles on triangle ECG
        angle_CEG_arc = Angle(line_EC_obj, line_EG_obj, radius=0.5, color=ORANGE)
        label_CEG_val_diag = MathTex("40^\\circ", color=ORANGE).scale(0.5).move_to(Angle(line_EC_obj, line_EG_obj, radius=0.7).point_from_proportion(0.5))
        self.play(Create(angle_CEG_arc), Write(label_CEG_val_diag))
        self.wait(0.5)

        line_GE = Line(G_coord, E_coord)
        line_GC = Line(G_coord, C_coord)
        angle_EGC_arc = Angle(line_GE, line_GC, radius=0.5, color=PURPLE_A)
        label_EGC_val_diag = MathTex("a=40^\\circ", color=PURPLE_A).scale(0.5).move_to(Angle(line_GE, line_GC, radius=0.7).point_from_proportion(0.5))
        self.play(Create(angle_EGC_arc), Write(label_EGC_val_diag))
        self.wait(0.5)

        line_CE = Line(C_coord, E_coord)
        line_CG_for_angle = Line(C_coord, G_coord) # Renamed to avoid conflict
        angle_ECG_arc = Angle(line_CE, line_CG_for_angle, radius=0.5, color=TEAL_A)
        label_ECG_val_diag = MathTex("b", color=TEAL_A).scale(0.6).move_to(Angle(line_CE, line_CG_for_angle, radius=0.7).point_from_proportion(0.5))
        self.play(Create(angle_ECG_arc), Write(label_ECG_val_diag))
        self.wait(1)

        # Reason and Calculation Text
        calc_b_group = VGroup().next_to(title, DOWN, buff=0.4, aligned_edge=LEFT).shift(LEFT*3.5)

        reason_text = Text("In ΔECG, sum of angles = 180°", font_size=24, color=YELLOW)
        calc_b_group.add(reason_text)

        equation = MathTex("b + \\text{∠CEG} + \\text{∠EGC} = 180^\\circ", font_size=24).next_to(reason_text, DOWN, aligned_edge=LEFT, buff=0.3)
        sub_eq = MathTex("b + 40^\\circ + 40^\\circ = 180^\\circ", font_size=24).next_to(equation, DOWN, aligned_edge=LEFT)
        simplify_eq = MathTex("b + 80^\\circ = 180^\\circ", font_size=24).next_to(sub_eq, DOWN, aligned_edge=LEFT)
        solve_b = MathTex("b = 180^\\circ - 80^\\circ", font_size=24).next_to(simplify_eq, DOWN, aligned_edge=LEFT)
        final_b = MathTex("b = 100^\\circ", font_size=28).next_to(solve_b, DOWN, aligned_edge=LEFT)
        calc_b_group.add(equation, sub_eq, simplify_eq, solve_b, final_b)

        self.play(Write(reason_text))
        self.wait(1)
        self.play(Write(equation))
        self.wait(0.5)
        self.play(TransformMatchingTex(equation.copy(), sub_eq, path_arc=PI/2))
        self.wait(1)
        self.play(TransformMatchingTex(sub_eq.copy(), simplify_eq, path_arc=PI/2))
        self.wait(1)
        self.play(TransformMatchingTex(simplify_eq.copy(), solve_b, path_arc=PI/2))
        self.wait(1)
        self.play(TransformMatchingTex(solve_b.copy(), final_b, path_arc=PI/2))
        self.wait(1)

        box_b = SurroundingRectangle(final_b, buff=0.1, color=GREEN)
        self.play(Create(box_b))
        final_reason = Text("Reason: Angles in a triangle", font_size=22, color=GREEN).next_to(final_b, DOWN, buff=0.2)
        self.play(Write(final_reason))
        self.wait(3)
