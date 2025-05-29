from manim import *
import numpy as np

class V1_Low_Q10b_Explainer(Scene):
    def construct(self):
        title = Text("Question 10 (b) - Find y", font_size=30).to_edge(UP)
        self.play(Write(title))

        E_coord = np.array([0, 1.0, 0])
        C_coord = np.array([-0.8, -1.0, 0])
        G_coord = np.array([1.2, -1.0, 0])

        # Lines forming the triangle
        line_EC_obj = Line(E_coord, C_coord, color=BLUE_C)
        line_EG_obj = Line(E_coord, G_coord, color=GREEN_C)
        line_CG_obj = Line(C_coord, G_coord, color=WHITE)

        # Dots at vertices
        dot_E = Dot(E_coord, color=YELLOW)
        dot_C = Dot(C_coord, color=YELLOW)
        dot_G = Dot(G_coord, color=YELLOW)

        # Labels for vertices
        label_E_text = Text("E", font_size=24).next_to(E_coord, UP, buff=0.1)
        label_C_text = Text("C", font_size=24).next_to(C_coord, DOWN, buff=0.1)
        label_G_text = Text("G", font_size=24).next_to(G_coord, DOWN, buff=0.1)

        triangle_ECG_group = VGroup(line_EC_obj, line_EG_obj, line_CG_obj, dot_E, dot_C, dot_G, label_E_text, label_C_text, label_G_text)
        self.play(Create(triangle_ECG_group))
        self.wait(1)

        # Angle CEG = 50°
        angle_CEG_arc = Angle(line_EC_obj, line_EG_obj, radius=0.7, color=GREEN_A)
        label_CEG_val = MathTex("50^\\circ", color=GREEN_A).scale(0.7)
        label_CEG_val.move_to(Angle(line_EC_obj, line_EG_obj, radius=0.9).point_from_proportion(0.5))
        self.play(Create(angle_CEG_arc), Write(label_CEG_val))
        self.wait(0.5)

        # Angle EGC = 30° (vector GE, vector GC)
        # Need to ensure other_angle is correct for interior angle
        angle_EGC_arc = Angle(Line(G_coord, E_coord), Line(G_coord, C_coord), radius=0.6, color=PURPLE_A)
        label_EGC_val = MathTex("30^\\circ", color=PURPLE_A).scale(0.7)
        label_EGC_val.move_to(Angle(Line(G_coord, E_coord), Line(G_coord, C_coord), radius=0.8).point_from_proportion(0.5))
        self.play(Create(angle_EGC_arc), Write(label_EGC_val))
        self.wait(0.5)

        # Angle ECG = y (vector CE, vector CG)
        angle_ECG_arc = Angle(Line(C_coord, E_coord), Line(C_coord, G_coord), radius=0.6, color=TEAL_A)
        label_ECG_val = MathTex("y", color=TEAL_A).scale(0.7)
        label_ECG_val.move_to(Angle(Line(C_coord, E_coord), Line(C_coord, G_coord), radius=0.8).point_from_proportion(0.5))
        self.play(Create(angle_ECG_arc), Write(label_ECG_val))
        self.wait(1)

        # Reason and calculation text
        text_group = VGroup(
            Text("In triangle ECG:", font_size=24),
            Text("Sum of angles in a triangle = 180°", font_size=24, color=YELLOW),
            MathTex("y + 50^\\circ + 30^\\circ = 180^\\circ", font_size=30),
            MathTex("y + 80^\\circ = 180^\\circ", font_size=30),
            MathTex("y = 180^\\circ - 80^\\circ", font_size=30),
            MathTex("y = 100^\\circ", font_size=36)
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT).next_to(title, DOWN, buff=0.3).shift(LEFT*3)

        self.play(Write(text_group[0]))
        self.wait(0.5)
        self.play(Write(text_group[1]))
        self.wait(1)
        self.play(Write(text_group[2]))
        self.wait(1)
        self.play(TransformMatchingTex(text_group[2].copy(), text_group[3], path_arc=PI/2))
        self.wait(1)
        self.play(TransformMatchingTex(text_group[3].copy(), text_group[4], path_arc=PI/2))
        self.wait(1)
        self.play(TransformMatchingTex(text_group[4].copy(), text_group[5], path_arc=PI/2))
        self.wait(1)

        final_answer_box = SurroundingRectangle(text_group[5], buff=0.1, color=GREEN)
        final_reason_text = Text("Reason: Angles in a triangle", font_size=24, color=GREEN).next_to(text_group[5], DOWN, buff=0.2)
        self.play(Create(final_answer_box), Write(final_reason_text))
        self.wait(3)
