from manim import *
import numpy as np

class V1_Low_Q10c_Explainer(Scene):
    def construct(self):
        title = Text("Question 10 (c) - Find z", font_size=30).to_edge(UP)
        self.play(Write(title))

        E_coord = np.array([0, 0, 0])
        A_coord = np.array([-2.5, 0, 0])
        B_coord = np.array([2.5, 0, 0])
        C_coord = np.array([1, -1.5, 0]) # Sloping down from E

        line_AEB_obj = Line(A_coord, B_coord, color=WHITE)
        trans_EC_obj = Line(E_coord, C_coord, color=BLUE_C) # Line from E to C
        dot_E = Dot(E_coord, color=YELLOW)
        dot_C_obj = Dot(C_coord, color=YELLOW) # Renamed dot

        label_A_text = Text("A", font_size=24).next_to(A_coord, UP, buff=0.1)
        label_E_text = Text("E", font_size=24).next_to(E_coord, UP, buff=0.1)
        label_B_text = Text("B", font_size=24).next_to(B_coord, UP, buff=0.1)
        label_C_text = Text("C", font_size=24).next_to(C_coord, DOWN, buff=0.1)

        diagram_elements = VGroup(line_AEB_obj, trans_EC_obj, dot_E, label_A_text, label_E_text, label_B_text, label_C_text, dot_C_obj)
        diagram_elements.shift(UP*0.5) # Shift diagram up a bit
        self.play(Create(diagram_elements))
        self.wait(1)

        # Helper lines for angles
        line_EA = Line(E_coord, A_coord) # Vector EA for angle AEC
        line_EB = Line(E_coord, B_coord) # Vector EB for angle CEB

        # Angle AEC = 70°
        angle_AEC_arc = Angle(line_EA, trans_EC_obj, radius=0.7, other_angle=True, color=RED)
        label_AEC_val = MathTex("70^\\circ", color=RED).scale(0.7)
        label_AEC_val.move_to(Angle(line_EA, trans_EC_obj, radius=0.9, other_angle=True).point_from_proportion(0.5))
        self.play(Create(angle_AEC_arc), Write(label_AEC_val))
        self.wait(0.5)

        # Angle CEB = z
        angle_CEB_arc = Angle(trans_EC_obj, line_EB, radius=0.7, color=PINK)
        label_CEB_val = MathTex("z", color=PINK).scale(0.7)
        label_CEB_val.move_to(Angle(trans_EC_obj, line_EB, radius=0.9).point_from_proportion(0.5))
        self.play(Create(angle_CEB_arc), Write(label_CEB_val))
        self.wait(1)

        # Reason and calculation text
        text_group = VGroup(
            Text("AEB is a straight line.", font_size=24),
            Text("Angles on a straight line sum to 180°.", font_size=24, color=YELLOW),
            MathTex("z + 70^\\circ = 180^\\circ", font_size=30),
            MathTex("z = 180^\\circ - 70^\\circ", font_size=30),
            MathTex("z = 110^\\circ", font_size=36)
        ).arrange(DOWN, buff=0.35, aligned_edge=LEFT).next_to(title, DOWN, buff=0.3).shift(LEFT*3.5)

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

        final_answer_box = SurroundingRectangle(text_group[4], buff=0.1, color=GREEN)
        final_reason_text = Text("Reason: Angles on a straight line", font_size=24, color=GREEN).next_to(text_group[4], DOWN, buff=0.2)
        self.play(Create(final_answer_box), Write(final_reason_text))
        self.wait(3)
