from manim import *
import numpy as np

class Q10aIV2HDiagram(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        VMobject.set_default(color=BLACK)
        Tex.set_default(color=BLACK)
        Text.set_default(color=BLACK)

        # Define key points - adjust coordinates as needed
        # Placing E on the left side of the diagram
        E_coord = np.array([-2, 1, 0])
        # Placing C and G on the right, below E
        C_coord = np.array([1, -1, 0])
        G_coord = np.array([3, -1, 0]) # G is just another point on line CD

        # Define lines AB and CD (parallel)
        # AB goes through E, horizontally
        line_AB = Line(E_coord + LEFT*4, E_coord + RIGHT*4)
        # CD goes through C and G, horizontally and parallel to AB
        line_CD = Line(C_coord + LEFT*2, C_coord + RIGHT*2) # Adjust length and position relative to C and G

        # Add parallel line indicators
        parallel_marker_AB = DoubleArrow(line_AB.get_left() + UP*0.2, line_AB.get_left() + UP*0.4, buff=0, tip_length=0.1)
        parallel_marker_CD = DoubleArrow(line_CD.get_left() + DOWN*0.2, line_CD.get_left() + DOWN*0.4, buff=0, tip_length=0.1)

        # Define line segments EC and EG
        line_EC = Line(E_coord, C_coord)
        line_EG = Line(E_coord, G_coord)

        # Add points
        point_E = Dot(E_coord)
        point_C = Dot(C_coord)
        point_G = Dot(G_coord)

        # Add labels for points
        label_E = MathTex("E").next_to(point_E, UP)
        label_C = MathTex("C").next_to(point_C, DOWN)
        label_G = MathTex("G").next_to(point_G, DOWN)

        # Define points on lines AB and CD for angle definition
        A_coord = E_coord + LEFT * 3 # A is on the line AB, left of E
        label_A = MathTex("A").next_to(A_coord, LEFT)
        D_coord = C_coord + RIGHT * 3 # D is on the line CD, right of C
        label_D = MathTex("D").next_to(D_coord, RIGHT)
        B_coord = E_coord + RIGHT * 3 # B is on the line AB, right of E
        label_B = MathTex("B").next_to(B_coord, RIGHT)

        # Angle AEC (2x + 20)°
        angle_AEC = Angle(Line(A_coord, E_coord), Line(E_coord, C_coord), radius=0.5, other_angle=False)
        label_AEC = angle_AEC.get_label(MathTex("(2x + 20)^{\circ}"))

        # Angle ECD (3x - 10)°
        angle_ECD = Angle(Line(C_coord, E_coord), Line(C_coord, D_coord), radius=0.5, other_angle=False)
        label_ECD = angle_ECD.get_label(MathTex("(3x - 10)^{\circ}"))

        # Angle CEG (x/3 + 30)° for part (b)
        angle_CEG = Angle(Line(E_coord, C_coord), Line(E_coord, G_coord), radius=0.5, other_angle=False)
        label_CEG = angle_CEG.get_label(MathTex("(\frac{x}{3} + 30)^{\circ}"))

        # Angle EGC (labeled 'a') for part (b)
        angle_EGC = Angle(Line(G_coord, E_coord), Line(G_coord, C_coord), radius=0.5, other_angle=False)
        label_EGC = angle_EGC.get_label(MathTex("a"))

        # Angle ECG (labeled 'b') for part (c)
        angle_ECG = Angle(Line(C_coord, E_coord), Line(C_coord, G_coord), radius=0.5, other_angle=False)
        label_ECG = angle_ECG.get_label(MathTex("b"))

        # Group all elements
        diagram_elements = VGroup(
            line_AB, line_CD, line_EC, line_EG,
            point_E, point_C, point_G,
            label_E, label_C, label_G,
            label_A, label_B, label_D,
            angle_AEC, label_AEC,
            angle_ECD, label_ECD,
            angle_CEG, label_CEG,
            angle_EGC, label_EGC,
            angle_ECG, label_ECG,
            parallel_marker_AB, parallel_marker_CD
        )

        # Add everything to the scene
        self.add(diagram_elements)

        # Optional: Add question number and "NOT TO SCALE"
        question_number = Text("10", font_size=36).to_corner(UL)
        not_to_scale = Text("NOT TO SCALE", font_size=24).to_corner(UR)
        self.add(question_number, not_to_scale)
