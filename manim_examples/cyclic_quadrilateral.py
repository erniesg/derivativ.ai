from manim import *
import numpy as np

class ExamDiagramSimple(Scene):
    def construct(self):
        # White background
        self.camera.background_color = WHITE

        # Create main circle
        circle = Circle(radius=2.5, color=BLACK, stroke_width=2)

        # Position vertices
        angle_A = -PI/2 + 0.1
        angle_B = -PI/6
        angle_C = PI/3.5
        angle_D = 5*PI/6

        A = circle.point_at_angle(angle_A)
        B = circle.point_at_angle(angle_B)
        C = circle.point_at_angle(angle_C)
        D = circle.point_at_angle(angle_D)

        # Create quadrilateral
        quad = Polygon(A, B, C, D, color=BLACK, stroke_width=2, fill_opacity=0)

        # Create diagonals
        diag_AC = Line(A, C, color=BLACK, stroke_width=1.5)
        diag_BD = Line(B, D, color=BLACK, stroke_width=1.5)

        # Find intersection
        def line_intersection(p1, p2, p3, p4):
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            x3, y3 = p3[0], p3[1]
            x4, y4 = p4[0], p4[1]

            det = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(det) < 1e-10:
                return (p1 + p2) / 2

            t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / det
            return p1 + t*(p2-p1)

        X = line_intersection(A, C, B, D)

        # Create labels
        labels = {
            'A': Text("A", font_size=24, color=BLACK).next_to(A, DOWN, buff=0.2),
            'B': Text("B", font_size=24, color=BLACK).next_to(B, DOWN+RIGHT, buff=0.15),
            'C': Text("C", font_size=24, color=BLACK).next_to(C, RIGHT, buff=0.15),
            'D': Text("D", font_size=24, color=BLACK).next_to(D, UP+LEFT, buff=0.15),
            'X': Text("X", font_size=24, color=BLACK).next_to(X, DOWN+LEFT, buff=0.1),
        }

        # Create angles with simple arcs
        # 74° angle at A
        angle_74 = Arc(
            start_angle=Line(A, B).get_angle(),
            angle=(Line(A, D).get_angle() - Line(A, B).get_angle()) % (2*PI),
            radius=0.5,
            arc_center=A,
            color=BLACK
        )
        label_74 = Text("74°", font_size=20, color=BLACK)
        label_74.move_to(A + 0.7 * ((B - A) + (D - A)) / np.linalg.norm((B - A) + (D - A)))

        # 34° angle at C
        angle_34 = Arc(
            start_angle=Line(C, A).get_angle(),
            angle=(Line(C, B).get_angle() - Line(C, A).get_angle()) % (2*PI),
            radius=0.5,
            arc_center=C,
            color=BLACK
        )
        label_34 = Text("34°", font_size=20, color=BLACK)
        label_34.move_to(C + 0.6 * ((B - C) + (A - C)) / np.linalg.norm((B - C) + (A - C)))

        # NOT TO SCALE text
        not_to_scale = Text("NOT TO\nSCALE", font_size=18, color=BLACK)
        not_to_scale.to_corner(UR).shift(LEFT*0.7 + DOWN*0.3)

        # Add everything
        self.add(circle)
        self.add(quad)
        self.add(diag_AC, diag_BD)
        for label in labels.values():
            self.add(label)
        self.add(angle_74, label_74)
        self.add(angle_34, label_34)
        self.add(not_to_scale)

# To render:
# manim -p -ql cyclic_quadrilateral.py ExamDiagramExact --format png
# or
# manim -p -ql cyclic_quadrilateral.py ExamDiagramSimple --format png
