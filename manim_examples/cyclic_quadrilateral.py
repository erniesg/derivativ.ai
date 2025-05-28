from manim import *
import numpy as np

class ExamDiagramExact(Scene):
    def construct(self):
        # Match the exam diagram's style
        self.camera.background_color = WHITE

        # Create the circle with black stroke
        circle = Circle(radius=2.5, color=BLACK, stroke_width=2)

        # Position vertices to match the exam diagram
        # A is at bottom, B is at bottom-right, C is at top-right, D is at top-left
        A = circle.point_at_angle(-PI/2)  # Bottom
        B = circle.point_at_angle(-PI/6)  # Bottom-right
        C = circle.point_at_angle(PI/3)   # Top-right
        D = circle.point_at_angle(5*PI/6) # Top-left

        # Create the quadrilateral (no fill, just black outline)
        quadrilateral = Polygon(A, B, C, D,
                               color=BLACK,
                               stroke_width=2,
                               fill_opacity=0)

        # Create diagonals with thinner lines
        diagonal_AC = Line(A, C, color=BLACK, stroke_width=1.5)
        diagonal_BD = Line(B, D, color=BLACK, stroke_width=1.5)

        # Calculate intersection point X
        # Using parametric equations for line intersection
        def line_intersection(p1, p2, p3, p4):
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            x3, y3 = p3[0], p3[1]
            x4, y4 = p4[0], p4[1]

            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1e-10:
                return np.array([0, 0, 0])

            t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
            x = x1 + t*(x2-x1)
            y = y1 + t*(y2-y1)
            return np.array([x, y, 0])

        X = line_intersection(A, C, B, D)

        # Create vertex labels
        label_A = Text("A", font_size=24, color=BLACK).next_to(A, DOWN, buff=0.15)
        label_B = Text("B", font_size=24, color=BLACK).next_to(B, RIGHT, buff=0.15)
        label_C = Text("C", font_size=24, color=BLACK).next_to(C, RIGHT, buff=0.15)
        label_D = Text("D", font_size=24, color=BLACK).next_to(D, LEFT, buff=0.15)
        label_X = Text("X", font_size=24, color=BLACK).next_to(X, DOWN+LEFT, buff=0.1)

        # Create angle arcs
        # Angle BAD (74°)
        angle_BAD = Angle(
            Line(A, B), Line(A, D),
            radius=0.4,
            color=BLACK,
            stroke_width=1
        )

        # Position the 74° label inside the angle
        angle_BAD_label = Text("74°", font_size=20, color=BLACK).move_to(
            A + 0.6 * (B + D - 2*A) / np.linalg.norm(B + D - 2*A)
        )

        # Angle BCA (34°)
        angle_BCA = Angle(
            Line(C, B), Line(C, A),
            radius=0.4,
            color=BLACK,
            stroke_width=1
        )

        # Position the 34° label inside the angle
        angle_BCA_label = Text("34°", font_size=20, color=BLACK).move_to(
            C + 0.6 * (B + A - 2*C) / np.linalg.norm(B + A - 2*C)
        )

        # NOT TO SCALE text
        not_to_scale = Text("NOT TO\nSCALE",
                          font_size=20,
                          color=BLACK,
                          line_spacing=0.5).to_corner(UR).shift(LEFT*0.5)

        # Add all elements in the correct order
        self.add(circle)
        self.add(quadrilateral)
        self.add(diagonal_AC, diagonal_BD)
        self.add(label_A, label_B, label_C, label_D, label_X)
        self.add(angle_BAD, angle_BAD_label)
        self.add(angle_BCA, angle_BCA_label)
        self.add(not_to_scale)

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
