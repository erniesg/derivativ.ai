from manim import *

class CyclicQuadrilateralAngles(Scene):
    def construct(self):
        # 1. Define Circle
        circle_radius = 3
        circle = Circle(radius=circle_radius, color=BLACK, stroke_width=2)

        # 2. Define Points (Approximate angular positions on the circle for visual resemblance)
        # Angles in degrees for point_at_angle, measured counter-clockwise from positive x-axis
        # After some trial and error to match the visual:
        A_angle_rad = 248 * DEGREES
        B_angle_rad = 328 * DEGREES # Adjusted to visually fit ACB=34
        C_angle_rad = 38 * DEGREES
        D_angle_rad = 145 * DEGREES # Adjusted to visually fit CAD=74

        A_coord = circle.point_at_angle(A_angle_rad)
        B_coord = circle.point_at_angle(B_angle_rad)
        C_coord = circle.point_at_angle(C_angle_rad)
        D_coord = circle.point_at_angle(D_angle_rad)

        A = Dot(A_coord, color=BLACK, radius=0.06)
        B = Dot(B_coord, color=BLACK, radius=0.06)
        C = Dot(C_coord, color=BLACK, radius=0.06)
        D = Dot(D_coord, color=BLACK, radius=0.06)

        A_label = MathTex("A", color=BLACK, font_size=36).next_to(A, DOWN, buff=0.2)
        B_label = MathTex("B", color=BLACK, font_size=36).next_to(B, DR, buff=0.15).shift(RIGHT*0.1)
        C_label = MathTex("C", color=BLACK, font_size=36).next_to(C, UR, buff=0.15)
        D_label = MathTex("D", color=BLACK, font_size=36).next_to(D, UL, buff=0.2)

        # 3. Draw Lines (Sides and Diagonals)
        line_AD = Line(A_coord, D_coord, color=BLACK, stroke_width=2)
        line_AC = Line(A_coord, C_coord, color=BLACK, stroke_width=2)
        line_AB = Line(A_coord, B_coord, color=BLACK, stroke_width=2)
        line_BC = Line(B_coord, C_coord, color=BLACK, stroke_width=2)
        line_CD = Line(C_coord, D_coord, color=BLACK, stroke_width=2)
        line_DB = Line(D_coord, B_coord, color=BLACK, stroke_width=2)

        # 4. Intersection Point X
        # Use manim's line_intersection utility
        from manim.utils.geometry import line_intersection as li
        X_coord = li([A_coord, C_coord], [D_coord, B_coord])
        # X_dot = Dot(X_coord, color=BLACK, radius=0.0) # X is not a dot in the image
        X_label = MathTex("X", color=BLACK, font_size=36).move_to(X_coord).shift(0.22*LEFT + 0.18*UP) # Fine-tuned

        # 5. Angle Markers and Labels
        # Angle CAD = 74 degrees
        # Ensure lines are defined from A for Angle object
        # Manim's Angle draws counter-clockwise from line1 to line2 by default.
        # For CAD, we need angle from AD to AC or AC to AD depending on other_angle
        angle_CAD_obj = Angle(line_AD, line_AC, radius=0.9, color=BLACK, stroke_width=1.5, other_angle=False)
        angle_CAD_label_val = MathTex("74^\\circ", color=BLACK, font_size=30).move_to(
            Angle(line_AD, line_AC, radius=0.9 + 0.25, other_angle=False).point_from_proportion(0.5)
        )

        # Angle ACB = 34 degrees
        # Ensure lines are defined from C for Angle object
        line_CA_for_angle = Line(C_coord, A_coord) # From C to A
        line_CB_for_angle = Line(C_coord, B_coord) # From C to B
        angle_ACB_obj = Angle(line_CA_for_angle, line_CB_for_angle, radius=0.6, color=BLACK, stroke_width=1.5, other_angle=False)
        angle_ACB_label_val = MathTex("34^\\circ", color=BLACK, font_size=30).move_to(
            Angle(line_CA_for_angle, line_CB_for_angle, radius=0.6 + 0.25, other_angle=False).point_from_proportion(0.5)
        )
        angle_ACB_label_val.shift(LEFT*0.05 + DOWN*0.05) # Minor adjustment

        # Group for drawing
        points_group = VGroup(A, B, C, D)
        labels_group = VGroup(A_label, B_label, C_label, D_label, X_label)
        lines_group = VGroup(line_AB, line_BC, line_CD, line_AD, line_AC, line_DB)
        angles_group = VGroup(angle_CAD_obj, angle_CAD_label_val, angle_ACB_obj, angle_ACB_label_val)

        # 6. Render (add everything at once for a static image)
        self.add(circle)
        self.add(lines_group) # Lines first so dots are on top
        self.add(points_group)
        self.add(labels_group)
        self.add(angles_group)

        # If you want animation:
        # self.play(Create(circle))
        # self.play(LaggedStart(*[FadeIn(p) for p in points_group], lag_ratio=0.2))
        # self.play(LaggedStart(*[Write(l) for l in VGroup(A_label, B_label, C_label, D_label)], lag_ratio=0.2))
        # self.play(Create(lines_group))
        # self.play(Write(X_label))
        # self.play(Create(angle_CAD_obj), Write(angle_CAD_label_val))
        # self.play(Create(angle_ACB_obj), Write(angle_ACB_label_val))
        # self.wait()
