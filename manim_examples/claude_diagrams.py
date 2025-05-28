from manim import *
import numpy as np

class NetDiagram(Scene):
    def setup(self):
        self.camera.background_color = WHITE
    def construct(self):
        # Create the net diagram (Image 1)
        # Central square
        square = Square(side_length=1.5)

        # Four triangular flaps
        triangle_up = Triangle().scale(0.75).next_to(square, UP, buff=0)
        triangle_down = Triangle().scale(0.75).next_to(square, DOWN, buff=0).rotate(PI)
        triangle_left = Triangle().scale(0.75).next_to(square, LEFT, buff=0).rotate(PI/2)
        triangle_right = Triangle().scale(0.75).next_to(square, RIGHT, buff=0).rotate(-PI/2)

        net = VGroup(square, triangle_up, triangle_down, triangle_left, triangle_right)
        net.set_stroke(BLACK, 2).set_fill(WHITE, 1)

        title = Text("The diagram shows the net of a solid.", font_size=24, color=BLACK).to_edge(UP)

        self.add(title, net)

class RotationalSymmetryDiagram(Scene):
    def setup(self):
        self.camera.background_color = WHITE
    def construct(self):
        # Create hexagon with triangular sections (Image 2)
        hexagon = RegularPolygon(6, radius=2)

        # Create 6 triangular sections
        center = ORIGIN
        vertices = hexagon.get_vertices()

        triangles = []
        # Shade every other triangle (5 total as specified)
        shaded_indices = [0, 1, 3, 4, 5]  # Leave one unshaded for the task

        for i in range(6):
            triangle = Polygon(center, vertices[i], vertices[(i+1)%6])
            if i in shaded_indices:
                triangle.set_fill(GRAY, 0.7)
            else:
                triangle.set_fill(WHITE, 1)
            triangle.set_stroke(BLACK, 2)
            triangles.append(triangle)

        diagram = VGroup(*triangles)

        title = Text("The diagram shows a shape with five shaded sections.", font_size=24, color=BLACK).to_edge(UP)
        instruction = Text("Shade one more section on the diagram so that it has rotational symmetry of order 3.",
                          font_size=20, color=BLACK).to_edge(DOWN)

        self.add(title, diagram, instruction)

class ParallelLinesAngles(Scene):
    def setup(self):
        self.camera.background_color = WHITE
    def construct(self):
        # Create the parallel lines diagram (Image 3)

        # Parallel lines AB and CD
        line_ab = Line(LEFT*3, RIGHT*1).shift(UP*1.5)
        line_cd = Line(LEFT*1, RIGHT*3).shift(DOWN*1.5)

        # Transversal lines EF and EG
        point_e = LEFT*1.5 + UP*0.3
        line_ef = Line(point_e + UP*1.5, point_e + DOWN*2)
        line_eg = Line(point_e, point_e + RIGHT*3 + DOWN*0.8)

        # Labels
        label_a = Text("A", font_size=20, color=BLACK).next_to(line_ab.get_start(), LEFT)
        label_b = Text("B", font_size=20, color=BLACK).next_to(line_ab.get_end(), RIGHT)
        label_c = Text("C", font_size=20, color=BLACK).next_to(line_cd.get_end(), RIGHT)
        label_d = Text("D", font_size=20, color=BLACK).next_to(line_cd.get_start(), LEFT)
        label_e = Text("E", font_size=20, color=BLACK).next_to(point_e, LEFT)
        label_f = Text("F", font_size=20, color=BLACK).next_to(line_ef.get_start(), UP)
        label_g = Text("G", font_size=20, color=BLACK).next_to(line_eg.get_end(), RIGHT)

        # Angle markings
        angle_60 = Arc(radius=0.3, start_angle=line_ef.get_angle() + PI/2,
                      angle=PI/3, arc_center=point_e + UP*0.8, color=BLACK)
        angle_60_label = Text("60°", font_size=16, color=BLACK).next_to(angle_60, UP)

        angle_46 = Arc(radius=0.3, start_angle=line_ab.get_angle(),
                      angle=PI/4, arc_center=point_e, color=BLACK)
        angle_46_label = Text("46°", font_size=16, color=BLACK).next_to(angle_46, DOWN+LEFT)

        angle_c = Arc(radius=0.3, start_angle=0, angle=PI/6,
                     arc_center=line_cd.get_end() + LEFT*0.5, color=BLACK)
        angle_c_label = Text("c°", font_size=16, color=BLACK).next_to(angle_c, LEFT)

        angle_a = Arc(radius=0.3, start_angle=0, angle=PI/4,
                     arc_center=line_cd.get_end() + LEFT*1, color=BLACK)
        angle_a_label = Text("a°", font_size=16, color=BLACK).next_to(angle_a, UP)

        angle_b = Arc(radius=0.3, start_angle=0, angle=PI/5,
                     arc_center=line_cd.get_end() + LEFT*1.5, color=BLACK)
        angle_b_label = Text("b°", font_size=16, color=BLACK).next_to(angle_b, UP)

        all_objects = VGroup(
            line_ab, line_cd, line_ef, line_eg,
            label_a, label_b, label_c, label_d, label_e, label_f, label_g,
            angle_60, angle_60_label, angle_46, angle_46_label,
            angle_a, angle_a_label, angle_b, angle_b_label, angle_c, angle_c_label
        )

        info_text = VGroup(
            Text("Lines AB and CD are parallel.", font_size=20, color=BLACK),
            Text("EF and EG are straight lines.", font_size=20, color=BLACK)
        ).arrange(DOWN, aligned_edge=LEFT).to_edge(DOWN)

        question = Text("(a) Find the value of a.", font_size=20, color=BLACK).to_edge(DOWN, buff=2)

        self.add(all_objects, info_text, question)

class ParallelogramArea(Scene):
    def construct(self):
        # Create compound parallelogram shape (Image 4)

        # Main parallelogram
        main_para = Polygon(
            [-2, 0, 0], [1, 0, 0], [1.5, 1, 0], [-1.5, 1, 0]
        )

        # Smaller parallelogram cutout
        small_para = Polygon(
            [-1.5, 0, 0], [-0.5, 0, 0], [-0.25, 0.4, 0], [-1.25, 0.4, 0]
        )

        # Create the compound shape
        compound_shape = main_para
        compound_shape.set_fill(LIGHT_GRAY, 0.3)
        compound_shape.set_stroke(BLACK, 2)

        small_para.set_fill(WHITE, 1)
        small_para.set_stroke(BLACK, 2)

        # Dimension labels
        width_label = Text("15 cm", font_size=16).next_to(main_para, UP)
        height_label = Text("x cm", font_size=16).next_to(main_para, LEFT)
        small_height_label = Text("4 cm", font_size=16).next_to(small_para, LEFT)

        # Dashed construction lines
        dashed_line_top = DashedLine(
            main_para.get_vertices()[3] + UP*0.2,
            main_para.get_vertices()[2] + UP*0.2
        )
        dashed_line_bottom = DashedLine(
            main_para.get_vertices()[0] + DOWN*0.2,
            main_para.get_vertices()[1] + DOWN*0.2
        )

        title = Text("The diagram shows a shape made from two different parallelograms.",
                    font_size=20).to_edge(UP)
        area_info = Text("The shape has a total area of 210 cm².", font_size=20).next_to(title, DOWN)
        question = Text("Find the value of x.", font_size=20).to_edge(DOWN)

        self.add(compound_shape, small_para, width_label, height_label, small_height_label,
                dashed_line_top, dashed_line_bottom, title, area_info, question)

class ScatterPlot(Scene):
    def construct(self):
        # Create scatter plot (Image 5)

        # Set up axes
        axes = Axes(
            x_range=[10.0, 11.4, 0.2],
            y_range=[23, 27, 1],
            x_length=8,
            y_length=6,
            axis_config={"color": BLACK, "stroke_width": 1},
            tips=False
        )

        # Add grid
        grid = axes.get_axis_labels()

        # Sample data points
        data_points = [
            (10.3, 24.1), (10.4, 24.3), (10.5, 24.5), (10.6, 24.2),
            (10.7, 24.4), (10.8, 24.9), (10.9, 25.1), (11.0, 24.6),
            (11.1, 25.2), (11.3, 26.1)
        ]

        # Create scatter points
        points = VGroup()
        for x, y in data_points:
            point = Text("×", font_size=16, color=BLACK)
            point.move_to(axes.coords_to_point(x, y))
            points.add(point)

        # Labels
        x_label = Text("Time to run 100m (s)", font_size=16).next_to(axes, DOWN)
        y_label = Text("Time to complete the swimming race (s)", font_size=16).rotate(PI/2).next_to(axes, LEFT)

        title = Text("As part of a sports competition, 14 athletes run 100m and complete a swimming race.",
                    font_size=16).to_edge(UP)
        subtitle = Text("The scatter diagram shows the times, in seconds, to run 100m and the times, in seconds, to",
                       font_size=16).next_to(title, DOWN)
        subtitle2 = Text("complete the swimming race, for 11 of these athletes.",
                        font_size=16).next_to(subtitle, DOWN)

        # Table data
        table_title = Text("The table shows the times for the other 3 athletes.",
                          font_size=16).to_edge(DOWN, buff=2)

        self.add(axes, points, x_label, y_label, title, subtitle, subtitle2, table_title)

# To render these, you would use:
# manim -pql script_name.py NetDiagram
# manim -pql script_name.py RotationalSymmetryDiagram
# manim -pql script_name.py ParallelLinesAngles
# manim -pql script_name.py ParallelogramArea
# manim -pql script_name.py ScatterPlot
