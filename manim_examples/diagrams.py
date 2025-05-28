from manim import *

class NetOfSquarePyramid(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        title_text = Text(
            "The diagram shows the net of a solid.",
            font_size=28,
            color=BLACK
        ).to_corner(UL).shift(RIGHT * 0.5 + DOWN * 0.5)

        square_side_length = 2.0
        s_half = square_side_length / 2.0

        # Define the central square by its vertices explicitly
        # Assume square is centered at ORIGIN for now, we'll move the whole net later.
        v_tl = np.array([-s_half,  s_half, 0])  # Top-Left
        v_tr = np.array([ s_half,  s_half, 0])  # Top-Right
        v_br = np.array([ s_half, -s_half, 0])  # Bottom-Right
        v_bl = np.array([-s_half, -s_half, 0])  # Bottom-Left

        central_square = Polygon(v_tl, v_tr, v_br, v_bl, color=BLACK, stroke_width=2)

        triangle_height_factor = 1.0
        triangle_height = square_side_length * triangle_height_factor

        # APEX CALCULATIONS using the explicitly defined vertices:
        # Midpoint of top edge: (v_tl + v_tr) / 2
        apex_top = ((v_tl + v_tr) / 2) + UP * triangle_height
        tri_top = Polygon(v_tl, v_tr, apex_top, color=BLACK, stroke_width=2)

        # Midpoint of right edge: (v_tr + v_br) / 2
        apex_right = ((v_tr + v_br) / 2) + RIGHT * triangle_height
        tri_right = Polygon(v_tr, v_br, apex_right, color=BLACK, stroke_width=2)

        # Midpoint of bottom edge: (v_br + v_bl) / 2
        apex_bottom = ((v_br + v_bl) / 2) + DOWN * triangle_height
        tri_bottom = Polygon(v_br, v_bl, apex_bottom, color=BLACK, stroke_width=2)

        # Midpoint of left edge: (v_bl + v_tl) / 2
        apex_left = ((v_bl + v_tl) / 2) + LEFT * triangle_height
        tri_left = Polygon(v_bl, v_tl, apex_left, color=BLACK, stroke_width=2)

        net = VGroup(central_square, tri_top, tri_right, tri_bottom, tri_left)

        # Position the entire net (which was built around ORIGIN)
        net.move_to(ORIGIN).shift(RIGHT * 1.5) # Example: Center it and shift right

        self.add(title_text, net)
        self.wait(0.5)
# To run this specific scene:
# manim your_file_name.py NetOfSquarePyramid -s --format png -ql

class RotationalSymmetryHexagon(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        title_text = Text(
            "7 The diagram shows a shape with five shaded sections.",
            font_size=28,
            color=BLACK
        ).to_corner(UL).shift(RIGHT * 0.5 + DOWN * 0.5)
        instruction_text = Text(
            "Shade one more section on the diagram so that it has rotational symmetry of order 3.",
            font_size=24,
            color=BLACK
        ).to_corner(DL).shift(RIGHT * 0.5 + UP * 0.5)

        center_point = ORIGIN
        outer_radius = 2.0
        inner_radius = outer_radius * 0.5

        # start_angle = 0 produces the "flat top/bottom" orientation
        # as confirmed by your labelled output.
        initial_start_angle = 0
        outer_hexagon = RegularPolygon(n=6, radius=outer_radius, color=BLACK, stroke_width=2, start_angle=initial_start_angle).move_to(center_point)
        inner_hexagon = RegularPolygon(n=6, radius=inner_radius, color=BLACK, stroke_width=1.5, start_angle=initial_start_angle).move_to(center_point)

        radial_lines = VGroup()
        outer_verts = outer_hexagon.get_vertices()
        for i in range(6):
            radial_lines.add(Line(center_point, outer_verts[i], color=BLACK, stroke_width=1.5))

        inner_verts = inner_hexagon.get_vertices()

        inner_triangles_list = []
        for i in range(6):
            tri = Polygon(center_point, inner_verts[i], inner_verts[(i + 1) % 6],
                          stroke_width=0, fill_opacity=0)
            inner_triangles_list.append(tri)
        inner_triangles = VGroup(*inner_triangles_list)

        outer_trapezoids_list = []
        for i in range(6):
            trap = Polygon(inner_verts[i], inner_verts[(i + 1) % 6], outer_verts[(i + 1) % 6], outer_verts[i],
                           stroke_width=0, fill_opacity=0)
            outer_trapezoids_list.append(trap)
        outer_trapezoids = VGroup(*outer_trapezoids_list)

        # SHADING BASED ON YOUR IDENTIFIED INDICES for start_angle = 0
        # Outer Trapezoids to Shade: 1 and 5
        outer_trapezoids[1].set_style(fill_color=GREY_B, fill_opacity=0.7)
        outer_trapezoids[5].set_style(fill_color=GREY_B, fill_opacity=0.7)

        # Inner Triangles to Shade: 0, 2, and 4
        inner_triangles[0].set_style(fill_color=GREY_B, fill_opacity=0.7)
        inner_triangles[2].set_style(fill_color=GREY_B, fill_opacity=0.7)
        inner_triangles[4].set_style(fill_color=GREY_B, fill_opacity=0.7)

        diagram_structure = VGroup(outer_hexagon, inner_hexagon, radial_lines)

        diagram_visual = VGroup()
        # Add shaded parts first so their fill is under the lines
        diagram_visual.add(*inner_triangles_list)
        diagram_visual.add(*outer_trapezoids_list)
        # Then add the structure lines
        diagram_visual.add(diagram_structure)

        # NO ROTATION of diagram_visual is needed, as start_angle=0
        # already gives the target orientation.

        diagram_visual.center().shift(UP * 0.3) # Position the final diagram

        point_marker = Text("[1]", font_size=20, color=BLACK).next_to(instruction_text, RIGHT, buff=0.1)
        self.add(title_text, diagram_visual, instruction_text, point_marker)
        self.wait(0.5)

# To run:
# manim your_script_name.py RotationalSymmetryHexagon -s --format png -ql
