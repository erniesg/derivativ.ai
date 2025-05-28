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

# Set background to white and default text/stroke to black
config.background_color = WHITE
# Manim's default text color is black when the background is white.
# We will explicitly set colors for drawn elements to black.

class ParallelogramProblemFinalRefined(Scene):
    def construct(self):
        element_color = BLACK

        # Visual heights for drawing to match target's proportions.
        # The problem is "NOT TO SCALE", so visual representation can differ.
        # x=10, h_bottom=4. Ratio 10:4 = 2.5:1.
        # Visually, the target image's top parallelogram is taller, but not 2.5x.
        # Let's use a visual ratio closer to 1.75:1 or 2:1 for drawing.
        x_draw_visual = 7.0  # Visual height for the 'x' part
        h_bottom_draw_visual = 4.0 # Visual height for the '4' part

        base_length = 15.0

        # Shear value 'S'. A positive S makes the outer non-common vertices shift right,
        # creating the '<' shape at the P_common_left vertex.
        S_shear = 2.0

        # 2. Define vertices for the parallelograms using visual heights
        P_common_left = np.array([0, 0, 0])
        P_common_right = np.array([base_length, 0, 0])

        P_top_TL = np.array([S_shear, x_draw_visual, 0])
        P_top_TR = np.array([base_length + S_shear, x_draw_visual, 0])
        poly_top_vertices = [P_common_left, P_common_right, P_top_TR, P_top_TL]

        P_bottom_BL = np.array([S_shear, -h_bottom_draw_visual, 0])
        P_bottom_BR = np.array([base_length + S_shear, -h_bottom_draw_visual, 0])
        poly_bottom_vertices = [P_common_left, P_common_right, P_bottom_BR, P_bottom_BL]

        # 3. Create Parallelogram Mobjects
        shape_stroke_width = 2.2
        poly_top = Polygon(*poly_top_vertices, stroke_color=element_color, stroke_width=shape_stroke_width, fill_opacity=0)
        poly_bottom = Polygon(*poly_bottom_vertices, stroke_color=element_color, stroke_width=shape_stroke_width, fill_opacity=0)

        # 4. Height dimensions
        height_dim_line_x_coord = -1.3 # Adjusted for spacing

        h_dim_line_top_pt = np.array([height_dim_line_x_coord, x_draw_visual, 0])
        h_dim_line_mid_pt = np.array([height_dim_line_x_coord, 0, 0])
        h_dim_line_bottom_pt = np.array([height_dim_line_x_coord, -h_bottom_draw_visual, 0])

        dim_line_stroke_width = 1.5
        h_line_x = Line(h_dim_line_top_pt, h_dim_line_mid_pt, stroke_color=element_color, stroke_width=dim_line_stroke_width)
        h_line_4 = Line(h_dim_line_mid_pt, h_dim_line_bottom_pt, stroke_color=element_color, stroke_width=dim_line_stroke_width)

        dash_props = {"stroke_color": element_color, "stroke_width": dim_line_stroke_width, "dash_length": 0.16, "dashed_ratio":0.6}
        dash_ext_top = DashedLine(h_dim_line_top_pt, P_top_TL, **dash_props)
        dash_ext_mid = DashedLine(h_dim_line_mid_pt, P_common_left, **dash_props)
        dash_ext_bottom = DashedLine(h_dim_line_bottom_pt, P_bottom_BL, **dash_props)

        dim_label_font_size = 42 # Increased label font size
        label_x = MathTex("x \\text{ cm}", font_size=dim_label_font_size, color=element_color).next_to(h_line_x, LEFT, buff=0.15)
        label_4 = MathTex("4 \\text{ cm}", font_size=dim_label_font_size, color=element_color).next_to(h_line_4, LEFT, buff=0.15)

        # Right angle markers
        right_angle_marker_size = 0.40 # Adjusted size
        right_angle_marker_top = Square(side_length=right_angle_marker_size,
                                   stroke_width=dim_line_stroke_width,
                                   color=element_color,
                                   fill_opacity=0)
        right_angle_marker_top.move_to(h_dim_line_mid_pt, aligned_edge=DL) # Bottom-Left corner at junction

        right_angle_marker_bottom = Square(side_length=right_angle_marker_size,
                                   stroke_width=dim_line_stroke_width,
                                   color=element_color,
                                   fill_opacity=0)
        right_angle_marker_bottom.move_to(h_dim_line_mid_pt, aligned_edge=UL) # Top-Left corner at junction


        # 5. Base dimension (for the top edge of the top parallelogram)
        base_dim_offset_y = 0.8

        b_dim_arrow_start = P_top_TL + UP * base_dim_offset_y
        b_dim_arrow_end = P_top_TR + UP * base_dim_offset_y

        base_ext_line1 = DashedLine(P_top_TL, b_dim_arrow_start, **dash_props)
        base_ext_line2 = DashedLine(P_top_TR, b_dim_arrow_end, **dash_props)

        dim_line_base = DoubleArrow(b_dim_arrow_start, b_dim_arrow_end, buff=0,
                                    stroke_color=element_color, stroke_width=dim_line_stroke_width, tip_length=0.22) # Slightly larger tip
        label_15 = MathTex("15 \\text{ cm}", font_size=dim_label_font_size, color=element_color).next_to(dim_line_base, UP, buff=0.12)

        # 6. Group all diagram elements
        diagram_vgroup = VGroup(
            poly_top, poly_bottom,
            h_line_x, h_line_4,
            dash_ext_top, dash_ext_mid, dash_ext_bottom,
            label_x, label_4, right_angle_marker_top, right_angle_marker_bottom,
            base_ext_line1, base_ext_line2, dim_line_base, label_15
        )

        diagram_vgroup.scale(0.37) # Fine-tuned scale
        diagram_vgroup.center().shift(DOWN * 0.1) # Fine-tuned position

        # 7. Text elements
        q_text_font_size = 29
        find_x_font_size = 29
        not_to_scale_font_size = 25

        q_text_line1 = Tex(r"\textbf{14}", " The diagram shows a shape made from two different parallelograms.", font_size=q_text_font_size, color=element_color)
        q_text_line2 = MathTex(r"\text{The shape has a total area of } 210 \, \text{cm}^2.", font_size=q_text_font_size, color=element_color)
        q_text_group = VGroup(q_text_line1, q_text_line2).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        q_text_group.to_corner(UL, buff=0.4)

        find_x_text = Tex("Find the value of $x$.", font_size=find_x_font_size, color=element_color)
        find_x_text.align_to(q_text_group, LEFT)
        find_x_text.to_edge(DOWN, buff=0.5)

        not_to_scale_text = Tex("NOT TO SCALE", font_size=not_to_scale_font_size, color=element_color)
        main_shapes_for_ref = VGroup(poly_top, poly_bottom) # For positioning reference
        not_to_scale_text.next_to(main_shapes_for_ref, RIGHT, buff=0.4)

        self.add(q_text_group)
        self.add(diagram_vgroup)
        self.add(find_x_text)
        self.add(not_to_scale_text)
