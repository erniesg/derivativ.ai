from manim import *
import numpy as np

class V1_Low_StaticDiagram(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        # Using explicit colors per element, not VMobject.set_default for this iteration
        # to match the "almost there" image style.
        default_text_color = BLACK # Labels for A, B, C, D, E, G
        angle_value_label_color = BLACK # For 70, 50, x, y, z, 30 labels

        # Configuration
        angle_label_scale = 0.65 # As per your last "good base"
        # NEW: This factor positions labels INSIDE the arc if < 1.0
        # Based on example: lbl_dist_factor_default = 0.65
        # Let's call it label_pos_factor and set it < 1.0
        label_pos_factor = 0.65
        point_label_buff = 0.25

        # Layout (from your "good base" code)
        frame_w = self.camera.frame_width
        frame_h = self.camera.frame_height
        line_start_x = -frame_w / 2 + 0.75
        line_end_x = frame_w / 2 - 0.75
        y_top = frame_h * 0.25
        y_bottom = -frame_h * 0.25

        # Points definition (from your "good base" code - these define the visual angles)
        A_pt = np.array([line_start_x, y_top, 0])
        B_pt = np.array([line_end_x, y_top, 0])
        E_pt = A_pt + (B_pt - A_pt) * 0.4

        CD_start_pt = np.array([line_start_x, y_bottom, 0])
        CD_end_pt = np.array([line_end_x, y_bottom, 0])

        C_pt_x_offset_from_E = (line_end_x - line_start_x) * 0.1
        C_pt = np.array([E_pt[0] + C_pt_x_offset_from_E, y_bottom, 0])

        G_pt_x_offset_from_C = (line_end_x - line_start_x) * 0.15
        G_pt = np.array([C_pt[0] + G_pt_x_offset_from_C, y_bottom, 0])

        if G_pt[0] > CD_end_pt[0] - 0.5: G_pt[0] = CD_end_pt[0] - 0.5
        if C_pt[0] > G_pt[0] - 0.5: C_pt[0] = G_pt[0] - 0.5
        if C_pt[0] < CD_start_pt[0] + 0.5 : C_pt[0] = CD_start_pt[0] + 0.5

        # Lines (colors to match your "almost there" image)
        line_AB_obj = Line(A_pt, B_pt, color=BLACK, stroke_width=3)
        line_CD_obj = Line(CD_start_pt, CD_end_pt, color=BLACK, stroke_width=3)
        trans_EC = Line(E_pt, C_pt, color=BLUE_C, stroke_width=2.5)
        trans_EG = Line(E_pt, G_pt, color=GREEN_C, stroke_width=2.5)

        # Helper lines for defining angles correctly
        line_EA = Line(E_pt, A_pt)
        line_EB = Line(E_pt, B_pt)
        line_CE_for_angle_at_C = Line(C_pt, E_pt)
        line_CG_for_angle_at_C = Line(C_pt, G_pt)
        line_CD_from_C_left = Line(C_pt, CD_start_pt)
        line_GE_for_angle_at_G = Line(G_pt, E_pt)
        line_GC_for_angle_at_G = Line(G_pt, C_pt)

        # Dots (Orange from image)
        dot_radius_val = 0.07
        dot_E = Dot(E_pt, color=ORANGE, radius=dot_radius_val)
        dot_C = Dot(C_pt, color=ORANGE, radius=dot_radius_val)
        dot_G = Dot(G_pt, color=ORANGE, radius=dot_radius_val)

        # Labels for points
        label_font_size = 28
        label_A = Text("A", font_size=label_font_size, color=default_text_color).next_to(A_pt, UP+LEFT, buff=point_label_buff*0.8)
        label_B = Text("B", font_size=label_font_size, color=default_text_color).next_to(B_pt, UP+RIGHT, buff=point_label_buff*0.8)
        label_E = Text("E", font_size=label_font_size, color=default_text_color).next_to(E_pt, UP, buff=point_label_buff*0.7)
        label_C_pt = Text("C", font_size=label_font_size, color=default_text_color).next_to(C_pt, DOWN, buff=point_label_buff)
        label_G_pt = Text("G", font_size=label_font_size, color=default_text_color).next_to(G_pt, DOWN, buff=point_label_buff)
        label_D_marker = Text("D", font_size=label_font_size, color=default_text_color).next_to(CD_end_pt, RIGHT, buff=point_label_buff*1.2)
        # Removed redundant C line marker

        # Parallel marks
        tick_len = 0.12; tick_offset = RIGHT * 0.05; tick_stroke_width = 2
        tick_template = Line(UP*tick_len + LEFT*tick_len, DOWN*tick_len + RIGHT*tick_len, stroke_width=tick_stroke_width, color=GRAY_BROWN)
        pm_AB_pos1 = line_AB_obj.point_from_proportion(0.40)
        pm_AB = VGroup(tick_template.copy().move_to(pm_AB_pos1), tick_template.copy().move_to(pm_AB_pos1 + tick_offset))
        pm_CD_pos1 = line_CD_obj.point_from_proportion(0.60)
        pm_CD = VGroup(tick_template.copy().move_to(pm_CD_pos1), tick_template.copy().move_to(pm_CD_pos1 + tick_offset))

        # --- ANGLES - Labels INSIDE arcs ---
        angle_stroke_width = 2.0
        # Radii for angles at E, to ensure distinction
        radius_AEC_E = 0.5
        radius_CEG_E = 0.65
        radius_CEB_E = 0.8

        # Function to place label inside the arc
        def get_internal_label_pos(vertex_coord, angle_obj, current_arc_radius, positioning_factor):
            # angle_obj.point_from_proportion(0.5) gives a point on the arc object.
            # We need a point on the virtual arc used for label positioning.
            # Simpler: Get the angle's bisector direction
            angle_val_rad = angle_obj.get_value() # Actual angle value in radians
            # Start direction is vector from vertex to start of first line of angle
            # For Angle(line1, line2), line1 starts at vertex.
            # Here, line_EA is Line(E_pt, A_pt). Vector is A_pt - E_pt.
            # Angle created is from line1 to line2. Bisector is at angle_val_rad / 2 from line1.

            # Let's use the example's method: VERTEX + normalize(ARC_MIDPOINT - VERTEX) * ARC_RADIUS * FACTOR
            # The Angle object's .point_from_proportion(0.5) is on the arc itself.
            if angle_obj.radius == 0 : return vertex_coord # Should not happen

            # Vector from vertex to the midpoint of the *actual drawn arc*
            vec_to_arc_midpoint = angle_obj.point_from_proportion(0.5) - vertex_coord

            # Position label along this bisector, at a distance = current_arc_radius * positioning_factor
            return vertex_coord + normalize(vec_to_arc_midpoint) * current_arc_radius * positioning_factor


        # Angle AEC = 70° (Red arc from image)
        angle_AEC_arc = Angle(line_EA, trans_EC, radius=radius_AEC_E, other_angle=False, color=RED_C, stroke_width=angle_stroke_width)
        label_AEC_val = MathTex("70^\\circ", color=angle_value_label_color).scale(angle_label_scale)
        label_AEC_val.move_to(get_internal_label_pos(E_pt, angle_AEC_arc, radius_AEC_E, label_pos_factor))

        # Angle CEG = 50° (Angle BETWEEN trans_EC and trans_EG)
        angle_CEG_arc = Angle(trans_EC, trans_EG, radius=radius_CEG_E, other_angle=False, color=GREEN_D, stroke_width=angle_stroke_width)
        label_CEG_val = MathTex("50^\\circ", color=angle_value_label_color).scale(angle_label_scale)
        label_CEG_val.move_to(get_internal_label_pos(E_pt, angle_CEG_arc, radius_CEG_E, label_pos_factor))

        # Angle CEB = z (Supplementary to AEC. From trans_EC to line_EB)
        angle_CEB_arc = Angle(trans_EC, line_EB, radius=radius_CEB_E, other_angle=False, color=PINK, stroke_width=angle_stroke_width)
        label_CEB_val = MathTex("z", color=angle_value_label_color).scale(angle_label_scale)
        label_CEB_val.move_to(get_internal_label_pos(E_pt, angle_CEB_arc, radius_CEB_E, label_pos_factor))


        # --- Angles at C (using radii and colors from your "almost there" image) ---
        radius_C_ECD = 0.5
        radius_C_ECG = 0.65

        angle_ECD_arc = Angle(line_CE_for_angle_at_C, line_CD_from_C_left, radius=radius_C_ECD, color=ORANGE, stroke_width=angle_stroke_width)
        label_ECD_val = MathTex("x", color=angle_value_label_color).scale(angle_label_scale)
        label_ECD_val.move_to(get_internal_label_pos(C_pt, angle_ECD_arc, radius_C_ECD, label_pos_factor))

        angle_ECG_arc = Angle(line_CG_for_angle_at_C, line_CE_for_angle_at_C, radius=radius_C_ECG, color=TEAL_C, stroke_width=angle_stroke_width)
        label_ECG_val = MathTex("y", color=angle_value_label_color).scale(angle_label_scale)
        label_ECG_val.move_to(get_internal_label_pos(C_pt, angle_ECG_arc, radius_C_ECG, label_pos_factor))

        # --- Angle at G (using radius and color from your "almost there" image) ---
        radius_G_EGC = 0.5
        angle_EGC_arc = Angle(line_GE_for_angle_at_G, line_GC_for_angle_at_G, radius=radius_G_EGC, color=PURPLE_C, stroke_width=angle_stroke_width)
        label_EGC_val = MathTex("30^\\circ", color=angle_value_label_color).scale(angle_label_scale)
        label_EGC_val.move_to(get_internal_label_pos(G_pt, angle_EGC_arc, radius_G_EGC, label_pos_factor))

        # Add all elements
        self.add(
            line_AB_obj, line_CD_obj,
            pm_AB, pm_CD,
            trans_EC, trans_EG,
            dot_E, dot_C, dot_G,
            label_A, label_B, label_E, label_C_pt, label_G_pt, label_D_marker,
            angle_AEC_arc, label_AEC_val,
            angle_CEG_arc, label_CEG_val,
            angle_CEB_arc, label_CEB_val,
            angle_ECD_arc, label_ECD_val,
            angle_ECG_arc, label_ECG_val,
            angle_EGC_arc, label_EGC_val
        )
