from manim import *
import numpy as np

class V1_Low_StaticDiagram(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        default_text_color = BLACK
        default_line_color = BLACK
        angle_value_label_color = BLACK

        numeric_angle_label_scale = 0.55
        variable_angle_label_scale = 0.65
        numeric_label_dist_factor = 0.55
        variable_label_dist_factor = 1.4
        point_label_buff = 0.25
        label_font_size = 28

        frame_w = self.camera.frame_width
        frame_h = self.camera.frame_height
        line_start_x = -frame_w / 2 + 0.75
        line_end_x = frame_w / 2 - 0.75
        y_top = frame_h * 0.25
        y_bottom = -frame_h * 0.25

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

        line_AB_obj = Line(A_pt, B_pt, color=default_line_color, stroke_width=3)
        line_CD_obj = Line(CD_start_pt, CD_end_pt, color=default_line_color, stroke_width=3)
        trans_EC = Line(E_pt, C_pt, color=BLUE_C, stroke_width=2.5)
        trans_EG = Line(E_pt, G_pt, color=GREEN_C, stroke_width=2.5)

        line_EA = Line(E_pt, A_pt)
        line_EB = Line(E_pt, B_pt)
        line_CE_for_angle_at_C = Line(C_pt, E_pt)
        line_CG_for_angle_at_C = Line(C_pt, G_pt)
        line_CD_from_C_left = Line(C_pt, CD_start_pt)
        line_GE_for_angle_at_G = Line(G_pt, E_pt)
        line_GC_for_angle_at_G = Line(G_pt, C_pt)

        dot_radius_val = 0.07
        dot_E = Dot(E_pt, color=ORANGE, radius=dot_radius_val)
        dot_C = Dot(C_pt, color=ORANGE, radius=dot_radius_val)
        dot_G = Dot(G_pt, color=ORANGE, radius=dot_radius_val)

        label_A = Text("A", font_size=label_font_size, color=default_text_color).next_to(A_pt, UP+LEFT, buff=point_label_buff*0.8)
        label_B = Text("B", font_size=label_font_size, color=default_text_color).next_to(B_pt, UP+RIGHT, buff=point_label_buff*0.8)
        label_E = Text("E", font_size=label_font_size, color=default_text_color).next_to(E_pt, UP, buff=point_label_buff*0.7)
        label_C_pt = Text("C", font_size=label_font_size, color=default_text_color).next_to(C_pt, DOWN, buff=point_label_buff)
        label_G_pt = Text("G", font_size=label_font_size, color=default_text_color).next_to(G_pt, DOWN, buff=point_label_buff)
        label_D_marker = Text("D", font_size=label_font_size, color=default_text_color).next_to(CD_end_pt, RIGHT, buff=point_label_buff*1.2)

        tick_len = 0.12; tick_offset = RIGHT * 0.05; tick_stroke_width = 2
        tick_template = Line(UP*tick_len + LEFT*tick_len, DOWN*tick_len + RIGHT*tick_len, stroke_width=tick_stroke_width, color=GRAY_BROWN)
        pm_AB_pos1 = line_AB_obj.point_from_proportion(0.40)
        pm_AB = VGroup(tick_template.copy().move_to(pm_AB_pos1), tick_template.copy().move_to(pm_AB_pos1 + tick_offset))
        pm_CD_pos1 = line_CD_obj.point_from_proportion(0.60)
        pm_CD = VGroup(tick_template.copy().move_to(pm_CD_pos1), tick_template.copy().move_to(pm_CD_pos1 + tick_offset))

        angle_stroke_width = 2.0
        radius_AEC_E = 0.5
        radius_CEG_E = 0.65
        radius_CEB_E = 0.8
        radius_C_ECD = 0.5
        radius_C_ECG = 0.65
        radius_G_EGC = 0.5

        def get_label_pos_along_bisector(vertex_pt, angle_arc_obj, arc_radius_val, dist_factor):
            vec_to_mid = angle_arc_obj.point_from_proportion(0.5) - vertex_pt
            if np.linalg.norm(vec_to_mid) < 1e-6:
                return angle_arc_obj.get_center() + UP * 0.1
            return vertex_pt + normalize(vec_to_mid) * arc_radius_val * dist_factor

        # Angle AEC = 70° (Numeric -> Inside, smaller)
        angle_AEC_arc = Angle(line_EA, trans_EC, radius=radius_AEC_E, other_angle=False, color=RED_C, stroke_width=angle_stroke_width)
        label_AEC_val = MathTex("70^\\circ", color=angle_value_label_color).scale(numeric_angle_label_scale)
        label_AEC_val.move_to(get_label_pos_along_bisector(E_pt, angle_AEC_arc, radius_AEC_E, numeric_label_dist_factor))

        # Angle CEG = 50° (Numeric -> Label outside with leader line pointing TO THE MIDDLE OF ITS ARC)
        angle_CEG_arc = Angle(trans_EC, trans_EG, radius=radius_CEG_E, other_angle=False, color=GREEN_D, stroke_width=angle_stroke_width)
        label_CEG_val = MathTex("50^\\circ", color=angle_value_label_color).scale(numeric_angle_label_scale)

        ceg_arc_target_point = angle_CEG_arc.point_from_proportion(0.5) # Midpoint ON the arc
        vec_E_to_ceg_arc_target = ceg_arc_target_point - E_pt # Vector from E to this midpoint

        label_offset_from_arc_midpoint_val = 0.4 # How far label is from arc midpoint (outwards along bisector)
        if np.linalg.norm(vec_E_to_ceg_arc_target) > 1e-6:
            label_CEG_val.move_to(ceg_arc_target_point + normalize(vec_E_to_ceg_arc_target) * label_offset_from_arc_midpoint_val)
        else:
            label_CEG_val.next_to(angle_CEG_arc, UR, buff=0.2)

        arrow_start_direction = normalize(ceg_arc_target_point - label_CEG_val.get_center())
        arrow_start_point = label_CEG_val.get_critical_point(arrow_start_direction)

        leader_CEG = Arrow(
            start=arrow_start_point,
            end=ceg_arc_target_point, # Arrow now points TO THE MIDDLE of the CEG arc
            stroke_width=1.5, color=BLACK,
            max_tip_length_to_length_ratio=0.15,
            buff=0.02
        )
        if np.linalg.norm(arrow_start_point - ceg_arc_target_point) < 0.15 :
             leader_CEG.max_tip_length_to_length_ratio = 0.1
             leader_CEG.tip.scale(0.6)

        # Angle CEB = z (Variable -> Outside)
        angle_CEB_arc = Angle(trans_EC, line_EB, radius=radius_CEB_E, other_angle=False, color=PINK, stroke_width=angle_stroke_width)
        label_CEB_val = MathTex("z", color=angle_value_label_color).scale(variable_angle_label_scale)
        label_CEB_val.move_to(get_label_pos_along_bisector(E_pt, angle_CEB_arc, radius_CEB_E, variable_label_dist_factor))

        # Angle ECD = x (Variable -> Outside)
        angle_ECD_arc = Angle(line_CE_for_angle_at_C, line_CD_from_C_left, radius=radius_C_ECD, color=ORANGE, stroke_width=angle_stroke_width)
        label_ECD_val = MathTex("x", color=angle_value_label_color).scale(variable_angle_label_scale)
        label_ECD_val.move_to(get_label_pos_along_bisector(C_pt, angle_ECD_arc, radius_C_ECD, variable_label_dist_factor))

        # Angle ECG = y (Variable -> Outside)
        angle_ECG_arc = Angle(line_CG_for_angle_at_C, line_CE_for_angle_at_C, radius=radius_C_ECG, color=TEAL_C, stroke_width=angle_stroke_width)
        label_ECG_val = MathTex("y", color=angle_value_label_color).scale(variable_angle_label_scale)
        label_ECG_val.move_to(get_label_pos_along_bisector(C_pt, angle_ECG_arc, radius_C_ECG, variable_label_dist_factor))

        # Angle EGC = 30° (Numeric -> Inside, smaller)
        angle_EGC_arc = Angle(line_GE_for_angle_at_G, line_GC_for_angle_at_G, radius=radius_G_EGC, color=PURPLE_C, stroke_width=angle_stroke_width)
        label_EGC_val = MathTex("30^\\circ", color=angle_value_label_color).scale(numeric_angle_label_scale)
        label_EGC_val.move_to(get_label_pos_along_bisector(G_pt, angle_EGC_arc, radius_G_EGC, numeric_label_dist_factor))

        self.add(
            line_AB_obj, line_CD_obj,
            pm_AB, pm_CD,
            trans_EC, trans_EG,
            dot_E, dot_C, dot_G,
            label_A, label_B, label_E, label_C_pt, label_G_pt, label_D_marker,
            angle_AEC_arc, label_AEC_val,
            angle_CEG_arc, label_CEG_val, leader_CEG,
            angle_CEB_arc, label_CEB_val,
            angle_ECD_arc, label_ECD_val,
            angle_ECG_arc, label_ECG_val,
            angle_EGC_arc, label_EGC_val
        )
