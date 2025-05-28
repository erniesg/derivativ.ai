from manim import *
import numpy as np

class GeometryDiagram(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        VMobject.set_default(color=BLACK)
        Tex.set_default(color=BLACK)
        Text.set_default(color=BLACK)

        scale_factor = 0.9 # REDUCED scale_factor for better text visibility
        diagram_center = ORIGIN + UP * 0.5 # SHIFTED diagram UP for text visibility

        # --- Define Key Points and Angles ---
        E_coord = np.array([-2.5, -1, 0]) * scale_factor
        C_coord = np.array([0.5, -1, 0]) * scale_factor

        angle_AEC_val = 46 * DEGREES
        angle_EHC_val = 60 * DEGREES

        angle_a_val = angle_AEC_val
        angle_c_val = PI - angle_a_val - angle_EHC_val

        # --- Calculate H_coord ---
        len_EC = np.linalg.norm(C_coord - E_coord)
        len_EH = (len_EC / np.sin(angle_EHC_val)) * np.sin(angle_a_val)
        vec_EC_unit = normalize(C_coord - E_coord)
        dir_EH = np.array([np.cos(angle_c_val), np.sin(angle_c_val), 0])
        H_coord = E_coord + len_EH * dir_EH

        main_line_stroke_width = 2 * (scale_factor/1.0) # Scale stroke width

        # --- Define End Points for Lines (A, B, D, F, G) ---
        G_label_pos = C_coord + vec_EC_unit * 2.3 * scale_factor
        line_EG = Line(E_coord - vec_EC_unit * 0.8 * scale_factor, G_label_pos, stroke_width=main_line_stroke_width)

        angle_EA_from_EC = -angle_AEC_val
        vec_EA_dir = normalize(np.array([np.cos(angle_EA_from_EC), np.sin(angle_EA_from_EC), 0]))
        A_pt = E_coord + vec_EA_dir * 2.0 * scale_factor
        B_pt = E_coord - vec_EA_dir * 1.8 * scale_factor
        line_AB = Line(B_pt, A_pt, stroke_width=main_line_stroke_width)

        vec_CH_dir = normalize(H_coord - C_coord)
        D_pt = H_coord + vec_CH_dir * 1.5 * scale_factor
        line_CD_start_for_draw = C_coord - vec_CH_dir * 0.5
        line_CD = Line(D_pt, line_CD_start_for_draw, stroke_width=main_line_stroke_width)

        vec_EH_dir = normalize(H_coord - E_coord)
        F_pt = H_coord + vec_EH_dir * 1.2 * scale_factor
        line_EF_start_pt = E_coord - vec_EH_dir * 0.5 * scale_factor
        line_EF = Line(line_EF_start_pt, F_pt, stroke_width=main_line_stroke_width)

        # --- Parallel Arrows ---
        arrow_sw = main_line_stroke_width;
        arrow_len = 0.25 * scale_factor;
        arrow_tip_len = 0.12 * scale_factor

        # Arrow on line AB: near A, pointing towards B (direction is -vec_EA_dir)
        vec_AB_points_to_B_dir = -vec_EA_dir
        arrow_AB_start_near_A = A_pt + vec_AB_points_to_B_dir * 0.3 * scale_factor # Start near A, move slightly towards B
        arrow_AB_end_near_A = arrow_AB_start_near_A + vec_AB_points_to_B_dir * arrow_len
        parallel_arrow_AB = Arrow(arrow_AB_start_near_A, arrow_AB_end_near_A,
                                  stroke_width=arrow_sw, buff=0, tip_length=arrow_tip_len, tip_shape=ArrowTriangleFilledTip)

        # Arrow on line CD: near C, pointing towards D (direction is vec_CH_dir or vec_CD_points_to_D_dir)
        vec_CD_points_to_D_dir = normalize(D_pt - C_coord) # Direction from C to D
        arrow_CD_start_near_C = C_coord + vec_CD_points_to_D_dir * 0.3 * scale_factor # Start near C, move slightly towards D
        arrow_CD_end_near_C = arrow_CD_start_near_C + vec_CD_points_to_D_dir * arrow_len
        parallel_arrow_CD = Arrow(arrow_CD_start_near_C, arrow_CD_end_near_C,
                                  stroke_width=arrow_sw, buff=0, tip_length=arrow_tip_len, tip_shape=ArrowTriangleFilledTip)

        # --- Point Labels ---
        label_fs = 28 * (scale_factor/1.0) # Scale font size
        E_lbl = MathTex("E", font_size=label_fs).next_to(E_coord, DL, buff=0.1*scale_factor)
        C_lbl = MathTex("C", font_size=label_fs).next_to(C_coord, DOWN + LEFT * 0.1, buff=0.15*scale_factor)
        G_lbl_txt = MathTex("G", font_size=label_fs).next_to(G_label_pos, RIGHT, buff=0.15*scale_factor)
        A_lbl = MathTex("A", font_size=label_fs).next_to(A_pt, DR, buff=0.1*scale_factor)
        B_lbl = MathTex("B", font_size=label_fs).next_to(B_pt, UL, buff=0.05*scale_factor)
        D_lbl = MathTex("D", font_size=label_fs).next_to(D_pt, UL, buff=0.05*scale_factor)
        F_lbl = MathTex("F", font_size=label_fs).next_to(F_pt, UR, buff=0.1*scale_factor)

        # --- Angle Arcs and Labels ---
        arc_r_base = 0.5 * scale_factor;
        arc_sw_val = 1.5 * (scale_factor/1.0); # Scale stroke width
        angle_txt_fs_val = 24 * (scale_factor/1.0); # Scale font size
        lbl_dist_factor_default = 0.65

        angle_46_arc = Angle.from_three_points(A_pt, E_coord, C_coord, radius=arc_r_base, stroke_width=arc_sw_val, other_angle=False)
        angle_46_lbl_pos = E_coord + normalize(angle_46_arc.point_from_proportion(0.5)-E_coord) * arc_r_base * lbl_dist_factor_default
        angle_46_txt = MathTex("46^{\circ}", font_size=angle_txt_fs_val).move_to(angle_46_lbl_pos)

        angle_c_r_val = arc_r_base * 0.85
        angle_c_arc = Angle.from_three_points(C_coord, E_coord, H_coord, radius=angle_c_r_val, stroke_width=arc_sw_val, other_angle=False)
        angle_c_lbl_pos = E_coord + normalize(angle_c_arc.point_from_proportion(0.5)-E_coord) * angle_c_r_val * lbl_dist_factor_default * 1.05
        angle_c_txt = MathTex("c^{\circ}", font_size=angle_txt_fs_val).move_to(angle_c_lbl_pos)

        angle_a_arc = Angle.from_three_points(E_coord, C_coord, H_coord, radius=arc_r_base, stroke_width=arc_sw_val, other_angle=True)
        angle_a_lbl_pos = C_coord + normalize(angle_a_arc.point_from_proportion(0.5)-C_coord) * arc_r_base * lbl_dist_factor_default
        angle_a_txt = MathTex("a^{\circ}", font_size=angle_txt_fs_val).move_to(angle_a_lbl_pos)

        angle_b_radius_val = arc_r_base * 0.90
        angle_b_arc = Angle.from_three_points(H_coord, C_coord, G_label_pos, radius=angle_b_radius_val, stroke_width=arc_sw_val, other_angle=True)
        angle_b_lbl_pos = C_coord + normalize(angle_b_arc.point_from_proportion(0.5)-C_coord) * angle_b_radius_val * lbl_dist_factor_default * 0.9
        angle_b_txt = MathTex("b^{\circ}", font_size=angle_txt_fs_val).move_to(angle_b_lbl_pos)

        angle_60_lbl_dist_factor = 0.8
        angle_60_arc = Angle.from_three_points(E_coord, H_coord, C_coord, radius=arc_r_base, stroke_width=arc_sw_val, other_angle=False)
        angle_60_lbl_pos = H_coord + normalize(angle_60_arc.point_from_proportion(0.5)-H_coord) * arc_r_base * angle_60_lbl_dist_factor
        angle_60_txt = MathTex("60^{\circ}", font_size=angle_txt_fs_val).move_to(angle_60_lbl_pos)

        # --- Group and Add Elements ---
        diagram_elems = [
            line_EG, line_AB, line_CD, line_EF,
            parallel_arrow_AB, parallel_arrow_CD,
            E_lbl, C_lbl, G_lbl_txt, A_lbl, B_lbl, D_lbl, F_lbl,
            angle_46_arc, angle_46_txt, angle_c_arc, angle_c_txt,
            angle_a_arc, angle_a_txt,
            angle_b_arc, angle_b_txt,
            angle_60_arc, angle_60_txt
        ]
        diagram = VGroup(*diagram_elems).move_to(diagram_center)

        txt_fs_val = 24 * (scale_factor/1.0); # Scale font size
        ttl_fs_val = 36 * (scale_factor/1.0);
        nts_fs_val = 20 * (scale_factor/1.0);

        title_10 = Text("10", font_size=ttl_fs_val).to_corner(UL, buff=0.5*scale_factor).shift(RIGHT*0.2*scale_factor)
        not_to_scale = Text("NOT TO\nSCALE", font_size=nts_fs_val, line_spacing=0.8).to_corner(UR, buff=0.5*scale_factor).shift(LEFT*0.5*scale_factor)

        txt_L1 = Text("Lines AB and CD are parallel.", font_size=txt_fs_val)
        txt_L2 = Text("EF and EG are straight lines.", font_size=txt_fs_val)
        txt_part_a = Text("(a) Find the value of a.", font_size=txt_fs_val)
        txt_reason = Text("Give a geometrical reason for your answer.", font_size=txt_fs_val)
        txt_ans_fmt_str = "a = " + "."*28 + " because " + "."*70
        txt_ans_fmt = Text(txt_ans_fmt_str, font_size=txt_fs_val, font="Consolas")
        txt_marks = Text("[2]", font_size=txt_fs_val)

        q_block = VGroup(txt_part_a, txt_reason).arrange(DOWN, aligned_edge=LEFT, buff=0.1*scale_factor)
        prob_txt_vg = VGroup(txt_L1, txt_L2, q_block, txt_ans_fmt).arrange(DOWN, aligned_edge=LEFT, buff=0.25*scale_factor).next_to(diagram, DOWN, buff=0.8*scale_factor) # Increased buff
        txt_marks.next_to(txt_ans_fmt, RIGHT, buff=0.3*scale_factor)

        self.add(title_10, not_to_scale)
        self.add(diagram)
        self.add(prob_txt_vg, txt_marks)
