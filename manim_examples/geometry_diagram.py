from manim import *
import numpy as np

class GeometryDiagram(Scene):
    def construct(self):
        # --- Overall scaling and positioning ---
        # Adjust these values to fine-tune the appearance
        scale_factor = 1.0
        diagram_center = ORIGIN + DOWN * 0.5

        # --- Define key points ---
        E_coord = np.array([-2.5, -1, 0]) * scale_factor
        C_coord = np.array([0.5, -1, 0]) * scale_factor
        G_label_pos = np.array([2.5, -1, 0]) * scale_factor

        # --- Calculate slopes and intersection point H ---
        # Angle AEC = 46 deg. Slope of AB/CD. AB goes from top-left (B) to bottom-right (A).
        m_CD = -np.tan(46 * DEGREES)  # Slope of CD and AB
        # Angle CEH = c = 74 deg. Slope of EF. EF goes from E upwards.
        m_EF = np.tan(74 * DEGREES)   # Slope of EF

        # Intersection H of CD and EF
        # Eq EF: y - E_coord[1] = m_EF * (x - E_coord[0])
        # Eq CD: y - C_coord[1] = m_CD * (x - C_coord[0])
        # m_EF * (x - E_coord[0]) = m_CD * (x - C_coord[0])  (since E_coord[1] == C_coord[1])
        # m_EF*x - m_EF*E_coord[0] = m_CD*x - m_CD*C_coord[0]
        # (m_EF - m_CD)*x = m_EF*E_coord[0] - m_CD*C_coord[0]
        x_H = (m_EF * E_coord[0] - m_CD * C_coord[0]) / (m_EF - m_CD)
        y_H = E_coord[1] + m_EF * (x_H - E_coord[0])
        H_coord = np.array([x_H, y_H, 0])

        # --- Define lines ---
        line_EG_start = E_coord + np.array([-0.8, 0, 0]) * scale_factor
        line_EG = Line(line_EG_start, G_label_pos, stroke_width=4)

        # Line AB (through E, parallel to CD)
        # Direction vector from B to A (top-left to bottom-right)
        vec_BA_dir = normalize(np.array([1, m_CD, 0]))
        B_pt = E_coord - 1.8 * vec_BA_dir * scale_factor # B is top-left of E
        A_pt = E_coord + 2.2 * vec_BA_dir * scale_factor # A is bottom-right of E
        line_AB = Line(B_pt, A_pt, stroke_width=4).add_tip(tip_length=0.2)

        # Line CD (through C and H)
        vec_CD_dir = normalize(np.array([1, m_CD, 0])) # Direction D to C-extension
        D_pt = H_coord - 1.5 * vec_CD_dir * scale_factor # D is to the left of H
        C_ext_pt = C_coord + 0.8 * vec_CD_dir * scale_factor # Extend line CD to the right of C
        line_CD = Line(D_pt, C_ext_pt, stroke_width=4).add_tip(tip_length=0.2)

        # Line EF (through E and H)
        vec_EH_dir = normalize(H_coord - E_coord)
        F_pt = H_coord + 1.2 * vec_EH_dir * scale_factor # F is further from E than H
        line_EF = Line(E_coord, F_pt, stroke_width=4)

        # --- Add point labels ---
        E_label = MathTex("E", font_size=28).next_to(E_coord, DL, buff=0.1*scale_factor)
        C_label = MathTex("C", font_size=28).next_to(C_coord, DOWN, buff=0.15*scale_factor)
        G_label = MathTex("G", font_size=28).next_to(G_label_pos, RIGHT, buff=0.2*scale_factor)
        A_label = MathTex("A", font_size=28).next_to(A_pt, DR, buff=0.1*scale_factor)
        B_label = MathTex("B", font_size=28).next_to(B_pt, UL, buff=0.05*scale_factor)
        D_label = MathTex("D", font_size=28).next_to(D_pt, UL, buff=0.05*scale_factor)
        F_label = MathTex("F", font_size=28).next_to(F_pt, UR, buff=0.1*scale_factor)

        # --- Define points for angle calculations ---
        # These points are on the lines defining the angles, at a certain distance from the vertex
        radius_factor = 0.5 * scale_factor

        # For 46 deg angle (AEC)
        pt_on_AE_for_angle = E_coord + vec_BA_dir * radius_factor
        pt_on_EC_for_angle = E_coord + normalize(C_coord - E_coord) * radius_factor
        angle_46_arc = Angle.from_three_points(pt_on_AE_for_angle, E_coord, pt_on_EC_for_angle, radius=radius_factor)
        angle_46_label_pos = angle_46_arc.point_from_proportion(0.5) + normalize(angle_46_arc.point_from_proportion(0.5)-E_coord)*0.2*scale_factor
        angle_46_text = MathTex("46^{\circ}", font_size=24).move_to(angle_46_label_pos)

        # For c deg angle (FEC)
        pt_on_FE_for_angle = E_coord + normalize(F_pt - E_coord) * radius_factor
        angle_c_arc = Angle.from_three_points(pt_on_FE_for_angle, E_coord, pt_on_EC_for_angle, radius=radius_factor*0.9)
        angle_c_label_pos = angle_c_arc.point_from_proportion(0.5) + normalize(angle_c_arc.point_from_proportion(0.5)-E_coord)*0.2*scale_factor
        angle_c_text = MathTex("c^{\circ}", font_size=24).move_to(angle_c_label_pos)

        # For a deg angle (ECH)
        pt_on_CE_for_angle = C_coord + normalize(E_coord - C_coord) * radius_factor
        pt_on_CH_for_angle = C_coord + normalize(H_coord - C_coord) * radius_factor
        angle_a_arc = Angle.from_three_points(pt_on_CE_for_angle, C_coord, pt_on_CH_for_angle, radius=radius_factor)
        angle_a_label_pos = angle_a_arc.point_from_proportion(0.5) + normalize(angle_a_arc.point_from_proportion(0.5)-C_coord)*0.2*scale_factor
        angle_a_text = MathTex("a^{\circ}", font_size=24).move_to(angle_a_label_pos)

        # For b deg angle (HCG) - double arc
        pt_on_CG_for_angle = C_coord + normalize(G_label_pos - C_coord) * radius_factor
        angle_b_arc1 = Angle.from_three_points(pt_on_CH_for_angle, C_coord, pt_on_CG_for_angle, radius=radius_factor*0.9)
        angle_b_arc2 = Angle.from_three_points(pt_on_CH_for_angle, C_coord, pt_on_CG_for_angle, radius=radius_factor*1.0)
        angle_b_label_pos = angle_b_arc1.point_from_proportion(0.5) + normalize(angle_b_arc1.point_from_proportion(0.5)-C_coord)*0.3*scale_factor
        angle_b_text = MathTex("b^{\circ}", font_size=24).move_to(angle_b_label_pos)

        # For 60 deg angle (DHF, which is EHC)
        pt_on_HD_for_angle = H_coord + normalize(D_pt - H_coord) * radius_factor
        pt_on_HF_for_angle = H_coord + normalize(F_pt - H_coord) * radius_factor
        angle_60_arc = Angle.from_three_points(pt_on_HD_for_angle, H_coord, pt_on_HF_for_angle, radius=radius_factor)
        angle_60_label_pos = angle_60_arc.point_from_proportion(0.5) + normalize(angle_60_arc.point_from_proportion(0.5)-H_coord)*0.25*scale_factor
        angle_60_text = MathTex("60^{\circ}", font_size=24).move_to(angle_60_label_pos)

        # --- Group diagram elements ---
        diagram = VGroup(
            line_EG, line_AB, line_CD, line_EF,
            E_label, C_label, G_label, A_label, B_label, D_label, F_label,
            angle_46_arc, angle_46_text,
            angle_c_arc, angle_c_text,
            angle_a_arc, angle_a_text,
            angle_b_arc1, angle_b_arc2, angle_b_text,
            angle_60_arc, angle_60_text
        )
        diagram.move_to(diagram_center)

        # --- Add titles and text ---
        title_10 = Text("10", font_size=36).to_corner(UL, buff=0.5).shift(RIGHT*0.2)
        title_6 = Text("6", font_size=36).next_to(title_10, RIGHT, buff=4.5).align_to(title_10, UP) # Centered based on image
        title_6.move_to(UP * (title_10.get_y() - title_6.get_y() + title_6.get_center()[1])) # Align Y with 10, keep X centered

        not_to_scale = Text("NOT TO\nSCALE", font_size=28, line_spacing=0.8).to_corner(UR, buff=0.5).shift(LEFT*0.5)

        # Problem text
        text_L1 = Text("Lines AB and CD are parallel.", font_size=24)
        text_L2 = Text("EF and EG are straight lines.", font_size=24)

        text_part_a = Text("(a) Find the value of a.", font_size=24)
        text_reason = Text("Give a geometrical reason for your answer.", font_size=24)
        text_ans_format = Text(
            "a = ............................ because ....................................................................................................",
            font_size=24
        )
        text_marks = Text("[2]", font_size=24)

        problem_text_vgroup = VGroup(
            text_L1, text_L2, text_part_a, text_reason, text_ans_format
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(diagram, DOWN, buff=0.7)
        text_marks.next_to(text_ans_format, RIGHT, buff=0.2)


        # --- Display elements ---
        self.add(title_10, title_6, not_to_scale)
        self.add(diagram)
        self.add(problem_text_vgroup, text_marks)
