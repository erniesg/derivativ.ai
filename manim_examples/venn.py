from manim import *
import numpy as np

class TreeDiagramComplete(Scene):
    def construct(self):
        # --- Basic Scene Setup ---
        config.background_color = WHITE
        text_color = BLACK
        line_color = BLACK

        # --- Font Size Definitions ---
        fs_prob_val = 22
        fs_main_text = 24
        fs_header_num = 26
        fs_question_letter = 26
        fs_bag_labels = 24
        fs_marks = 22
        fs_dots = 22 # Font size for "........"

        # --- Problem Statement and Question Text ---
        problem_number = Text("17", weight=BOLD, font_size=fs_header_num, color=text_color)
        problem_desc_lines = [
            "Two bags, A and B, each contain blue beads and white beads only.",
            "The probability of taking a blue bead at random from bag A is 0.8 .",
            "The probability of taking a blue bead at random from bag B is 0.3 ."
        ]
        problem_desc_vgroup = VGroup()
        for line in problem_desc_lines:
            t = Text(line, font_size=fs_main_text, color=text_color)
            problem_desc_vgroup.add(t)
        # Increased buff for better readability of problem statement
        problem_desc_vgroup.arrange(DOWN, aligned_edge=LEFT, buff=0.2)

        problem_statement = VGroup(problem_number, problem_desc_vgroup).arrange(RIGHT, buff=0.15, aligned_edge=UP)

        question_a_letter = Text("(a)", weight=BOLD, font_size=fs_question_letter, color=text_color)
        question_a_text = Text("Complete the tree diagram.", font_size=fs_main_text, color=text_color)
        question_a_group = VGroup(question_a_letter, question_a_text).arrange(RIGHT, buff=0.15, aligned_edge=UP)

        # Increased buff between problem statement and question
        full_header = VGroup(problem_statement, question_a_group).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        full_header.to_corner(UL, buff=0.6) # Slightly more buff from corner
        self.add(full_header)

        # --- Tree Diagram Parameters and Probabilities ---
        # Adjusted spacing parameters for a less cramped layout
        x_spacing1 = 3.2  # Increased horizontal distance
        y_spacing1 = 2.3  # Increased vertical spread
        x_spacing2 = 3.2  # Increased horizontal distance
        y_spacing2 = 1.5  # Increased vertical spread

        prob_A_blue_val = 0.8
        # prob_A_white_val = round(1 - prob_A_blue_val, 1) # Not shown
        prob_B_blue_val = 0.3
        # prob_B_white_val = round(1 - prob_B_blue_val, 1) # Not shown

        # --- Tree Construction ---
        # Moved starting point of the tree significantly down
        P0 = LEFT * 4.5 + DOWN * 2.0

        # Stage 1 (Bag A)
        N_A_Blue = P0 + RIGHT * x_spacing1 + UP * y_spacing1
        N_A_White = P0 + RIGHT * x_spacing1 + DOWN * y_spacing1

        L_A_Blue = Line(P0, N_A_Blue, color=line_color, stroke_width=2)
        L_A_White = Line(P0, N_A_White, color=line_color, stroke_width=2)

        # Adjusted probability text positioning
        prob_text_prop = 0.35 # Closer to the start of the branch
        prob_text_offset_up = UP * 0.3 + LEFT * 0.15
        prob_text_offset_down = DOWN * 0.3 + LEFT * 0.15

        T_Prob_A_Blue = Text(f"{prob_A_blue_val:.1f}", font_size=fs_prob_val, color=text_color)
        T_Prob_A_Blue.move_to(L_A_Blue.point_from_proportion(prob_text_prop) + prob_text_offset_up)

        # Placeholder for P(White from A)
        T_Prob_A_White = Text("........", font_size=fs_dots, color=text_color)
        T_Prob_A_White.move_to(L_A_White.point_from_proportion(prob_text_prop) + prob_text_offset_down)

        T_Label_A_Blue = Text("Blue", font_size=fs_main_text, color=text_color).next_to(N_A_Blue, RIGHT, buff=0.35)
        T_Label_A_White = Text("White", font_size=fs_main_text, color=text_color).next_to(N_A_White, RIGHT, buff=0.35)

        # Stage 2 (Bag B) - Following "Blue" from Bag A
        N_B_Blue_from_ABlue = N_A_Blue + RIGHT * x_spacing2 + UP * y_spacing2
        N_B_White_from_ABlue = N_A_Blue + RIGHT * x_spacing2 + DOWN * y_spacing2

        L_B_Blue_from_ABlue = Line(N_A_Blue, N_B_Blue_from_ABlue, color=line_color, stroke_width=2)
        L_B_White_from_ABlue = Line(N_A_Blue, N_B_White_from_ABlue, color=line_color, stroke_width=2)

        T_Prob_B_Blue1 = Text(f"{prob_B_blue_val:.1f}", font_size=fs_prob_val, color=text_color)
        T_Prob_B_Blue1.move_to(L_B_Blue_from_ABlue.point_from_proportion(prob_text_prop) + prob_text_offset_up)
        # Placeholder for P(White from B | Blue from A)
        T_Prob_B_White1 = Text("........", font_size=fs_dots, color=text_color)
        T_Prob_B_White1.move_to(L_B_White_from_ABlue.point_from_proportion(prob_text_prop) + prob_text_offset_down)

        T_Label_B_Blue1 = Text("Blue", font_size=fs_main_text, color=text_color).next_to(N_B_Blue_from_ABlue, RIGHT, buff=0.35)
        T_Label_B_White1 = Text("White", font_size=fs_main_text, color=text_color).next_to(N_B_White_from_ABlue, RIGHT, buff=0.35)

        # Stage 2 (Bag B) - Following "White" from Bag A
        N_B_Blue_from_AWhite = N_A_White + RIGHT * x_spacing2 + UP * y_spacing2
        N_B_White_from_AWhite = N_A_White + RIGHT * x_spacing2 + DOWN * y_spacing2

        L_B_Blue_from_AWhite = Line(N_A_White, N_B_Blue_from_AWhite, color=line_color, stroke_width=2)
        L_B_White_from_AWhite = Line(N_A_White, N_B_White_from_AWhite, color=line_color, stroke_width=2)

        # Placeholder for P(Blue from B | White from A)
        T_Prob_B_Blue2 = Text("........", font_size=fs_dots, color=text_color)
        T_Prob_B_Blue2.move_to(L_B_Blue_from_AWhite.point_from_proportion(prob_text_prop) + prob_text_offset_up)
        # Placeholder for P(White from B | White from A)
        T_Prob_B_White2 = Text("........", font_size=fs_dots, color=text_color)
        T_Prob_B_White2.move_to(L_B_White_from_AWhite.point_from_proportion(prob_text_prop) + prob_text_offset_down)

        T_Label_B_Blue2 = Text("Blue", font_size=fs_main_text, color=text_color).next_to(N_B_Blue_from_AWhite, RIGHT, buff=0.35)
        T_Label_B_White2 = Text("White", font_size=fs_main_text, color=text_color).next_to(N_B_White_from_AWhite, RIGHT, buff=0.35)

        # --- Bag Labels ("Bag A", "Bag B") ---
        # Position Bag labels above their respective stages with more clearance
        bag_label_y_pos = P0[1] + y_spacing1 + 0.8 # Y-coordinate for Bag A and B labels

        T_BagA = Text("Bag A", font_size=fs_bag_labels, color=text_color)
        T_BagA.move_to(np.array([P0[0] + x_spacing1 / 2, bag_label_y_pos, 0]))

        T_BagB = Text("Bag B", font_size=fs_bag_labels, color=text_color)
        # Ensure Bag B's x-position is centered on its stage
        bag_b_center_x = N_A_Blue[0] + x_spacing2 / 2
        T_BagB.move_to(np.array([bag_b_center_x, bag_label_y_pos, 0]))


        tree_elements = VGroup(
            L_A_Blue, L_A_White, T_Prob_A_Blue, T_Prob_A_White, T_Label_A_Blue, T_Label_A_White,
            L_B_Blue_from_ABlue, L_B_White_from_ABlue, T_Prob_B_Blue1, T_Prob_B_White1, T_Label_B_Blue1, T_Label_B_White1,
            L_B_Blue_from_AWhite, L_B_White_from_AWhite, T_Prob_B_Blue2, T_Prob_B_White2, T_Label_B_Blue2, T_Label_B_White2,
            T_BagA, T_BagB
        )
        self.add(tree_elements)

        # --- Marks ---
        marks_text = Text("[2]", font_size=fs_marks, color=text_color)
        marks_text.to_corner(DR, buff=0.6)
        self.add(marks_text)
