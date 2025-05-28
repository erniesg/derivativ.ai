from manim import *
import numpy as np

class TreeDiagramComplete(Scene):
    def construct(self):
        # Force a white background by adding a full-screen rectangle
        background_rect = Rectangle(
            width=config.frame_width,
            height=config.frame_height,
            stroke_width=0, # No border
            fill_color=WHITE,
            fill_opacity=1.0
        )
        self.add(background_rect) # Add this first

        # --- The rest of your code remains the same ---
        # config.background_color = WHITE # This can be kept or removed if using the rect
        text_color = BLACK
        line_color = BLACK
        # ... (all your existing code from "Font Size Definitions" downwards) ...
        # --- Font Size Definitions (approximating from image) ---
        fs_prob_val = 22       # For probabilities like 0.8, 0.2
        fs_main_text = 24      # For problem description, labels like Blue/White
        fs_header_num = 26     # For "17"
        fs_question_letter = 26# For "(a)"
        fs_bag_labels = 24     # For "Bag A", "Bag B"
        fs_marks = 22          # For "[2]"

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
        problem_desc_vgroup.arrange(DOWN, aligned_edge=LEFT, buff=0.12)

        problem_statement = VGroup(problem_number, problem_desc_vgroup).arrange(RIGHT, buff=0.15, aligned_edge=UP)

        question_a_letter = Text("(a)", weight=BOLD, font_size=fs_question_letter, color=text_color)
        question_a_text = Text("Complete the tree diagram.", font_size=fs_main_text, color=text_color)
        question_a_group = VGroup(question_a_letter, question_a_text).arrange(RIGHT, buff=0.15, aligned_edge=UP)

        full_header = VGroup(problem_statement, question_a_group).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        full_header.to_corner(UL, buff=0.5)
        self.add(full_header)

        # --- Tree Diagram Parameters and Probabilities ---
        x_spacing1 = 2.8
        y_spacing1 = 2.0
        x_spacing2 = 2.8
        y_spacing2 = 1.2

        prob_A_blue_val = 0.8
        prob_A_white_val = round(1 - prob_A_blue_val, 1)
        prob_B_blue_val = 0.3
        prob_B_white_val = round(1 - prob_B_blue_val, 1)

        # --- Tree Construction ---
        P0 = LEFT * 4.5 + DOWN * 0.8

        N_A_Blue = P0 + RIGHT * x_spacing1 + UP * y_spacing1
        N_A_White = P0 + RIGHT * x_spacing1 + DOWN * y_spacing1

        L_A_Blue = Line(P0, N_A_Blue, color=line_color, stroke_width=2)
        L_A_White = Line(P0, N_A_White, color=line_color, stroke_width=2)

        prob_text_offset_up = UP * 0.25 + LEFT * 0.2
        prob_text_offset_down = DOWN * 0.25 + LEFT * 0.2

        T_Prob_A_Blue = Text(f"{prob_A_blue_val:.1f}", font_size=fs_prob_val, color=text_color)
        T_Prob_A_Blue.move_to(L_A_Blue.point_from_proportion(0.45) + prob_text_offset_up)

        T_Prob_A_White = Text(f"{prob_A_white_val:.1f}", font_size=fs_prob_val, color=text_color)
        T_Prob_A_White.move_to(L_A_White.point_from_proportion(0.45) + prob_text_offset_down)

        T_Label_A_Blue = Text("Blue", font_size=fs_main_text, color=text_color).next_to(N_A_Blue, RIGHT, buff=0.3)
        T_Label_A_White = Text("White", font_size=fs_main_text, color=text_color).next_to(N_A_White, RIGHT, buff=0.3)

        N_B_Blue_from_ABlue = N_A_Blue + RIGHT * x_spacing2 + UP * y_spacing2
        N_B_White_from_ABlue = N_A_Blue + RIGHT * x_spacing2 + DOWN * y_spacing2

        L_B_Blue_from_ABlue = Line(N_A_Blue, N_B_Blue_from_ABlue, color=line_color, stroke_width=2)
        L_B_White_from_ABlue = Line(N_A_Blue, N_B_White_from_ABlue, color=line_color, stroke_width=2)

        T_Prob_B_Blue1 = Text(f"{prob_B_blue_val:.1f}", font_size=fs_prob_val, color=text_color)
        T_Prob_B_Blue1.move_to(L_B_Blue_from_ABlue.point_from_proportion(0.45) + prob_text_offset_up)
        T_Prob_B_White1 = Text(f"{prob_B_white_val:.1f}", font_size=fs_prob_val, color=text_color)
        T_Prob_B_White1.move_to(L_B_White_from_ABlue.point_from_proportion(0.45) + prob_text_offset_down)

        T_Label_B_Blue1 = Text("Blue", font_size=fs_main_text, color=text_color).next_to(N_B_Blue_from_ABlue, RIGHT, buff=0.3)
        T_Label_B_White1 = Text("White", font_size=fs_main_text, color=text_color).next_to(N_B_White_from_ABlue, RIGHT, buff=0.3)

        N_B_Blue_from_AWhite = N_A_White + RIGHT * x_spacing2 + UP * y_spacing2
        N_B_White_from_AWhite = N_A_White + RIGHT * x_spacing2 + DOWN * y_spacing2

        L_B_Blue_from_AWhite = Line(N_A_White, N_B_Blue_from_AWhite, color=line_color, stroke_width=2)
        L_B_White_from_AWhite = Line(N_A_White, N_B_White_from_AWhite, color=line_color, stroke_width=2)

        T_Prob_B_Blue2 = Text(f"{prob_B_blue_val:.1f}", font_size=fs_prob_val, color=text_color)
        T_Prob_B_Blue2.move_to(L_B_Blue_from_AWhite.point_from_proportion(0.45) + prob_text_offset_up)
        T_Prob_B_White2 = Text(f"{prob_B_white_val:.1f}", font_size=fs_prob_val, color=text_color)
        T_Prob_B_White2.move_to(L_B_White_from_AWhite.point_from_proportion(0.45) + prob_text_offset_down)

        T_Label_B_Blue2 = Text("Blue", font_size=fs_main_text, color=text_color).next_to(N_B_Blue_from_AWhite, RIGHT, buff=0.3)
        T_Label_B_White2 = Text("White", font_size=fs_main_text, color=text_color).next_to(N_B_White_from_AWhite, RIGHT, buff=0.3)

        T_BagA = Text("Bag A", font_size=fs_bag_labels, color=text_color)
        T_BagA.move_to(P0 + RIGHT * (x_spacing1 / 2) + UP * (y_spacing1 + 0.6))

        T_BagB = Text("Bag B", font_size=fs_bag_labels, color=text_color)
        bag_b_center_x = N_A_Blue[0] + x_spacing2 / 2 # N_A_Blue[0] is the x-coordinate
        T_BagB.move_to(np.array([bag_b_center_x, T_BagA.get_y(), 0]))

        tree_elements = VGroup(
            L_A_Blue, L_A_White, T_Prob_A_Blue, T_Prob_A_White, T_Label_A_Blue, T_Label_A_White,
            L_B_Blue_from_ABlue, L_B_White_from_ABlue, T_Prob_B_Blue1, T_Prob_B_White1, T_Label_B_Blue1, T_Label_B_White1,
            L_B_Blue_from_AWhite, L_B_White_from_AWhite, T_Prob_B_Blue2, T_Prob_B_White2, T_Label_B_Blue2, T_Label_B_White2,
            T_BagA, T_BagB
        )
        self.add(tree_elements)

        marks_text = Text("[2]", font_size=fs_marks, color=text_color)
        marks_text.to_corner(DR, buff=0.5)
        self.add(marks_text)
