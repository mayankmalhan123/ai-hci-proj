
from os import environ
import pygame
import imageio

import numpy as np
from wonderwords import RandomWord
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
environ["SDL_VIDEODRIVER"] = "dummy"
import random


class KEYBOARD():
    """
    Represents a keyboard object.

    Attributes:
        key_w (int): The width of each key in mm.
        key_h (int): The height of each key in mm.
        origin (list): The origin coordinates of the keyboard.
        shift_proportion (float): The proportion of horizontal shift for keys.
        horizontal_shift (float): The horizontal shift for keys.
        horizontal_gap (int): The horizontal gap between keys.
        vertical_gap (int): The vertical gap between keys.
        keyboard_details (dict): The details of each key on the keyboard.
        current_pos (numpy.ndarray): The current position of the cursor.
        fitts_w (int): The minimum width or height of a key for Fitts' Law.
        fitts_a (int): The parameter 'a' for Fitts' Law.
        fitts_b (int): The parameter 'b' for Fitts' Law.
        alpha_x (float): The parameter 'alpha_x' for the covariance matrix.
        sigma_alpha_x (float): The parameter 'sigma_alpha_x' for the 
            covariance matrix.
        alpha_y (float): The parameter 'alpha_y' for the covariance matrix.
        sigma_alpha_y (float): The parameter 'sigma_alpha_y' for the 
            covariance matrix.
        covariance_matrix (numpy.ndarray): The covariance matrix for 
            cursor movement.
        position_history (list): The history of cursor positions.
        render_position_history (numpy.ndarray): The history of cursor positions
            for rendering.
        movement_time_history (list): The history of movement times.
        success_history (list): The history of success statuses.
        activated_key_history (list): The history of activated keys.

    Methods:
        update_covariance_matrix(): Updates the covariance matrix.
        reset_keyboard(key_w, key_h): Resets the keyboard with the given key 
            width and key height.
        type_char(target_key): Types a character on the keyboard.
        type_sentence(sentence): Types a given sentence by iterating over each
            character.
        assign_key_dim(key_dim): Assigns the dimensions of the keys on the
            keyboard and resets it.
        swap_keys(key_1, key_2): Swaps the characters associated with two keys
            on the keyboard.
        evaluate(sentence, render=False, render_duration=5): Evaluates the
            performance of the keyboard by typing a given sentence and
            calculating completion time and error rate.
        get_activated_key(landed_pos): Returns the activated key based on
            the given position.
        render(cursor_history=None, duration=6): Renders the keyboard and saves
            the rendered frames as a video.
    """

    def __init__(self, fitts_a=230, fitts_b=166,
                 alpha_x=0.0075, sigma_alpha_x=1.24,
                 alpha_y=0.0104, sigma_alpha_y=1.12,
                 key_w=30, key_h=30):
        self.key_w = key_w
        self.key_h = key_h
        self.origin = [0,0]
        self.shift_proportion = 0.3
        self.horizontal_shift = key_w * self.shift_proportion 
        self.horizontal_gap = 2
        self.vertical_gap = 2
        self.keyboard_details = self.get_initial_keyboard()
        self.current_pos = np.array([self.keyboard_details[' ']['x'],
                                     self.keyboard_details[' ']['y'] + 100])
        self.fitts_w = min(key_w, key_h)

        ### DEFAULT USER PARAMETER VALUES FROM LITERATURE
        # For Fitts' Law: https://www.yorku.ca/mack/p219-mackenzie.pdf
        # For typing error: https://dl.acm.org/doi/pdf/10.1145/2984511.2984546
        #              and  https://dl.acm.org/doi/pdf/10.1145/2501988.2502058
        self.fitts_a = fitts_a
        self.fitts_b = fitts_b
        self.alpha_x = alpha_x
        self.sigma_alpha_x = sigma_alpha_x
        self.alpha_y = alpha_y
        self.sigma_alpha_y = sigma_alpha_y

        self.covariance_matrix = self.update_covariance_matrix()

        self.position_history = []
        self.render_position_history = []
        self.movement_time_history = []
        self.success_history = []
        self.activated_key_history = []

    def fitts_law(self,x1, y1, x2, y2, width):
        #distance : key A to key B
        distance = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
        index_of_difficulty = np.log((distance / width) + 1)
        time = self.fitts_a + self.fitts_b * index_of_difficulty
        return time

    def shift_fun(self,shift_prop, horizontal_shift, width):
        return shift_prop * width + horizontal_shift

    def key_overlap(self,x_ij, y_ij, w_i, w_j, h_i, h_j):
        return np.max(0, x_ij - (w_i/2) - (w_j/2)) + np.max(0, y_ij - (h_i/2) - (h_j/2))

    def weight_cost_function(self, x1, y1, x2, y2, width, shift_prop, horizontal_shift):
        o1 = self.shift_fun(shift_prop, horizontal_shift, width) 
        o2 = self.fitts_law(x1, y1, x2, y2, width)
        # o3 = self.key_overlap(x_ij, y_ij, w_i, w_j, h_i, h_j)
        
        a, b = .5, .5
        s = a * o1 + b * o2 
        return s

    def random_search(self, events = 20):
        
        print("startintg random search")
        best_cost = float('inf')  # Assuming we are looking for the minimum value.
        best_solution = None

        for _ in range(events):
            x1 = random.choice([17.0, 47.0,77.0,107.0,137.0,167.0,197.0,227.0,257.0,287.0,26.0,56.0,86.0,116.0,146.0,176.0,206.0,236.0,266.0,35.0,65.0,95.0,125.0,155.0,185.0,215.0,245.0,275.0,44.0])
            y1 = random.choice([15.0, 47.0, 79.0, 11.0])
            x2 = random.choice([17.0, 47.0,77.0,107.0,137.0,167.0,197.0,227.0,257.0,287.0,26.0,56.0,86.0,116.0,146.0,176.0,206.0,236.0,266.0,35.0,65.0,95.0,125.0,155.0,185.0,215.0,245.0,275.0,44.0])
            w  = random.choice([20.,25.,30.,35.,40.])
            y2 = random.choice([15.0, 47.0, 79.0, 11.0])
            shift_prop = random.choice([.2, .25, .3, .35])
            horiz_shift =  random.choice([2, 3, 4])
            
            current_cost = self.weight_cost_function(x1, y1, x2, y2, w, shift_prop, horiz_shift)
            best_solution = [x1, y1, x2, y2, w, shift_prop, horiz_shift]
            
            if current_cost < best_cost:
                best_cost = current_cost

        print(best_cost, best_solution)
        return best_cost, best_solution
        
    def exhaustive_search(self):

        print("startintg exhaustive search")
        best_cost = float('inf')  # Assuming we are looking for the minimum value.
        best_solution = None

        # Iterate over each dimension according to constraints and intervals.
        for x1 in [17.0, 47.0,77.0,107.0,137.0,167.0,197.0,227.0,257.0,287.0,26.0,56.0,86.0,116.0,146.0,176.0,206.0,236.0,266.0,35.0,65.0,95.0,125.0,155.0,185.0,215.0,245.0,275.0,44.0]:
            for y1 in [15.0, 47.0, 79.0, 11.0]:
                for x2 in [17.0, 47.0,77.0,107.0,137.0,167.0,197.0,227.0,257.0,287.0,26.0,56.0,86.0,116.0,146.0,176.0,206.0,236.0,266.0,35.0,65.0,95.0,125.0,155.0,185.0,215.0,245.0,275.0,44.0]:
                    for w in [20.,25.,30.,35.,40.]:
                        for y2 in [15.0, 47.0, 79.0, 11.0]:
                            for shift_prop in [.2, .25, .3, .35]:
                                for horiz_shift in [2, 3, 4]:
                                    current_cost = self.weight_cost_function(x1, y1, x2, y2, w, shift_prop, horiz_shift)
                                    best_solution = [x1, y1, x2, y2, w, shift_prop, horiz_shift]
                                    if current_cost < best_cost:
                                        best_cost = current_cost

        print(best_cost, best_solution)
        return best_cost, best_solution

    def compare_search(self):
        b1, bs1 = self.exhaustive_search() 
        b2, bs2 = self.random_search()

        print(f"Exhaustive search score : {b1}")
        print(f"Random search score     : {b2}")  
        

    def update_covariance_matrix(self):
        """
            Updates the covariance matrix based on the current values of alpha, 
            key width, and key height.

            Returns:
                numpy.ndarray: The updated covariance matrix.
        """
        return np.array([
            [self.alpha_x * self.key_w**2 + self.sigma_alpha_x, 0],
            [0, self.alpha_y * self.key_h**2 + self.sigma_alpha_y]
            ])


    def reset_keyboard(self, key_w, key_h):
        """
        Resets the keyboard with the given key width and key height.

        Args:
            key_w (int): The width of each key in mm.
            key_h (int): The height of each key in mm.

        Returns:
            None
        """
        self.key_w = key_w
        self.key_h = key_h
        self.horizontal_shift = key_w * self.shift_proportion
        self.keyboard_details = self.get_initial_keyboard()
        self.current_pos = np.array([self.keyboard_details[' ']['x'],
                                     self.keyboard_details[' ']['y'] + 100])
        self.position_history = [self.current_pos]
        self.movement_time_history = []
        self.success_history = []


    def type_char(self, target_key):
        """
        Types a character on the keyboard.

        Args:
            target_key (str): The key to be typed.

        Returns:
            tuple: A tuple containing the movement time (float), success 
                   status (bool), activated key (str), and new 
                   position (numpy.ndarray) after typing.
        """
        ## Fitts' law from https://www.yorku.ca/mack/p219-mackenzie.pdf
        target_pos = np.array([
            self.keyboard_details[target_key]['x'] +
            1/2 * self.keyboard_details[target_key]['width'],
            self.keyboard_details[target_key]['y'] +
            1/2 * self.keyboard_details[target_key]['height']
            ])
        dist = np.linalg.norm(target_pos - self.current_pos)
        fitts_id = np.log2(dist/self.fitts_w + 1)
        movement_time = self.fitts_a + self.fitts_b * fitts_id
        new_pos = np.random.multivariate_normal(target_pos,
                                                self.covariance_matrix)
        activated_key = self.get_activated_key(new_pos)
        success = activated_key == target_key
        self.current_pos = new_pos
        return movement_time, success, activated_key, new_pos


    def type_sentence(self, sentence):
        """
        Types a given sentence by iterating over each character.

        Args:
            sentence (str): The sentence to be typed.

        Returns:
            None
        """
        for char in sentence:
            mov_time, success, activated_key, new_pos = self.type_char(char)
            self.movement_time_history.append(mov_time)
            self.success_history.append(success)
            self.position_history.append(new_pos)
            self.activated_key_history.append(activated_key)
        self.render_position_history = np.array(self.position_history)


    def assign_key_dim(self, key_dim):
        """
        Assigns the dimensions of the keys on the keyboard and resets it.

        Args:
            key_dim (tuple): The dimensions of the keys (width, height)

        Returns:
            None
        """
        self.reset_keyboard(key_dim[0], key_dim[1])


    def swap_keys(self, key_1, key_2):
        """
        Swaps the characters associated with two keys on the keyboard.

        Args:
            key_1 (str): The character associated with the first key.
            key_2 (str): The character associated with the second key.

        Returns:
            None
        """
        for _, details in self.keyboard_details.items():
            if details['char'] == key_1:
                details['char'] = key_2
            elif details['char'] == key_2:
                details['char'] = key_1


    def evaluate(self, sentence=None, render=False, render_duration=5):
        """
        Evaluates the performance of the keyboard by typing a given sentence 
        and calculating completion time and error rate.

        Args:
            sentence (str): The sentence to be typed.
            render (bool, optional): Whether to render the typing process.
            render_duration (int, optional): The duration (in seconds) to render
                the typing process. Defaults to 5.

        Returns:
            tuple: A tuple containing the completion time (s) and error rate.
        """
        if sentence is None:
            sentence=get_target_sentence_wonderwords()
        self.type_sentence(sentence)
        comp_time = np.sum(self.movement_time_history)/len(sentence) / 1000
        error_rate = 1 - np.sum(self.success_history)/len(sentence)

        if render:
            self.render(cursor_history=self.render_position_history,
                        duration=render_duration)
        self.reset_keyboard(self.key_w, self.key_h)
        return comp_time, error_rate


    def get_activated_key(self, landed_pos):
        """
        Returns the activated key based on the given  position.

        Args:
        - landed_pos (tuple): The position the user pressed on the keyboard.

        Returns:
        - char (str): The character associated with the activated key.
        - None: If no key is activated.

        """
        for _, details in self.keyboard_details.items():
            x_pos = details['x']
            y_pos = details['y']
            if (x_pos <= landed_pos[0] <= x_pos + details['width'] and
                y_pos <= landed_pos[1] <= y_pos + details['height']):
                return details['char']
        return None


    def render(self, cursor_history=None, duration=6, mm_to_px_ratio=4.0):
        """
        Renders the keyboard and saves the rendered frames as a video.

        Args:
            cursor_history (list, optional): A list of cursor positions 
                over time.
            duration (int, optional): The duration of the rendering in seconds.
        """
        print("== Rendering keyboard ==")
        pygame.init()

        # Create a Pygame window
        window_size = (1200, 608)
        screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption('keyboard')

        refresh_rate = 5

        writer = imageio.get_writer('keyboard_typing.mp4', fps=refresh_rate)

        font = pygame.font.Font(None, 24)

        surfaces = []
        rendered_chars = []
        text_rects = []
        rects = []

        cursor_radius = 10
        cursor_thick = 5

        for _, value in self.keyboard_details.items():
            surfaces.append(pygame.Surface((value['width'] * mm_to_px_ratio,
                                            value['height'] * mm_to_px_ratio)))
            rendered_chars.append(font.render(value['char'], True, (0, 0, 0)))
            text_rects.append(rendered_chars[-1].get_rect(
                center=(surfaces[-1].get_width()/2,
                        surfaces[-1].get_height()/2)
                        ))
            rects.append(pygame.Rect(value['x'] * mm_to_px_ratio, value['y']
                                     * mm_to_px_ratio, value['width'] *
                                     mm_to_px_ratio, value['height'] *
                                     mm_to_px_ratio))

        clock_count = 0
        while clock_count < duration * refresh_rate:
            screen.fill((40, 40, 40))
            cursor_index = -1
            if cursor_history is not None:
                cursor_index = min(len(cursor_history)-1,
                                   int(clock_count/refresh_rate))
                current_cur_position = [
                    cursor_history[cursor_index][0] * mm_to_px_ratio,
                    cursor_history[cursor_index][1] * mm_to_px_ratio
                    ]
            else:
                current_cur_position = [0,0]


            for i, surface in enumerate(surfaces):
                target_left = rects[i].x
                target_right = rects[i].x + rects[i].width
                target_top = rects[i].y
                target_bottom = rects[i].y + rects[i].height
                if (current_cur_position[0] > target_left
                    and current_cur_position[0] < target_right
                    and current_cur_position[1] < target_bottom
                    and current_cur_position[1] > target_top):
                    pygame.draw.rect(surface, (127, 255, 212),
                                     (1, 1, surface.get_width()-2,
                                      surface.get_height()-2))
                else:
                    pygame.draw.rect(surface, (255, 255, 255),
                                     (1, 1, surface.get_width()-2,
                                      surface.get_height()-2))
                # Show the button text
                surface.blit(rendered_chars[i], text_rects[i])

                # Draw the button on the screen
                screen.blit(surface, (rects[i].x, rects[i].y))

            if cursor_history is not None:
                pygame.draw.circle(screen, (255,0,0),
                                   (current_cur_position[0]-cursor_radius/2,
                                    current_cur_position[1]-cursor_radius/2),
                                    cursor_radius, cursor_thick)

            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            writer.append_data(frame.transpose(1, 0, 2))

            pygame.display.flip()

            clock_count += 1
        writer.close()
        pygame.quit()


    def get_initial_keyboard(self):
        """
        Returns a dictionary with the keyboard configuration.
        
        Returns: dict with keyboard configuration
        """
        def calculate_position(row, index, shift=0):
            x = self.origin[0] + shift + (index * self.key_w +
                                          self.horizontal_gap) + self.key_w / 2
            y = self.origin[1] + (row * (self.key_h + self.vertical_gap)
                                  + self.key_h / 2)
            return x, y

        rows = {
            0: "qwertyuiop",
            1: "asdfghjkl",
            2: "zxcvbnm,.",
            3: " "
        }
        keyboard = {}

        for row, keys in rows.items():
            shift = self.horizontal_shift * row
            for index, char in enumerate(keys):
                x, y = calculate_position(row, index, shift)
                width = self.key_w if char != " " else 8 * self.key_w
                keyboard[char] = {
                    'char': char,
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': self.key_h
                }

        return keyboard


def get_target_sentence_wonderwords(char_limit=500):
    """""""""
    Generate a sentence with a random number of characters.
    :param char_limit: The maximum number of characters in the sentence.
    :return: A string of random words.
    """""""""
    r = RandomWord()
    char_count = 0
    temp_length = 10
    final_sentence = ""
    while char_count <= char_limit:
        this_sentence = r.random_words(temp_length)
        this_sentence = " ".join(this_sentence).replace("-", " ")
        this_sentence = this_sentence.replace("'","").replace("Ã±","n").lower()
        char_count += len(this_sentence)
        final_sentence += this_sentence
    return final_sentence[0:char_limit]


