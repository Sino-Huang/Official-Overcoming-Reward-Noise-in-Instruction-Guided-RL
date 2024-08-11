from typing import Any
import gymnasium as gym
import numpy as np
import pygame
from icecream import ic
import pickle 
import pandas as pd
import os
from pathlib import Path
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize as th_f_resize
import torch as th

TRAJ_LENGTH_LIMIT = 10 
AUTO_ACTIVATE_INPUT = True 
RECORD_WALKTHROUGH = False 
CLIP_SIZE = 224
ENV_SIZE = 64

class MontezumaInfoWrapper(gym.Wrapper):
    """apply this before FourOutputWrapper
    goal_seq_df cols = ['goal', 'room', 'x', 'y'], it is a game walkthrough df 
    """
    def __init__(self, env, goal_seq_df=None ):
        super().__init__(env)
        
        # static
        self.goal_seq_df = goal_seq_df
        if self.goal_seq_df is not None:
            self.total_goal_seq = self.goal_seq_df['goal'].values.tolist()
            self.total_goal_num = len(self.total_goal_seq)
            
            # dynamic
            self.cur_goal_index = 0
            self.cur_goal = self.total_goal_seq[self.cur_goal_index]
            self.completed_goals = []
        
    def reset_goal_seq_info(self):
        self.cur_goal_index = 0
        self.cur_goal = self.total_goal_seq[self.cur_goal_index]
        self.completed_goals = []
        
    def step(self, action: Any):
        obs, reward, done, truncated, info = self.env.step(action)
        # add agent_pos_info
        info['agent_pos'] = self.agent_pos()
        # add room info 
        info['room'] = self.room()
        
        # add goal info 
        if self.goal_seq_df is not None:
            info['total_goal_seq'] = self.total_goal_seq
            info['total_goal_num'] = self.total_goal_num
            # ! check if the goal is reached
            goal_x = int(self.goal_seq_df.iloc[self.cur_goal_index]['x'])
            goal_y = int(self.goal_seq_df.iloc[self.cur_goal_index]['y'])
            
            if self.reached_pos(goal_x, goal_y):
                self.completed_goals.append(self.cur_goal)
                self.cur_goal_index += 1
                self.cur_goal = self.total_goal_seq[self.cur_goal_index]

        
            info['cur_goal'] = self.cur_goal   
            info['cur_goal_index'] = self.cur_goal_index
        
        # if lives being 0, reset the goal seq info
        if info['lives'] == 0:
            self.reset_goal_seq_info()
            
        return obs, reward, done, truncated, info
        
    def agent_pos(self):
        ale = self.env.get_wrapper_attr('ale')
        x, y = ale.getRAM()[42:44]
        return int(x), int(y)
    
    def room(self):
        ale = self.env.get_wrapper_attr('ale')
        return int(ale.getRAM()[3])
    
    def reached_pos(self, x_, y_):
        x, y = self.agent_pos()
        return (x_ - 5 <= x <= x_ + 5) and (y_ - 5 <= y <= y_ + 5)
    
    def save_state(self, filename):
        state = self.env.clone_full_state()
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        print('File written : {}'.format(filename))
    
    def load_state(self, filename):
        with open(filename, 'rb') as f:
            state_bak = pickle.load(f)
        self.env.restore_full_state(state_bak)
        obs, *_ = self.step(0)
        return obs
    

def main():
    def wrap_text(text, font, max_width):
        """Wraps text into multiple lines so it fits within the specified width."""
        words = text.split(" ")
        lines = []
        current_line = ""

        for word in words:
            # Check if adding this word would exceed the width
            test_line = current_line + word + " "
            # Measure the line width
            line_width, _ = font.size(test_line)
            if line_width > max_width:
                # If adding the word exceeds the width, start a new line
                lines.append(current_line.rstrip())
                current_line = word + " "
            else:
                # Otherwise, add the word to the current line
                current_line += word + " "

        if current_line:
            # Add the last line
            lines.append(current_line.rstrip())
        return lines

    def pygame_display_image(screen, obs, x=0, y=0):
        """Display the image on the screen"""
        nonlocal screen_width, screen_height
        # swap the axes to match the screen coordinate system
        rgb_img = obs.transpose(1, 0, 2)
        img = pygame.surfarray.make_surface(rgb_img)
        # scale the image to fit the screen
        img = pygame.transform.scale(
            img, (int(screen_width * 0.8), int(screen_height * 0.8))
        )
        screen.blit(img, (x, y))  # draw the image on the screen

    def pygame_display_text(screen, text, x=0, y=0, color=(255, 255, 255)):
        """Render text on the screen"""
        nonlocal font, font_size, screen_width
        lines = text.split("\n")
        # wrap the text
        wrap_lines = []
        for i, line in enumerate(lines):
            new_lines = wrap_text(line, font, screen_width - x)
            wrap_lines.extend(new_lines)
        for i, line in enumerate(wrap_lines):
            text_surface = font.render(line, True, color)
            screen.blit(text_surface, (x, y + i * font_size))

    def pygame_action(
        event,
    ):
        if event.key == pygame.K_0:
            action = 0
            is_fire = True
        elif event.key == pygame.K_SPACE:
            action = 1
            is_fire = True
        elif event.key == pygame.K_w:
            action = 2
            is_fire = True
        elif event.key == pygame.K_d:
            action = 3
            is_fire = True
        elif event.key == pygame.K_a:
            action = 4
            is_fire = True
        elif event.key == pygame.K_s:
            action = 5
            is_fire = True
        elif event.key == pygame.K_e:
            action = 14
            is_fire = True
        elif event.key == pygame.K_q:
            action = 15
            is_fire = True
        else:
            action = None
            is_fire = False

        return action, is_fire

    def reset_env():
        nonlocal env
        obs, _ = env.reset()
        return obs


    def handle_notes_input(event):
        nonlocal notes_text
        if event.key == pygame.K_BACKSPACE:
            notes_text = notes_text[:-1]
        else:
            notes_text += event.unicode

    def display_notes(screen, text, box):
        pygame.draw.rect(screen, (200, 200, 200), box, 0)  # Draw the background of the note box
        pygame.draw.rect(screen, (0, 0, 0), box, 2)  # Draw the border of the note box
        txt_surface = font.render(text, True, (0, 0, 0))
        screen.blit(txt_surface, (box.x + 5, box.y + 5))

    env_name = "ALE/MontezumaRevenge-v5"
    save_dir = os.path.join(os.environ['PWD'], "data/03_traj_instr_pairs/montezuma")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    save_state_path = os.path.join(save_dir, "montezuma_state.pkl")
    
    if not RECORD_WALKTHROUGH: # we will test if the game walkthrough is already recorded
        walk_through_csv_path = os.path.join(save_dir, "game_walkthrough.csv")
        if os.path.exists(walk_through_csv_path):
            goal_seq_df = pd.read_csv(walk_through_csv_path)
    else:
        goal_seq_df = None

    env = MontezumaInfoWrapper(gym.make(env_name), goal_seq_df=goal_seq_df)
    
        
    env_height = env.observation_space.shape[0]
    env_width = env.observation_space.shape[1]

    w_h_ratio = env_height / env_width

    pygame.init()
    # setup the screen
    screen_width = 500
    screen_height = screen_width * w_h_ratio

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption(f"Environment: {env_name}")
    # setup the font
    pygame.font.init()
    # font size based on the screen height
    font_size = int(screen_height * 0.03)
    font = pygame.font.SysFont("Noto Sans", font_size)

    

    obs = reset_env()
    if os.path.exists(save_state_path):
        obs = env.load_state(save_state_path)

    rew = None 
    info = None 
    action = 0 
    clock = pygame.time.Clock()  # create a clock object to control the fps

    # Note taking variables
    notes_active = False
    notes_text = ''
    notes_box = pygame.Rect(50, screen_height - 100, screen_width - 100, 90)  # Adjust size as needed



    if os.path.exists(os.path.join(save_dir, "instr.csv")):
        instruction_df = pd.read_csv(os.path.join(save_dir, "instr.csv"))
        data_id = len(instruction_df)

        video_path = os.path.join(save_dir, "expert_traj_chunk_0.pkl")
        with open(video_path, "rb") as f:
            video_lst = pickle.load(f)
            # make it to list 
            video_lst = [video for video in video_lst]
    else:
        instruction_df = pd.DataFrame(columns=['data_id','instruction','trajectory_chunk_file','trajectory_local_idx'])
        data_id = 0 
        video_lst = []
        
    trajectory_chunk_file = 'montezuma/expert_traj_chunk_0.pkl'
    
    # ! record the game walkthrough
    if os.path.exists(os.path.join(save_dir, "game_walkthrough.csv")):
        game_walkthrough_df = pd.read_csv(os.path.join(save_dir, "game_walkthrough.csv"))
    else:
        game_walkthrough_df = pd.DataFrame(columns=['goal', 'room', 'x', 'y'])
    

    running = True
    is_recording = False 
    cur_room = None 
    
    def activate_notes():
        nonlocal notes_active
        nonlocal notes_text
        notes_active = True
        notes_text = ""
    try:
        while running:
            # handle events
            is_fire = False  # flag to check if the action is fired
            hold_keys = pygame.key.get_pressed()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if notes_active:# catch all inputs until enter 
                        if event.key == pygame.K_RETURN:
                            notes_active = False
                        else:
                            handle_notes_input(event)

                    elif event.key == pygame.K_r:
                        obs = reset_env()
                        continue
                    elif event.key == pygame.K_n:
                        activate_notes()
                        notes_text = ""

                    elif event.key == pygame.K_p:
                        assert not is_recording, "Already recording"
                        is_recording = True 
                        frame_lst = []
                        if AUTO_ACTIVATE_INPUT:
                            activate_notes()
                    elif event.key == pygame.K_l:
                        assert is_recording, "Not recording"
                        is_recording = False
                        # ! put the frame_lst to the video_lst
                        frame_lst = np.array(frame_lst)
                        # limit the length of the trajectory
                        if len(frame_lst) >= TRAJ_LENGTH_LIMIT:
                            frame_lst = frame_lst[-TRAJ_LENGTH_LIMIT:]
                        else: # repeat the first frame to make the length equal to TRAJ_LENGTH_LIMIT
                            first_frame = frame_lst[0]
                            num_repeat = TRAJ_LENGTH_LIMIT - len(frame_lst)
                            repeat_frames = [first_frame] * num_repeat
                            repeat_frames = np.array(repeat_frames)
                            frame_lst = np.concatenate([repeat_frames, frame_lst], axis=0)
                        video_lst.append(frame_lst)
                        # ! add the instruction to the instruction_df
                        instruction = notes_text
                        trajectory_local_idx = data_id

                        instruction_df.loc[len(instruction_df)] = dict(
                            data_id=data_id,
                            instruction=instruction,
                            trajectory_chunk_file=trajectory_chunk_file,
                            trajectory_local_idx=trajectory_local_idx,
                        )
                        data_id += 1 
                        
                        if RECORD_WALKTHROUGH:
                            game_walkthrough_df.loc[len(game_walkthrough_df)] = dict(
                                goal=instruction,
                                room=info['room'],
                                x=info['agent_pos'][0],
                                y=info['agent_pos'][1]
                            )
                        

                    else:
                        action, is_fire = pygame_action(event)

                if is_fire:
                    obs, rew, done, trunc, info = env.step(action)
                    is_fire = False
                    if is_recording:
                        # ! resize the frame to CLIP_SIZE
                        # to tensor
                        resize_obs = th.from_numpy(obs).permute(2, 0, 1) # shape (3, h, w)
                        resize_obs = th_f_resize(resize_obs, (ENV_SIZE, ENV_SIZE), interpolation=InterpolationMode.BICUBIC, antialias=True) # ! need to resize twice (the first resizing will cause the the observation image to be blurred). This is because during RL, the language reward model can only receive blurred image coming from the rl policy model and then resize it to 224x224
                        resize_obs = th_f_resize(resize_obs, (CLIP_SIZE, CLIP_SIZE), interpolation=InterpolationMode.BICUBIC, antialias=True)
                        # permute back to (h, w, 3)
                        resize_obs = resize_obs.permute(1, 2, 0)
                        resize_obs = resize_obs.numpy()
                        # assert uint8
                        assert resize_obs.dtype == np.uint8
                        frame_lst.append(resize_obs)
                        ic(resize_obs.shape)

                    if done:
                        obs = reset_env()

            # ! ----- Render screen -----
            screen.fill((0, 0, 0))  # fill the screen with black color

            # display the image
            pygame_display_image(
                screen, obs, int(screen_width * 0.1), int(screen_height * 0.2)
            )
            # display the text
            text = f"""Notes: {notes_text}
    Cur Rew: {rew}
    """
            if info is not None:
                info_lines = [f"{k}: {v}" for k, v in info.items() if k != 'total_goal_seq']
                formatted_info = "\t\n".join(info_lines)
                text += f"Info:\n {formatted_info}\n"
            text += "Record status: " + "not recording" if not is_recording else "recording"
            pygame_display_text(screen, text, 10, 10)
            if notes_active:
                display_notes(screen, notes_text, notes_box)

            pygame.display.flip()  # update the screen

            # control the fps
            clock.tick(30)  # 30 fps
            
            # ! auto save state 
            if info is not None and info['room'] != cur_room:
                cur_room = info['room']
                # save the state
                env.save_state(save_state_path)
            
            # ! ---- End of Rendering ----
    except AssertionError as e:
        print(e)
        
    pygame.quit()
    env.close()

    # Save the video_lst
    video_lst = np.array(video_lst)
    
    with open(os.path.join(save_dir, "expert_traj_chunk_0.pkl"), "wb") as f:
        pickle.dump(video_lst, f)

    instruction_df.to_csv(os.path.join(save_dir, "instr.csv"), index=False)
    if RECORD_WALKTHROUGH:
        game_walkthrough_df.to_csv(os.path.join(save_dir, "game_walkthrough.csv"), index=False)

if __name__ == "__main__":
    main()
    # (256, 10, 224, 224, 3)
