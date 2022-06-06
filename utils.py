import numpy as np
import cv2 
import math

def add_arrow(image, action):
    # action: [steering_angle, linear_speed]
    # image: cv2 read image

    start_pos = (320, 320) # TODO: Will change this
    # print('action: {}'.format(action))

    # Calculate the length according the linear speed
    # The maximum velocity is 0.2 m/s
    arrow_len = 100 * abs(action[1])/0.6 # TODO: check this again

    # Calculate the ending point
    action_x = math.ceil(arrow_len * math.sin(action[0]))
    action_y = math.ceil(arrow_len * math.cos(action[0]))

    # Signs are for pictures:
    # up means negative in y pixel axis, and right means positive 
    if action[1] > 0:
        end_pos = (start_pos[0] + action_x, start_pos[1] - action_y)
    else:
        end_pos = (start_pos[0] - action_x, start_pos[1] + action_y)

    cv2.arrowedLine(image, start_pos, end_pos, (0,255,255), 6)
    image[start_pos[0]-1:start_pos[0]+1, start_pos[1]-1:start_pos[1]+1] = (0,0,0)

    return image



def construct_run_command(script, arguments):
    command = f'python3 {script}'
    for k, v in arguments.items():
        if isinstance(v, bool):
            if v:
                command += f' --{k}'
        else:
            command += f' --{k} {str(v)}'
    return command


# Expects a list of dictionaries
def construct_variants(variants, default_dict=dict(), name_key=None):
    level_keys = []
    variant_levels = []
    for var_level in variants:
        keys, values = zip(*var_level.items())
        assert all([len(v) == len(values[0]) for v in values])

        variants = list(zip(*values))
        level_keys.append(keys)
        variant_levels.append(variants)
    all_keys = sum(level_keys, tuple())
    all_variants = list(itertools.product(*variant_levels))
    all_variants = [sum(v, tuple()) for v in all_variants]
    assert all([len(v) == len(all_keys) for v in all_variants])

    final_variants = []
    for variant in all_variants:
        d = default_dict.copy()
        d.update({k: v for k, v in zip(all_keys, variant)})
        if name_key:
            d[name_key] = '_'.join([f"[{k}]_{v.replace('/', '_') if type(v) == str else v}" for k, v in zip(all_keys, variant)])
        final_variants.append(d)

    return final_variants

# Methods to calculate similarity between embeddings
# def euclidean_similarity(z1, z2): 
