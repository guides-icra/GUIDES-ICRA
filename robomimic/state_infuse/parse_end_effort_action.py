import json
import numpy as np
import os
import re
import sys
import torch
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
from tqdm import tqdm
from robomimic.state_infuse.get_state_awarness_of_openai import get_internal_state_form_openai
import pickle
import cv2
import re
import ast
import time
import concurrent.futures
from multiprocessing import cpu_count
from collections import defaultdict
import scipy.spatial.transform as tf
from PIL import Image

TASK_PATH_MAPPING = {
    # "OpenDrawer": "/home/minquangao/robocasa/datasets/v0.1/single_stage/kitchen_drawer/OpenDrawer/mg/2024-05-04-22-38-42/demo_gentex_im128_randcams.hdf5",
    # "CloseDrawer": "/home/minquangao/robocasa/datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo_gentex_im128_randcams.hdf5",
    # "PnPCabToCounter": "/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/mg/2024-07-12-04-33-29/demo_gentex_im128_randcams.hdf5",
    # "PnPSinkToCounter": "/home/minquangao/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/demo_gentex_im128_randcams.hdf5",
    # "PnPCounterToSink": "/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/mg/2024-05-04-22-14-06_and_2024-05-07-07-40-17/demo_gentex_im128_randcams.hdf5",
    # "PnPStoveToCounter": "/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/mg/2024-05-04-22-14-40/demo_gentex_im128_randcams.hdf5",
    # "PnPCounterToMicrowave": "/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/mg/2024-05-04-22-13-21_and_2024-05-07-07-41-17/demo_gentex_im128_randcams.hdf5",
    # "PnPCounterToStove": "/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/mg/2024-05-04-22-14-20/demo_gentex_im128_randcams.hdf5",
    # "PnPMicrowaveToCounter": "/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/mg/2024-05-04-22-14-26_and_2024-05-07-07-41-42/demo_gentex_im128_randcams.hdf5",
    "PnPCounterToCab": "/scratch/tshu2/xli383/research/completion-infuse-robot/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/mg/2024-05-04-22-12-27_and_2024-05-07-07-39-33/demo_gentex_im128_randcams.hdf5",
    # "CoffeePressButton": "/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeePressButton/mg/2024-05-04-22-21-32/demo_gentex_im128_randcams.hdf5",
    # "CoffeeSetupMug": "/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeSetupMug/mg/2024-05-04-22-22-13_and_2024-05-08-05-52-13/demo_gentex_im128_randcams.hdf5",
    # "CoffeeServeMug": "/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/mg/2024-05-04-22-21-50/demo_gentex_im128_randcams.hdf5",
    # "CloseDoubleDoor": "/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseDoubleDoor/mg/2024-05-04-22-22-42_and_2024-05-08-06-02-36/demo_gentex_im128_randcams.hdf5",
    # "CloseSingleDoor": "/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseSingleDoor/mg/2024-05-04-22-34-56/demo_gentex_im128_randcams.hdf5",
    # "OpenSingleDoor": "/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor/mg/2024-05-04-22-37-39/demo_gentex_im128_randcams.hdf5",
    # "OpenDoubleDoor": "/home/minquangao/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenDoubleDoor/2024-04-26/demo_gentex_im128_randcams.hdf5",
    # "TurnSinkSpout": "/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnSinkSpout/mg/2024-05-09-09-31-12/demo_gentex_im128_randcams.hdf5",
    # "TurnOffSinkFaucet": "/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOffSinkFaucet/mg/2024-05-04-22-17-26/demo_gentex_im128_randcams.hdf5",
    # "TurnOnSinkFaucet": "/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet/mg/2024-05-04-22-17-46/demo_gentex_im128_randcams.hdf5",
    # "TurnOffStove": "/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOffStove/mg/2024-05-08-09-20-45/demo_gentex_im128_randcams.hdf5",
    # "TurnOnStove": "/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOnStove/mg/2024-05-08-09-20-31/demo_gentex_im128_randcams.hdf5",
    # "TurnOnMicroWave": "/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/mg/2024-05-04-22-40-00/demo_gentex_im128_randcams.hdf5",
    # "TurnOffMicroWave": "/data3/mgao/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOffMicrowave/mg/2024-05-04-22-39-23/demo_gentex_im128_randcams.hdf5"
    # "arrange_vegatable": "/data3/mgao/robocasa/datasets/v0.1/multi_stage/chopping_food/ArrangeVegetables/2024-05-11/demo_im128.hdf5",
    # "microwave-thawing": "/data3/mgao/robocasa/datasets/v0.1/multi_stage/defrosting_food/MicrowaveThawing/2024-05-11/demo_im128.hdf5",
    # "prepare-coffee": "/data3/mgao/robocasa/datasets/v0.1/multi_stage/brewing/PrepareCoffee/2024-05-07/demo_im128.hdf5",
    # "presoak-pan": "/data3/mgao/robocasa/datasets/v0.1/multi_stage/washing_dishes/PreSoakPan/2024-05-10/demo_im128.hdf5",
    # "restock-pantry": "/data3/mgao/robocasa/datasets/v0.1/multi_stage/restocking_supplies/RestockPantry/2024-05-10/demo_im128.hdf5",
}

task_length = defaultdict(list)
# dict: key: task_name: lengthes


def format_image(image):
    image = np.array(image)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def get_task_length(all_demo_dataset):
    task_data = []

    saved_keys = set()

    if hasattr(all_demo_dataset, 'datasets'):
        all_demo_dataset = all_demo_dataset
    else:
        all_demo_dataset = [all_demo_dataset]

    for di, demo_dataset in enumerate(all_demo_dataset):
        exporting_dataset = demo_dataset

        index = 0

        while index < len(exporting_dataset):
            demo_id = exporting_dataset._index_to_demo_id[index]
            demo_length = exporting_dataset._demo_id_to_demo_length[demo_id]
            task_description = exporting_dataset._demo_id_to_demo_lang_str[demo_id]
            print('PROCESSING... dataset index with: ', index)

            task_length[task_description].append(demo_length)

            index += demo_length


def collect_task_data(all_demo_dataset):
    task_data = []

    saved_keys = set()

    if hasattr(all_demo_dataset, 'datasets'):
        all_demo_dataset = all_demo_dataset
    else:
        all_demo_dataset = [all_demo_dataset]

    for di, demo_dataset in enumerate(all_demo_dataset):
        exporting_dataset = demo_dataset
        eye_names = ['robot0_agentview_left_image', 'robot0_eye_in_hand_image', 'robot0_agentview_right_image']

        print('PROCESSING... dataset index with: ', di)
        print("len(exporting_dataset)",len(exporting_dataset))

        previous_id = -1
        count = 0
        # for i in tqdm(range(len(exporting_dataset))):
        #     demo_id = exporting_dataset._index_to_demo_id[i]

        #     if previous_id != -1 and demo_id == previous_id: continue
        #     previous_id = demo_id
        #     count += 1
        # print("count",count)

        for i in tqdm(range(len(exporting_dataset))):
            demo_id = exporting_dataset._index_to_demo_id[i]

            if previous_id != -1 and demo_id == previous_id: continue

            previous_id = demo_id
            demo_start_index = exporting_dataset._demo_id_to_start_indices[demo_id]
            demo_length = exporting_dataset._demo_id_to_demo_length[demo_id]

            # start at offset index if not padding for frame stacking
            demo_index_offset = 0 if exporting_dataset.pad_frame_stack else (exporting_dataset.n_frame_stack - 1)
            index_in_demo = i - demo_start_index + demo_index_offset
            complete_rate = round(index_in_demo / demo_length, 2)
            task_description = exporting_dataset._demo_id_to_demo_lang_str[demo_id]

            task_length = exporting_dataset._demo_id_to_demo_length[demo_id]

            movement_ndarray = np.zeros((demo_length - 1, 14))

            for d in range(task_length):
                dir_name = f'demo_{task_description}_{demo_id}'
                if not os.path.exists((dir_name)):
                    os.makedirs(dir_name)

                if d > 0:
                    current_image = exporting_dataset[i+d]['obs']['robot0_eye_in_hand_image'][-1]
                    print("current_image.shape", current_image.shape)
                    image = Image.fromarray(current_image)
                    image.save(os.path.join(dir_name, f'{d}.png'))

                    current_gripper_dis = exporting_dataset[i+d]['obs']['robot0_gripper_qpos'][-1]
                    last_gripper_dis = exporting_dataset[i+d-1]['obs']['robot0_gripper_qpos'][-1]
                    gd = lambda dis: dis[0] - dis[1]
                    gripper_width_delta = gd(current_gripper_dis) - gd(last_gripper_dis)
                    print('---step: ', d, '--')
                    print('delta gripper distance:', gripper_width_delta)

                    current_eef_quat = exporting_dataset[i+d]['obs']['robot0_base_to_eef_quat'][-1]
                    last_eef_quat = exporting_dataset[i+d-1]['obs']['robot0_base_to_eef_quat'][-1]

                    current_eef_rpy = tf.Rotation.from_quat(current_eef_quat.reshape(-1, 4)).as_euler('xyz')[0]
                    last_eef_rpy = tf.Rotation.from_quat(last_eef_quat.reshape(-1, 4)).as_euler('xyz')[0]

                    delta_rpy = current_eef_rpy - last_eef_rpy

                    current_eef_pos = exporting_dataset[i+d]['obs']['robot0_base_to_eef_pos'][-1]
                    last_eef_pos = exporting_dataset[i+d-1]['obs']['robot0_base_to_eef_pos'][-1]

                    delta_xyz = current_eef_pos - last_eef_pos

                    print('delta of RPY: ', delta_rpy)
                    print('delta of xyz: ', delta_xyz)

                    current_base_xyz = exporting_dataset[i]['obs']['robot0_base_pos'][-1]
                    current_base_quat = exporting_dataset[i]['obs']['robot0_base_quat'][-1]

                    movement_ndarray[d - 1] = np.concatenate([delta_xyz, delta_rpy, [gripper_width_delta], current_base_xyz, current_base_quat])

            np.save(os.path.join(dir_name, f"movement_data_{demo_id}.npy"), movement_ndarray)


def process_task(task):
    save_key = task['save_key']
    task_description = task['task_description']
    complete_rate = task['complete_rate']
    left_image = task['left_image']
    hand_image = task['hand_image']
    right_image = task['right_image']

    try:
        print(f'Getting task {task_description} in progress {complete_rate} from openai')
        s = time.time()

        # Call the function to get internal state
        internal_state = get_internal_state_form_openai(
            left_image, hand_image, right_image,
            complete_rate, task_description,
            with_complete_rate=True,
            write_image=False,
            with_image_format_change=False
        )

        # Clean the result
        internal_state = re.sub(r'[\n\t]+', '', internal_state)
        internal_state = internal_state.replace('python', '')
        internal_state = internal_state.strip('`')
        internal_state = ast.literal_eval(internal_state)

    except Exception as e:
        print('get error: ', e)
        print('when processing task: ', task_description, ' with progress: ', complete_rate)

    return (task_description, save_key, internal_state)


def extract_and_export_image_parallel(all_demo_dataset):
    # Initialize containers for task progress and error recordings
    task_progress_states_mapping = {}

    # Collect all tasks into a container and update task_progress_states_mapping
    task_data = collect_task_data(all_demo_dataset)

def generate_concated_images_from_demo_path(task_name=None, file_path=None):
    config_path_compsoite = "robomimic/scripts/run_configs/seed_123_ds_human-50.json"
    # config_path_compsoite = "/home/minquangao/pretrained_models/configs/seed_123_ds_human-50.json"
    ext_cfg = json.load(open(config_path_compsoite, 'r'))

    if task_name:
        ext_cfg['train']['data'].append(
            {'path':file_path}
        )
        # print('loading from path ', TASK_PATH_MAPPING[task_name])
    else:
        for path in TASK_PATH_MAPPING.values():
            ext_cfg['train']['data'].append({'path': path})

    config = config_factory(ext_cfg["algo_name"])

    with config.values_unlocked():
        config.update(ext_cfg)

    """
    Train a model using the algorithm.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    # set num workers
    torch.set_num_threads(1)

    # print("\n============= New Training Run with Config =============")
    # print(config)
    # print("")
    # print(config)
    log_dir, ckpt_dir, video_dir, vis_dir = TrainUtils.get_exp_dir(config)

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # extract the metadata and shape metadata across all datasets
    env_meta_list = []
    shape_meta_list = []
    for dataset_cfg in config.train.data:
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        ds_format = config.train.data_format
        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

        # load basic metadata from training file
        # print("\n============= Loaded Environment Metadata =============")
        # print(dataset_path)
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path, ds_format=ds_format)

        # populate language instruction for env in env_meta
        env_meta["env_lang"] = dataset_cfg.get("lang", None)

        # update env meta if applicable
        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=dataset_path,
            action_keys=config.train.action_keys,
            all_obs_keys=config.all_obs_keys,
            ds_format=ds_format,
            verbose=False
        )
        shape_meta_list.append(shape_meta)

    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"], lang_encoder=None)

    extract_and_export_image_parallel(trainset)
    # get_task_length(demo_dataset)

if __name__ == '__main__':
    # import argparse

    #
    # parser = argparse.ArgumentParser(description='Train a Value Predication Model Via Vision Transformer model.')
    # parser.add_argument('--task_id', type=int, required=True, help='specify the task id to expoert')
    #
    # task_id = parser.parse_args().task_id

    # task_path_mapping = list(TASK_PATH_MAPPING.items())

    for key, value in TASK_PATH_MAPPING.items():
        print('PROCESSING.... ', key)
        print('FROM PATH.... ', value)
        generate_concated_images_from_demo_path(key, value)

    # for ts, vs in task_length.items():
    #     print('task: ', ts, ' with length: ', task_length[ts])
    #     task_length[ts] = sum(vs) / len(vs)

    # with open('task_length.pkl', 'wb') as f:
    #     pickle.dump(task_length, f)


