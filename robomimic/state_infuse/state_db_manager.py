import os
import pickle
from collections import defaultdict


class TaskDBManager:
    def __init__(self, db_root, preload=True):
        self.db_root = db_root
        # Original structure for fast retrieval
        self.task_progress_states_mapping = defaultdict(dict)
        self.error_recoding = defaultdict(dict)

        # Auxiliary dictionary for quick access to tasks by task name (no float) for missing cases
        self.task_name_to_floats = defaultdict(list)
        if preload:
            self._load_all_data()

    def _load_all_data(self):
        """
        Load all dictionaries from subdirectories under 'stata_db' and combine them into one for faster retrieval.
        """
        for subdir in os.listdir(self.db_root):
            subdir_path = os.path.join(self.db_root, subdir)
            if os.path.isdir(subdir_path):
                # Load task_progress_states_mapping.pkl
                progress_file = os.path.join(subdir_path, "task_progress_states_mapping.pkl")

                if os.path.exists(progress_file):
                    with open(progress_file, 'rb') as f:
                        progress_data = pickle.load(f)
                        self.task_progress_states_mapping.update(progress_data)
                        for key in progress_data.keys():
                            task_name, task_float = key
                            self.task_name_to_floats[task_name].append(task_float)

    def find_nearest_float(self, task_name, task_float):
        """
        Find the nearest float for the given task_name in the task_name_to_floats auxiliary dictionary.
        """
        if task_name not in self.task_name_to_floats:
            raise KeyError(f"No data found for task name {task_name}")

        # Find the closest float in the list for the task_name
        float_list = self.task_name_to_floats[task_name]
        nearest_float = min(float_list, key=lambda x: abs(x - task_float))
        return nearest_float

    def retrieve_data(self, task_name, task_float, state_key=None):
        """
        Retrieve the corresponding dictionary value from task_progress_states_mapping or error_recoding.
        """
        task_float = round(task_float, 2)

        key = (task_name, task_float)
        # First, try to find the key in task_progress_states_mapping
        record = None
        if key in self.task_progress_states_mapping:
            record = self.task_progress_states_mapping[key]
        elif task_name in self.task_name_to_floats:
            nearest_float = self.find_nearest_float(task_name, task_float)
            nearest_key = (task_name, nearest_float)
            if nearest_key in self.task_progress_states_mapping:
                record = self.task_progress_states_mapping[nearest_key]

        if record:
            return record if not state_key else record[state_key]
        else:
            raise KeyError(f"Task {task_name} with float {task_float} not found in any database")


state_db = TaskDBManager("state_db")


if __name__ == '__main__':
    import time

    s = time.time()
    db_manager = TaskDBManager("state_db")
    print("Time to load db:", time.time() - s)

    assert "pick the fish from the plate and place it in the pan" in db_manager.task_name_to_floats

    s = time.time()
    result = db_manager.retrieve_data("pick the fish from the plate and place it in the pan", 0.1)
    print(result)
    print("Time to load db:", time.time() - s)

    s = time.time()
    result = db_manager.retrieve_data("turn on the sink faucet", 0.12, 'Next Action')
    print(result)
    print("Time to load db:", time.time() - s)

    s = time.time()
    try:
        result = db_manager.retrieve_data("turn on the sink faucet", 0.123)
        print(result)
    except KeyError as e:
        print(e)
    print("Time to load db:", time.time() - s)

    s = time.time()
    try:
        result = db_manager.retrieve_data("turn on the sink faucet A", 0.8)
        print(result)
    except KeyError as e:
        print(e)
    print("Time to load db:", time.time() - s)
