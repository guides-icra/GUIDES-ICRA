import os.path

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from PIL import Image
import numpy as np

class VLMThinker:
    vl_chat_processor = None
    tokenizer = None
    vl_gpt = None

    @classmethod
    def set_model(cls, model_path="deepseek-ai/deepseek-vl-7b-chat", cache_dir="/weka/scratch/tshu2/xli383/models", device="cuda:0"):

        if cls.vl_gpt is None:
            if not os.path.exists(cache_dir):
                cache_dir = None
            cls.device = torch.device(device if torch.cuda.is_available() else "cpu")

            cls.vl_chat_processor = VLChatProcessor.from_pretrained(model_path, cache_dir=cache_dir)
            cls.tokenizer = cls.vl_chat_processor.tokenizer

            cls.vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir)
            cls.vl_gpt = cls.vl_gpt.to(torch.float16).to(cls.device).eval()


class Describer(VLMThinker):
    def __init__(self, model_path="deepseek-ai/deepseek-vl-7b-chat", cache_dir="/weka/scratch/tshu2/xli383/models", device="cuda:0"):
        VLMThinker.set_model(model_path, cache_dir, device)

    def describe(self, task: str = None, pil_img: Image.Image = None, eef: np.ndarray = None, rpy: np.ndarray = None, gripper_distance: np.ndarray = None):
        prompt = (
            "<image_placeholder>\n"
            f"Task: '{task}'\n"
            f"EEF Pos: {eef.tolist()}\n"
            f"Orientation (RPY): {rpy.tolist()}\n"
            f"Gripper: {gripper_distance.tolist()} m\n\n"

            "Generate:\n"
            "Condition: Analyze the image and robot state to concisely describe only the key observations critical to task execution or potential hazards.\n"
            "Next Action: what the robot should do next (one specific sentence).\n\n"
            
            "Format:\n"
            "Condition:\n"
            "Next Action:\n\n"
        )
        conversation = [
            {"role": "User", "content": prompt, "images": [pil_img]},
            {"role": "Assistant", "content": ""}
        ]

        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=[pil_img],  
            force_batchify=True,
            padding=True
        ).to(self.device, dtype=torch.float16)

        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True
        )
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        lines = answer.strip().split('\n')
        condition = "None"
        action = "VLM does not provide next action"
        for line in lines:
            if line.lower().startswith("condition:"):
                condition = line[len("Condition:"):].strip()
            elif line.lower().startswith("next action:"):
                action = line[len("Next Action:"):].strip()
                action = ' '.join(action.split())
        
        # if condition and action:
        return condition, action
        # else:
        #     raise ValueError("Could not parse both condition and next action.")


class Strategist(VLMThinker):
    def __init__(self, model_path="deepseek-ai/deepseek-vl-7b-chat", cache_dir="/weka/scratch/tshu2/dzhang98/models", device="cuda:0"):
        VLMThinker.set_model(model_path, cache_dir, device)

    def strategize(self, task: str = None, pil_img: Image.Image = None, condition:str = None, eef: np.ndarray = None, rpy: np.ndarray = None):#, gripper_distance: np.ndarray = None):
        prompt = (
            "<image_placeholder>\n"
            f"The robot is currently performing the task: '{task}'. The description of the robot's situation is: {condition}"
        )

        if eef is not None and rpy is not None:
            prompt += (
                f"\nThe robot's end-effector position relative to the robot base is {eef.tolist()}, "
                f"with an orientation of {rpy.tolist()} in Roll-Pitch-Yaw (RPY) format."
            )

        # prompt += (
        #     "\n\nConsidering both the visual information from the image and the provided robot state, please address the following points explicitly:"
        #     "\n- Analyze and provide recommendations on how the robot can effectively perform the current task."
        #     "\n- Identify and describe any obstacles or objects in the scene that could interfere with the robot's task."
        #     "\n- Highlight any nearby objects that require special caution from the robot to prevent unintended collisions or disturbances."
        #     "\n- Assess potential collision risks or hazards based on the current robot pose and the spatial arrangement of the objects."
        #     "\n- Evaluate environmental conditions such as lighting, clutter, and spatial constraints. Recommend any necessary adjustments to the robot's movements or trajectory to ensure safe and successful task completion."
        #     )
        
        prompt += (
            "\n-Please describe what the robot should do next in one sentence. You need to be very specific about the robot's action. If the robot is picking up something, you need to describe what it is picking up. And if robot is interacting with something, you need to describe what it is interacting with."
            "\n-You just need to describe what the robot should do next in one sentence."
            )
            # print(prompt)

        conversation = [
            {"role": "User", "content": prompt, "images": [pil_img]},
            {"role": "Assistant", "content": ""}
        ]

        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=[pil_img],  
            force_batchify=True,
            padding=True
        ).to(self.device, dtype=torch.float16)

        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=256,
            do_sample=False,
            use_cache=True
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer

class Strategist_future:
    def __init__(self, model_path="deepseek-ai/deepseek-vl-7b-chat", cache_dir="/weka/scratch/tshu2/dzhang98/models", device="cuda:0"):
        if not os.path.exists(cache_dir):
            cache_dir = None
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path, cache_dir=cache_dir)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            cache_dir=cache_dir
        )
        self.vl_gpt = self.vl_gpt.to(torch.float16).to(self.device).eval()

    def strategize(self, 
                   task: str = None, 
                   pil_img: Image.Image = None, 
                   condition: str = None, 
                   eef: np.ndarray = None, 
                   rpy: np.ndarray = None):

        prompt = "<image_placeholder>\n"
        prompt += f"The robot is currently performing the task: '{task}'. The description of the robot's situation is: {condition}"
        
        if eef is not None and rpy is not None:
            prompt += (
                f"{eef.tolist()}represents the next positional displacement (next position minus current position) for the robot's end-effector., {rpy.tolist()}represents the next rotational displacement (change in orientation) for the robot's end-effector relative to its current orientation in Roll-Pitch-Yaw (RPY) format..\n"
            )
        
        prompt += (
            "Based on this analysis, clearly and succinctly state what the robot should do next in one very specific, command-form sentence, indicating explicitly what object it is picking up or interacting with.You just need to describe the robot's next immediate action in one sentence"
        )

        conversation = [
            {"role": "User", "content": prompt, "images": [pil_img]},
            {"role": "Assistant", "content": ""}
        ]
#         conversation = [
#     {
#     "role": "User",
#     "content": f"""<image_placeholder> Based on the provided image and the robot state information:

# - {eef} represents the next positional displacement (next position minus current position) for the robot's end-effector.
# - {rpy} represents the next rotational displacement (change in orientation) for the robot's end-effector relative to its current orientation in Roll-Pitch-Yaw (RPY) format.

# Given the task: '{task}' and the current condition: '{condition}', explicitly address the following points:

# - Analyze and provide recommendations on how the robot should best approach or complete the task based on visual cues from the image.
# - Identify and describe any obstacles or objects in the scene that could interfere with the robot's task.
# - Highlight any nearby objects that require special caution from the robot to prevent unintended collisions or disturbances.
# - Assess potential collision risks or hazards based on the current robot pose and the spatial arrangement of the objects.
# - Evaluate environmental conditions such as lighting, clutter, and spatial constraints, and recommend necessary adjustments to the robot’s movements or trajectory.

# Based on this analysis, clearly and succinctly state what the robot should do next in one very specific, command-form sentence, indicating explicitly what object it is picking up or interacting with.

# You just need to describe the robot's next immediate action in one sentence.""",
# }
#         ]


        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=[pil_img],  
            force_batchify=True,
            padding=True
        ).to(self.device, dtype=torch.float16)

        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=128,  
            do_sample=False,
            use_cache=True
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer