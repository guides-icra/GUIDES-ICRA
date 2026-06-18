"""
Implementation of Diffusion Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi
"""
from typing import Callable, Union
import math
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# requires diffusers==0.11.1
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

import random
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.state_infuse.describer import Describer, Strategist
from robomimic.state_infuse.cal_similarity import *
import csv
import clip
from robomimic.state_infuse.state_estimator_model import CompletionTaskEmbeddingModel, CompletionEstimationWithStateDescription
from robomimic.state_infuse.get_state_awarness_of_openai import get_internal_state_form_openai
from robomimic.state_infuse.get_state_awarness_of_openai import get_embeddings as get_openai_embedding
import scipy.spatial.transform as tf
from robomimic.macros import LANG_EMB_KEY
import numpy as np
from PIL import Image
import os

@register_algo_factory_func("diffusion_policy")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    if algo_config.unet.enabled:
        print("diffusion policy unet enabled")
        return DiffusionPolicyUNet, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()

class DiffusionPolicyUNet(PolicyAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        
        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        
        obs_dim = obs_encoder.output_shape()[0]

        # create network object
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim*self.algo_config.horizon.observation_horizon
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'obs_encoder': obs_encoder,
                'noise_pred_net': noise_pred_net
            })
        })

        nets = nets.float().to(self.device)
        
        # setup noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
        else:
            raise RuntimeError()
        
        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)
                
        # set attrs
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
        
        self.total_step = 0
        self.state_mapping_model = None
        self.progress_provider = None
        self.ensemble_state_mappers = []
        self.ensemble_optimizers = []
        self.ensemble_scheduler = [] 
        self.describer = Describer(device=self.device)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.image_dic = {}
        self.acc_sim = {}
        self.image_step = 5
        self.last_condition = {}
        self.last_action = {}
    
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, :Tp, :]
        
        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError('"actions" must be in range [-1,1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.')
            self.action_check_done = True
        
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        
    def train_on_batch(self, batch, epoch, validate=False, lang_encoder=None):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        B = batch['actions'].shape[0]
        
        
        with TorchUtils.maybe_no_grad(no_grad=validate):
            if self.ensemble_state_mappers:
                current_completion_batch = batch['obs']['progresses'][:, 0, :]
                current_img_batch = batch['obs']["robot0_eye_in_hand_image"][:, 0, :, :]
                current_eef_batch = batch['obs']["robot0_base_to_eef_pos"][:, 0, :]
                current_quat_batch = batch['obs']["robot0_base_to_eef_quat"][:, 0, :]
                current_gripper_batch = batch["obs"]["robot0_gripper_qpos"][:, 0]

                del batch['obs']['progresses']
                
                def process_index(index):
                    task_str = batch['task_str'][index]
                    task_complete_rate = current_completion_batch[index].cpu().numpy()
                    task_complete_rate = task_complete_rate[0]

                    task_complete_rate = round(task_complete_rate, 2)

                    img = current_img_batch[index].cpu().numpy()
                    img_np = img.transpose(1, 2, 0)
                    img_np = (img_np * 255).astype(np.uint8)

                    PIL_img = Image.fromarray(img_np)

                    if index not in self.image_dic.keys():
                        self.image_dic[index] = [PIL_img]
                    else:

                        if len(self.image_dic[index]) > 10:
                            self.image_dic[index].pop(0)
                        else:
                            a = 1
                        self.image_dic[index].append(PIL_img)


                    if index not in self.acc_sim.keys():
                        self.acc_sim[index] = 0

                    else:
                        a = 1
                    eef = current_eef_batch[index].cpu().numpy()
                    quat = current_quat_batch[index].cpu().numpy()
                    rpy = tf.Rotation.from_quat(quat.reshape(-1, 4)).as_euler('xyz')[0]
                    gripper_dis = current_gripper_batch[index].cpu().numpy()
                    # PIL_img.save(f"output_image{index}.png")

                    if len(self.image_dic[index]) ==1:
                        self.acc_sim[index] = 2

                    else:
                        
                        image_now = PIL_img
                        image_past = self.image_dic[index][len(self.image_dic[index])-1-1] # not the last one.
                        
                        image_features_now = image_features(image_now,self.clip_model,self.clip_preprocess,self.device)
                        image_features_past = image_features(image_past,self.clip_model,self.clip_preprocess,self.device)

                        res = image_features_now @ image_features_past.T
                        self.acc_sim[index] += (1-res)

                    if self.acc_sim[index] >1:
                        condition, next_action = self.describer.describe(task=task_str, pil_img=PIL_img, eef=eef, rpy=rpy, gripper_distance=gripper_dis)
                        self.acc_sim[index] = 0
                        self.last_action[index] = next_action
                        # print("condition",condition)
                        # print("next_action",next_action)
                        # save_dir = "./action_record"
                        # if not os.path.exists(save_dir):
                        #     os.makedirs(save_dir)
                        
                        # file_path = os.path.join(save_dir, f"{task_str}.csv")
                        # file_exists = os.path.isfile(file_path)
                        # with open(file_path, "a", newline='', encoding="utf-8") as f:
                        #     writer = csv.writer(f)
                        #     if not file_exists:
                        #         writer.writerow(["condition", "next_action"])
                            
                        #     writer.writerow([
                        #         condition,
                        #         next_action
                        #     ])        
                    else:
                        next_action = self.last_action.get(index, "VLM does not provide next action")

                    return (next_action,None)
                current_task_emb_batch = batch['obs'][LANG_EMB_KEY][:, 0, :]
                timestep = batch['obs'][LANG_EMB_KEY].size()[1]
                first_frame_right_images = batch['obs']['robot0_agentview_right_image'][:, 0, :]
                batch_size = first_frame_right_images.size()[0]

                results = []
                for i in range(batch_size):
                    results.append(process_index(i))

                next_action_db = list(map(lambda x: x[0], results))
               
                next_action_embedding_db = lang_encoder.get_lang_emb(next_action_db)    
                next_action_embedding_db = TensorUtils.to_numpy(next_action_embedding_db)
                next_action_tensor_db = torch.tensor(next_action_embedding_db).to(self.device).to(torch.float)

        
                embedding_tensor_from_openai = next_action_tensor_db

                x = (current_completion_batch, current_task_emb_batch, embedding_tensor_from_openai)
                # print("batch_size",batch_size,"timestep",timestep)
                next_ac_from_openai = self.predict_boosted_state_mapping(x, batch_size, timestep, finetune=True)

                target_action_tensor = batch["actions"].to(self.device)

                expanded_tensor = next_ac_from_openai.clone()
            else:
                expanded_tensor = None

            info = super(DiffusionPolicyUNet, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch['actions']
            
            # encode obs
            inputs = {
                'obs': batch["obs"],
                'goal': batch["goal_obs"]
            }
            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                assert inputs['obs'][k].ndim - 2 == len(self.obs_shapes[k])
            
            obs_features = TensorUtils.time_distributed(inputs, self.nets['policy']['obs_encoder'], inputs_as_kwargs=True)
            assert obs_features.ndim == 3  # [B, T, D]

            obs_cond = obs_features.flatten(start_dim=1)
            
            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=self.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=self.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)
            
            expanded_tensor = expanded_tensor.mean(dim=1)
            global_cond = obs_cond + expanded_tensor
            # predict the noise residual
            noise_pred = self.nets['policy']['noise_pred_net'](
                noisy_actions, timesteps, global_cond=global_cond)
            
            # L2 loss
            loss = F.mse_loss(noise_pred, noise)
            
            # logging
            losses = {
                'l2_loss': loss
            }
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # gradient step
               if self.ensemble_state_mappers:
                   self.zero_grad_optimizer_for_ensemble_state_mapping()
                   loss.backward(retain_graph=True)
                   self.grad_norm_optimizer_for_ensemble_state_mapping()
                   self.step_optimizer_for_ensemble_state_mapping()
                   self.schedule_optimizer_for_ensemble_state_mapping(loss.item())
              #     print("xytestttttttttttttttttt")
               policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )
                
                # update Exponential Moving Average of the model weights
               if self.ema is not None:
                   self.ema.step(self.nets)
                
               step_info = {
                    'policy_grad_norms': policy_grad_norms
                }
               info.update(step_info)

        return info
    
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(DiffusionPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
    
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=To)
        action_queue = deque(maxlen=Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue
    
    def get_action(self, obs_dict, goal_dict=None, x_delta_emb=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        # obs_dict: key: [1,D]
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon

        # TODO: obs_queue already handled by frame_stack
        # make sure we have at least To observations in obs_queue
        # if not enough, repeat
        # if already full, append one to the obs_queue
        # n_repeats = max(To - len(self.obs_queue), 1)
        # self.obs_queue.extend([obs_dict] * n_repeats)
        
        if len(self.action_queue) == 0:
            # no actions left, run inference
            # turn obs_queue into dict of tensors (concat at T dim)
            # import pdb; pdb.set_trace()
            # obs_dict_list = TensorUtils.list_of_flat_dict_to_dict_of_list(list(self.obs_queue))
            # obs_dict_tensor = dict((k, torch.cat(v, dim=0).unsqueeze(0)) for k,v in obs_dict_list.items())
            
            # run inference
            # [1,T,Da]
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict,x_delta_emb=x_delta_emb)
            
            # put actions into the queue
            self.action_queue.extend(action_sequence[0])
        
        # has action, execute from left to right
        # [Da]
        action = self.action_queue.popleft()
        
        # [1,Da]
        action = action.unsqueeze(0)
        return action
        
    def _get_action_trajectory(self, obs_dict, goal_dict=None, x_delta_emb=None):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError
        
        # select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model
        
        # encode obs
        inputs = {
            'obs': obs_dict,
            'goal': goal_dict
        }
        for k in self.obs_shapes:
            # first two dimensions should be [B, T] for inputs
            assert inputs['obs'][k].ndim - 2 == len(self.obs_shapes[k])
        obs_features = TensorUtils.time_distributed(inputs, self.nets['policy']['obs_encoder'], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)
        if x_delta_emb is not None:
            obs_cond = obs_cond + x_delta_emb.mean(dim=1)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=self.device)
        naction = noisy_action
        
        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets['policy']['noise_pred_net'](
                sample=naction, 
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        # process action using Ta
        start = To - 1
        end = start + Ta
        action = naction[:,start:end]
        return action
    
    def save_ensemble_state_mapping(self, epoch, path):
        if not os.path.exists(path):
            os.makedirs(path)

        for t, model in enumerate(self.ensemble_state_mappers):
            model_path = os.path.join(path, f"weak_learner_{t + 1}_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Weak Learner {t + 1} at Epoch {epoch + 1} saved to {model_path}")
            
    def load_ensemble_state_mapping(self, epoch, path, num_learners=1):
        for t in range(num_learners):
            # model_path = os.path.join(path, f"weak_learner_{t + 1}_epoch_{epoch + 1}.pt")
            model_path = path

            if os.path.exists(model_path):
                self.ensemble_state_mappers[t].load_state_dict(torch.load(model_path))
                self.ensemble_state_mappers[t].eval()

                print(f"Weak Learner {t + 1} at Epoch {epoch + 1} loaded from {model_path}")
            else:
                raise FileNotFoundError(f"Model file {model_path} not found. Please check the path and epoch number.")
                # print(f"Model file {model_path} not found. Please check the path and epoch number.")
   
    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        self.nets.load_state_dict(model_dict["nets"])
        if model_dict.get("ema", None) is not None:
            self.ema.averaged_model.load_state_dict(model_dict["ema"])
    
    def zero_grad_optimizer_for_ensemble_state_mapping(self):
        for optimizer in self.ensemble_optimizers:
            optimizer.zero_grad()

    def step_optimizer_for_ensemble_state_mapping(self):
        for optimizer in self.ensemble_optimizers:
            optimizer.step()

    def grad_norm_optimizer_for_ensemble_state_mapping(self):
        for optimizer in self.ensemble_optimizers:
            parameters = [p for group in optimizer.param_groups for p in group['params']]
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=10)

    def schedule_optimizer_for_ensemble_state_mapping(self, epoch_loss):
        for scheduler in self.ensemble_scheduler:
            scheduler.step(epoch_loss)
    
    def predict_boosted_state_mapping(self, x, batch_size, timestep, action_size=12, ensemble_lr=0.1
                                      , finetune=False):

        # Set all models to evaluation mode
        y_pred_size = self.ensemble_state_mappers[0].output_size
        #print("y_pred_size", y_pred_size)

        for model in self.ensemble_state_mappers:
            if not finetune:
                model.eval()
            else:
                model.train()

        # Make predictions with each model in the ensemble

        def predict_from_ensemble():
            y_pred_combine = torch.zeros(batch_size, timestep, y_pred_size)
            y_pred_combine = y_pred_combine.to(self.device)
           # print("y_pred_combine", y_pred_combine.shape)
            for t, model in enumerate(self.ensemble_state_mappers):
                y_pred = model(*x)
                y_pred = y_pred.view([batch_size, timestep, -1])
                y_pred_combine += ensemble_lr * y_pred

            return y_pred_combine

        if not finetune:
            with torch.no_grad():
                # print(f"Weak Learner {t + 1} prediction added")
                y_hat = predict_from_ensemble()
        else:
            y_hat = predict_from_ensemble()

        return y_hat
    
    
    def initial_ensemble_state_mapper(self, num_learner=1):
        if self.algo_config.progress_dim_size == 0:
            self.ensemble_state_mappers = []
            return

        for _ in range(num_learner):
            model = CompletionEstimationWithStateDescription(
                task_str_emb_size=self.algo_config.lang_embed_dim,
                hidden_mapping_size=self.algo_config.progress_dim_size,
                transformer_encoding_size=self.algo_config.transformer.embed_dim,
                # state_descp_size=config.algo.openai_emb_size,
                state_descp_size=self.algo_config.lang_embed_dim,
            ).to(self.device)

            self.ensemble_state_mappers.append(model)

            _optimizer = torch.optim.Adam(
                model.parameters(),
                lr=1e-5,
                weight_decay=1e-4
            )

            self.ensemble_optimizers.append(_optimizer)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                _optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=False
            )

            self.ensemble_scheduler.append(scheduler)
        self.num_ensemble = len(self.ensemble_state_mappers)
        self.y_pred_size = self.ensemble_state_mappers[0].output_size

        input_dim = self.num_ensemble * self.y_pred_size
        output_dim = self.y_pred_size

        self.reprojection = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
            ).to(self.device)

        self.aem_embed_dim = self.y_pred_size
        self.attention_query = nn.Parameter(
            torch.randn(1, 1, self.aem_embed_dim, device=self.device)  
        )
        self.num_heads = 4
        self.aem_attention = nn.MultiheadAttention(
            embed_dim=self.aem_embed_dim,
            num_heads=self.num_heads,
            batch_first=True
        ).to(self.device)
            
            

# =================== Vision Encoder Utils =====================
def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version('1.9.0'):
        raise ImportError('This function requires pytorch >= 1.9.0')

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module, 
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group, 
            num_channels=x.num_features)
    )
    return root_module

# =================== UNet for Diffusion ==============

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM 
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level. 
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x
