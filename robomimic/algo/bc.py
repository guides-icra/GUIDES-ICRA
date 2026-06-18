"""
Implementation of Behavioral Cloning (BC).
"""
import sys
from collections import OrderedDict
import concurrent.futures

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D

import robomimic.models.base_nets as BaseNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.vae_nets as VAENets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.lang_utils as LangUtils
import robomimic.utils.aem_ablation as SBERT
from robomimic.macros import LANG_EMB_KEY

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.state_infuse.state_estimator_model import CompletionTaskEmbeddingModel, CompletionEstimationWithStateDescription
from robomimic.state_infuse.get_state_awarness_of_openai import get_internal_state_form_openai
from robomimic.state_infuse.get_state_awarness_of_openai import get_embeddings as get_openai_embedding
from robomimic.state_infuse.utils import parse_next_action
import cv2
import os
import random
from robomimic.state_infuse.state_db_manager import state_db as task_db_manager
from PIL import Image
import scipy.spatial.transform as tf
from robomimic.state_infuse.describer import Describer, Strategist
from robomimic.state_infuse.cal_similarity import *
import csv
import clip

@register_algo_factory_func("bc")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    # note: we need the check below because some configs import BCConfig and exclude
    # some of these optionsde
    gaussian_enabled = ("gaussian" in algo_config and algo_config.gaussian.enabled)
    gmm_enabled = ("gmm" in algo_config and algo_config.gmm.enabled)
    vae_enabled = ("vae" in algo_config and algo_config.vae.enabled)

    rnn_enabled = algo_config.rnn.enabled
    transformer_enabled = algo_config.transformer.enabled

    if gaussian_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            raise NotImplementedError
        else:
            algo_class, algo_kwargs = BC_Gaussian, {}
    elif gmm_enabled:
        if rnn_enabled:
            algo_class, algo_kwargs = BC_RNN_GMM, {}
        elif transformer_enabled:
            algo_class, algo_kwargs = BC_Transformer_GMM, {}
        else:
            algo_class, algo_kwargs = BC_GMM, {}
    elif vae_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            raise NotImplementedError
        else:
            algo_class, algo_kwargs = BC_VAE, {}
    else:
        if rnn_enabled:
            algo_class, algo_kwargs = BC_RNN, {}
        elif transformer_enabled:
            algo_class, algo_kwargs = BC_Transformer, {}
        else:
            algo_class, algo_kwargs = BC, {}

    return algo_class, algo_kwargs


class BC(PolicyAlgo):
    """
    Normal BC training.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.ActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets = self.nets.float().to(self.device)

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
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, 0, :]
        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

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
        
        with TorchUtils.maybe_no_grad(no_grad=validate):

            # current_completion_batch = batch['obs']['progresses'][:, 0, :]
            current_img_batch = batch['obs']["robot0_eye_in_hand_image"][:, 0, :, :]
            current_eef_batch = batch['obs']["robot0_base_to_eef_pos"][:, 0, :]
            current_quat_batch = batch['obs']["robot0_base_to_eef_quat"][:, 0, :]
            current_gripper_batch = batch["obs"]["robot0_gripper_qpos"][:, 0]

            del batch['obs']['progresses']

            if self.ensemble_state_mappers:

                def process_index(index):
                    task_str = batch['task_str'][index]
                    # task_complete_rate = current_completion_batch[index].cpu().numpy()
                    # task_complete_rate = task_complete_rate[0]

                    # task_complete_rate = round(task_complete_rate, 2)

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
                        # condition, next_action = ' ',' '
                        next_action = self.last_action.get(index, "VLM does not provide next action")

                    # condition = self.describer.describe(task=task_str, pil_img=PIL_img, eef=eef, rpy=rpy, gripper_distance=gripper_dis)
                    # print("condition",condition)
                    # print("next_action",next_action)

                    # next_action = self.strategist.strategize(task=task_str, pil_img= PIL_img, condition=condition, eef= eef, rpy=rpy)
                    # # print(f"next_action{index}",next_action)
           

                    # pr.disable()
                    # pr.dump_stats("train_batch.prof") 
                    return (next_action,None)
                    # global task_db_manager

                    # try:
                    #     next_action = task_db_manager.retrieve_data(task_str,
                    #                                                 task_complete_rate,
                    #                                                 state_key='Next Action'
                    #                                                 )
                    # except KeyError as e:
                    #     error_path = batch['hd5_file_path'][index]
                    #     with open('error_log.txt', 'a') as f:
                    #         f.write(f'{error_path}\n')
                    #         f.write(f'{task_str} : {task_complete_rate}\n')
                    #         f.write(f'{e}\n\n')
                    #     print(e)
                    #     next_action = ""

                    # try:
                    #     error_avoidance = task_db_manager.retrieve_data(task_str,
                    #                                                     task_complete_rate,
                    #                                                     state_key='Error Avoidance'
                    #                                                     )
                    # except KeyError as e:
                    #     error_avoidance = ""
                    # # print(f'task {index} : {task_str} : {next_action}')

                    # return (next_action, error_avoidance, task_complete_rate)

                # Use ThreadPoolExecutor for parallel processing
                # with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Create a list of tasks for the executor
                    # results = list(executor.map(process_index, range(batch_size)))

                # if self.state_mapping_model and self.state_mapping_model.hidden_mapping_size > 0:
                current_task_emb_batch = batch['obs'][LANG_EMB_KEY][:, 0, :]
                timestep = batch['obs'][LANG_EMB_KEY].size()[1]
                first_frame_right_images = batch['obs']['robot0_agentview_right_image'][:, 0, :]
                batch_size = first_frame_right_images.size()[0]

                results = []
                for i in range(batch_size):
                    results.append(process_index(i))

                error_avoidance = list(map(lambda x: x[0], results))
                # error_avoidance_embedding = lang_encoder.get_lang_emb(error_avoidance)
                # error_avoidance_embedding = TensorUtils.to_numpy(error_avoidance_embedding)
                # error_avoidance_tensor = torch.tensor(error_avoidance_embedding).to(self.device).to(torch.float)

                # sbert_encoder = SBERT.SBERTLangEncoder(device=self.device,)
                
                # if not self.algo_config.embedding_methods:
                next_action_db = list(map(lambda x: x[0], results))
                # next_action_db = results
                # print("next_action_db",next_action_db)
                next_action_embedding_db = lang_encoder.get_lang_emb(next_action_db)    
                next_action_embedding_db = TensorUtils.to_numpy(next_action_embedding_db)
                next_action_tensor_db = torch.tensor(next_action_embedding_db).to(self.device).to(torch.float)

        
                embedding_tensor_from_openai = next_action_tensor_db

                x = (current_task_emb_batch, embedding_tensor_from_openai)
                next_ac_from_openai = self.predict_boosted_state_mapping(x, batch_size, timestep, finetune=True)
                 

                target_action_tensor = batch["actions"].to(self.device)

                # repeats = 512 // 12 + 1
                # expanded_tensor = 10 * next_ac_from_openai.repeat(1, 1, repeats)[:, :, :512]  # Trim to the exact size
                # expanded_tensor = target_action_tensor.repeat(1, 1, repeats)[:, :, :512]  # Trim to the exact size
                expanded_tensor = next_ac_from_openai.clone()

                # action_loss = torch.nn.MSELoss()(next_ac_from_openai, batch["actions"])
                # print('mse of next action from openai and db', action_loss.item())
            else:
                expanded_tensor = None


            info = super(BC, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch, completion_embedding=expanded_tensor)

            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # if embedding_tensor_from_openai is not None:
                if self.ensemble_state_mappers:
                    self.zero_grad_optimizer_for_ensemble_state_mapping()
                    # with torch.autograd.detect_anomaly():
                    losses['action_loss'].backward(retain_graph=True)
                    # action_loss.backward(retain_graph=True)
                    # Normalize gradients before optimizer step
                    self.grad_norm_optimizer_for_ensemble_state_mapping()
                    self.step_optimizer_for_ensemble_state_mapping()
                    self.schedule_optimizer_for_ensemble_state_mapping(losses['action_loss'])

                step_info = self._train_step(losses)
                info.update(step_info)

        return info

    def _forward_training(self, batch, completion_embedding=None):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        predictions = OrderedDict()
        actions = self.nets["policy"](
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
            completion_embedding=completion_embedding
        )
        predictions["actions"] = actions
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        losses = OrderedDict()
        a_target = batch["actions"]
        actions = predictions["actions"]
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss
        return losses

    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        # with torch.autograd.detect_anomaly():
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
            max_grad_norm=self.global_config.train.max_grad_norm,
        )
        info["policy_grad_norms"] = policy_grad_norms

        # step through optimizers
        for k in self.lr_schedulers:
            if self.lr_schedulers[k] is not None:
                self.lr_schedulers[k].step()
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
        log = super(BC, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        return self.nets["policy"](obs_dict, goal_dict=goal_dict)


class BC_Gaussian(BC):
    """
    BC training with a Gaussian policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gaussian.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GaussianActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            fixed_std=self.algo_config.gaussian.fixed_std,
            init_std=self.algo_config.gaussian.init_std,
            std_limits=(self.algo_config.gaussian.min_std, 7.5),
            std_activation=self.algo_config.gaussian.std_activation,
            low_noise_eval=self.algo_config.gaussian.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets = self.nets.clone().float().to(self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 1
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_GMM(BC_Gaussian):
    """
    BC training with a Gaussian Mixture Model policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets = self.nets.float().to(self.device)


class BC_VAE(BC):
    """
    BC training with a VAE policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.VAEActor(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            device=self.device,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **VAENets.vae_args_from_config(self.algo_config.vae),
        )

        self.nets = self.nets.float().to(self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Update from superclass to set categorical temperature, for categorical VAEs.
        """
        if self.algo_config.vae.prior.use_categorical:
            temperature = self.algo_config.vae.prior.categorical_init_temp - epoch * self.algo_config.vae.prior.categorical_temp_anneal_step
            temperature = max(temperature, self.algo_config.vae.prior.categorical_min_temp)
            self.nets["policy"].set_gumbel_temperature(temperature)
        return super(BC_VAE, self).train_on_batch(batch, epoch, validate=validate)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        vae_inputs = dict(
            actions=batch["actions"],
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
            freeze_encoder=batch.get("freeze_encoder", False),
        )

        vae_outputs = self.nets["policy"].forward_train(**vae_inputs)
        predictions = OrderedDict(
            actions=vae_outputs["decoder_outputs"],
            kl_loss=vae_outputs["kl_loss"],
            reconstruction_loss=vae_outputs["reconstruction_loss"],
            encoder_z=vae_outputs["encoder_z"],
        )
        if not self.algo_config.vae.prior.use_categorical:
            with torch.no_grad():
                encoder_variance = torch.exp(vae_outputs["encoder_params"]["logvar"])
            predictions["encoder_variance"] = encoder_variance
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # total loss is sum of reconstruction and KL, weighted by beta
        kl_loss = predictions["kl_loss"]
        recons_loss = predictions["reconstruction_loss"]
        action_loss = recons_loss + self.algo_config.vae.kl_weight * kl_loss
        return OrderedDict(
            recons_loss=recons_loss,
            kl_loss=kl_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["KL_Loss"] = info["losses"]["kl_loss"].item()
        log["Reconstruction_Loss"] = info["losses"]["recons_loss"].item()
        if self.algo_config.vae.prior.use_categorical:
            log["Gumbel_Temperature"] = self.nets["policy"].get_gumbel_temperature()
        else:
            log["Encoder_Variance"] = info["predictions"]["encoder_variance"].mean().item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_RNN(BC):
    """
    BC training with an RNN policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

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
        input_batch = dict()
        input_batch["obs"] = batch["obs"]
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"]

        if self._rnn_is_open_loop:
            # replace the observation sequence with one that only consists of the first observation.
            # This way, all actions are predicted "open-loop" after the first observation, based
            # on the rnn hidden state.
            n_steps = batch["actions"].shape[1]
            obs_seq_start = TensorUtils.index_at_time(batch["obs"], ind=0)
            input_batch["obs"] = TensorUtils.unsqueeze_expand_at(obs_seq_start, size=n_steps, dim=1)

        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.nets["policy"].get_rnn_init_state(batch_size=batch_size, device=self.device)

            if self._rnn_is_open_loop:
                # remember the initial observation, and use it instead of the current observation
                # for open-loop action sequence prediction
                self._open_loop_obs = TensorUtils.clone(TensorUtils.detach(obs_dict))

        obs_to_use = obs_dict
        if self._rnn_is_open_loop:
            # replace current obs with last recorded obs
            obs_to_use = self._open_loop_obs

        self._rnn_counter += 1
        action, self._rnn_hidden_state = self.nets["policy"].forward_step(
            obs_to_use, goal_dict=goal_dict, rnn_state=self._rnn_hidden_state)
        return action

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._rnn_hidden_state = None
        self._rnn_counter = 0


class BC_RNN_GMM(BC_RNN):
    """
    BC training with an RNN GMM policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.rnn.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2 # [B, T]
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_Transformer(BC):
    """
    BC training with a Transformer policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.transformer.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.TransformerActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.transformer_args_from_config(self.algo_config.transformer),
        )
        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)

    def _set_params_from_config(self):
        """
        Read specific config variables we need for training / eval.
        Called by @_create_networks method
        """
        self.context_length = self.algo_config.transformer.context_length
        self.supervise_all_steps = self.algo_config.transformer.supervise_all_steps
        self.pred_future_acs = self.algo_config.transformer.pred_future_acs
        if self.pred_future_acs:
            assert self.supervise_all_steps is True

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
        input_batch = dict()
        h = self.context_length
        input_batch["obs"] = {k: batch["obs"][k][:, :h, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present

        if self.supervise_all_steps:
            # supervision on entire sequence (instead of just current timestep)
            if self.pred_future_acs:
                ac_start = h - 1
            else:
                ac_start = 0
            input_batch["actions"] = batch["actions"][:, ac_start:ac_start+h, :]
        else:
            # just use current timestep
            input_batch["actions"] = batch["actions"][:, h-1, :]

        if self.pred_future_acs:
            assert input_batch["actions"].shape[1] == h

        input_batch = TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        return input_batch

    def _forward_training(self, batch, epoch=None):
        """
        Internal helper function for BC_Transformer algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        # ensure that transformer context length is consistent with temporal dimension of observations
        TensorUtils.assert_size_at_dim(
            batch["obs"],
            size=(self.context_length),
            dim=1,
            msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(self.context_length),
        )

        predictions = OrderedDict()
        predictions["actions"] = self.nets["policy"](obs_dict=batch["obs"], actions=None, goal_dict=batch["goal_obs"])
        if not self.supervise_all_steps:
            # only supervise final timestep
            predictions["actions"] = predictions["actions"][:, -1, :]
        return predictions

    def get_action(self, obs_dict, goal_dict=None, x_delta_emb=None):
        """
        Get policy action outputs.
        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal
        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        if x_delta_emb is None:
            output = self.nets["policy"](obs_dict, actions=None, goal_dict=goal_dict)
        else:
            output = self.nets["policy"](obs_dict, actions=None, goal_dict=goal_dict, x_delta_emb=x_delta_emb)

        if self.supervise_all_steps:
            if self.algo_config.transformer.pred_future_acs:
                output = output[:, 0, :]
            else:
                output = output[:, -1, :]
        else:
            output = output[:, -1, :]

        return output


class BC_Transformer_GMM(BC_Transformer):
    """
    BC training with a Transformer GMM policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """

        assert self.algo_config.gmm.enabled
        assert self.algo_config.transformer.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.TransformerGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.transformer_args_from_config(self.algo_config.transformer),
        )
        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)
        self.openai_emb_size = self.algo_config.openai_emb_size

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
        #self.strategist = Strategist(model_path=self.global_config.model_name, cache_dir=self.global_config.cache_dir, device=self.device)

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

    def predict_boosted_state_mapping(self, x, batch_size, timestep, action_size=12, ensemble_lr=0.1
                                      , finetune=False):

        # Set all models to evaluation mode
        y_pred_size = self.ensemble_state_mappers[0].output_size

        for model in self.ensemble_state_mappers:
            if not finetune:
                model.eval()
            else:
                model.train()

        # Make predictions with each model in the ensemble

        def predict_from_ensemble():
            y_pred_combine = torch.zeros(batch_size, timestep, y_pred_size)
            y_pred_combine = y_pred_combine.to(self.device)

            for t, model in enumerate(self.ensemble_state_mappers):
                y_pred = model(*x)
                # print("y_pred11",y_pred.shape)
                y_pred = y_pred.view([batch_size, timestep, -1])
                # print("y_pred22",y_pred.shape)
                y_pred_combine += ensemble_lr * y_pred

            return y_pred_combine

        if not finetune:
            with torch.no_grad():
                # print(f"Weak Learner {t + 1} prediction added")
                y_hat = predict_from_ensemble()
        else:
            y_hat = predict_from_ensemble()

        return y_hat
    
    def predict_boosted_state_mapping_reprojection(self, x, batch_size, timestep, action_size=12, ensemble_lr=0.1
                                      , finetune=False):

        # Set all models to evaluation mode
        y_pred_size = self.ensemble_state_mappers[0].output_size
        num_models = len(self.ensemble_state_mappers)

        for model in self.ensemble_state_mappers:
            if not finetune:
                model.eval()
            else:
                model.train()

        # Make predictions with each model in the ensemble

        def predict_from_ensemble():
            preds = []
            for t, model in enumerate(self.ensemble_state_mappers):
                y_pred = model(*x).detach().clone()
                y_pred = y_pred.view([batch_size, timestep, -1])
                preds.append(y_pred)
            y_all = torch.cat(preds, dim=-1)
            y_pred_combine = self.reprojection(y_all)

            return y_pred_combine

        if not finetune:
            with torch.no_grad():
                # print(f"Weak Learner {t + 1} prediction added")
                y_hat = predict_from_ensemble()
        else:
            y_hat = predict_from_ensemble()

        return y_hat
    def predict_boosted_state_mapping_attention(self, x, batch_size, timestep, action_size=12, ensemble_lr=0.1
                                      , finetune=False):

        # Set all models to evaluation mode
        y_pred_size = self.ensemble_state_mappers[0].output_size
        num_models = len(self.ensemble_state_mappers)

        for model in self.ensemble_state_mappers:
            if not finetune:
                model.eval()
            else:
                model.train()

        # Make predictions with each model in the ensemble

        def predict_from_ensemble():
            preds = []
            for t, model in enumerate(self.ensemble_state_mappers):
                y_pred = model(*x)
                y_pred = y_pred.view([batch_size, timestep, -1])
                preds.append(y_pred)
            stacked_preds = torch.stack(preds, dim=2)
            B, T, N, D = stacked_preds.shape
            print("stacked_preds.shape",stacked_preds.shape)
            combined = stacked_preds.reshape(B, T*N, D).contiguous().clone()
            query = self.attention_query.repeat(B,1,1).to(self.device)
            attention_output,_ = self.aem_attention(
                query=query,
                key=combined,
                value=combined,
                need_weights=False
            )
            y_pred_combine = attention_output.unsqueeze(1).expand(-1, T, -1).contiguous().clone()

            return y_pred_combine

        if not finetune:
            with torch.no_grad():
                # print(f"Weak Learner {t + 1} prediction added")
                y_hat = predict_from_ensemble()
        else:
            y_hat = predict_from_ensemble()

        return y_hat

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

    def build_optimizer_from_state_mapping(self):
        if self.state_mapping_model:
            self.state_mapping_model.train()  # set model to train
            self.completion_task_embedding_optimizer = torch.optim.Adam(
                self.state_mapping_model.parameters(),
                lr=1e-3,
                weight_decay=1e-4
            )

            self.schedulers_for_completion_task_embedding = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.completion_task_embedding_optimizer,
                mode='min',
                factor=0.5,
                patience=50,
                verbose=True
            )

    def _compute_auxiliary_losses(self, ac_predict, ac_true):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        l2_loss = nn.MSELoss()(ac_predict, ac_true)
        l1_loss = nn.SmoothL1Loss()(ac_predict, ac_true)
        # cosine direction loss on eef delta position
        cos_loss = LossUtils.cosine_loss(ac_predict[..., :3], ac_true[..., :3])

        loss = (
                self.algo_config.loss.l2_weight * l2_loss +
                self.algo_config.loss.l1_weight * l1_loss +
                self.algo_config.loss.cos_weight * cos_loss
        )

        return loss

    def train_boosted_state_mapping(self, x, y, batch_size, timestep, ensemble_lr=0.1):
        y_pred_combine = torch.zeros_like(y)

        epoch_loss = 0  # Initialize epoch loss

        for t in range(len(self.ensemble_state_mappers)):
            residuals = y - y_pred_combine
            model = self.ensemble_state_mappers[t]
            optimizer = self.ensemble_optimizers[t]
            model.train()

            optimizer.zero_grad()
            y_pred = model(*x).clone()
            y_pred = y_pred.view([batch_size, timestep, -1])

            loss = self._compute_auxiliary_losses(y_pred, residuals)

            loss.backward(retain_graph=True)

            epoch_loss += loss.item()

            optimizer.step()

            y_pred_combine = y_pred_combine + ensemble_lr * model(*x).view([batch_size, timestep, -1])

            print(f"  Weak Learner {t + 1}, Loss: {loss.item()}")

        for scheduler in self.ensemble_scheduler:
            scheduler.step(epoch_loss)

        return epoch_loss, y_pred_combine

    def save_ensemble_state_mapping(self, epoch, path):
        if not os.path.exists(path):
            os.makedirs(path)

        for t, model in enumerate(self.ensemble_state_mappers):
            model_path = os.path.join(path, f"weak_learner_{t + 1}_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Weak Learner {t + 1} at Epoch {epoch + 1} saved to {model_path}")

    def build_optimizer_for_progress_provider(self):
        if self.progress_provider:
            self.progress_provider.train()
            self.progress_optimizer = torch.optim.Adam(self.progress_provider.parameters(), lr=self.algo_config.progress_lr, weight_decay=1e-4)
            self.progress_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.progress_optimizer, 'min', patience=3, factor=0.75)

    def _forward_training(self, batch, epoch=None, completion_embedding=None):
        """
        Modify from super class to support GMM training.
        """
        # ensure that transformer context length is consistent with temporal dimension of observations
        TensorUtils.assert_size_at_dim(
            batch["obs"],
            size=(self.context_length),
            dim=1,
            msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(self.context_length),
        )

        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"],
            actions=None,
            goal_dict=batch["goal_obs"],
            low_noise_eval=False,
            completion_embedding=completion_embedding
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2 # [B, T]

        if not self.supervise_all_steps:
            # only use final timestep prediction by making a new distribution with only final timestep.
            # This essentially does `dists = dists[:, -1]`
            component_distribution = D.Normal(
                loc=dists.component_distribution.base_dist.loc[:, -1],
                scale=dists.component_distribution.base_dist.scale[:, -1],
            )
            component_distribution = D.Independent(component_distribution, 1)
            mixture_distribution = D.Categorical(logits=dists.mixture_distribution.logits[:, -1])
            dists = D.MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                component_distribution=component_distribution,
            )

        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC_Transformer_GMM algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of info
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log