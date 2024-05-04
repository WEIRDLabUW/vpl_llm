import math
from functools import partial

import ipdb
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import torch.nn as nn
from transformers import Trainer, EvalPrediction
import wandb
import matplotlib.pyplot as plt

class PairEncoder(nn.Module):
    """
    Model to encode pairs of accepted and rejected responses
    """

    def __init__(self, embed_dim, hidden_dim, output_dim):
        super(PairEncoder, self).__init__()

        self._model = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, e_c, e_r):
        x = torch.cat([e_c, e_r], dim=1)
        return self._model(x)


class SequenceEncoder(nn.Module):
    """
    Model to encode sequence of responses
    """

    def __init__(self, input_dim, latent_dim):
        super(SequenceEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.linear = nn.Identity()  # TODO: Do we need linear layer?
        self.w_q = nn.Linear(input_dim, input_dim)
        self.w_k = nn.Linear(input_dim, input_dim)
        self.w_v = nn.Linear(input_dim, input_dim)
        # self.w_q = nn.Identity()
        # self.w_k = nn.Identity()
        self.mean_layer = nn.Linear(input_dim, latent_dim)
        self.log_var_layer = nn.Linear(input_dim, latent_dim)
        self.layer_norm = nn.Identity()     # nn.LayerNorm(latent_dim)      # todo: add LayerNorm?

    def forward(
        self, sequences, seq_start_end
    ):  # (C_1+C_2+...+C_n, D), [(0, C_1), (C_1, C_1+C_2), ..., (C_1+...+C_n-1, C_1+...+C_n)]
        outputs = []
        for _, (start, end) in enumerate(seq_start_end):
            context = sequences[start:end]  # C_i x D
            q = self.w_q(context)
            k = self.w_k(context)
            attention_scores = torch.matmul(
                q, k.transpose(0, 1)
            )
            # transformed_seq = self.linear(context)  # C_i x D'
            # attention_scores = torch.matmul(
            #     transformed_seq, transformed_seq.transpose(0, 1)
            # )  # C_i x C_i
            attention_scores = attention_scores / (context.shape[-1] ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)  # C_i x C_i
            weighted_values = torch.matmul(attention_weights, self.w_v(context))  # C_i x D
            output = torch.mean(weighted_values, dim=0)  # D
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)  # n x D

        mean = self.layer_norm(self.mean_layer(outputs))
        log_var = self.layer_norm(self.log_var_layer(outputs))
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Decoder, self).__init__()
        self._model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, xc, xr, z):
        # xc = torch.cat([xc, z], dim=1)
        # xr = torch.cat([xr, z], dim=1)
        rc = self._model(xc)
        rr = self._model(xr)
        return rc, rr


class VAEModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, latent_dim, llm_encoder,
                 fixed_contexts=False, fixed_llm_embeddings=False, use_causal_lm=False):
        super(VAEModel, self).__init__()
        self.llm_encoder = llm_encoder
        self.pair_encoder = PairEncoder(embed_dim, hidden_dim, latent_dim)
        self.sequence_encoder = SequenceEncoder(latent_dim, latent_dim)
        # self.decoder = Decoder(embed_dim + latent_dim, hidden_dim)
        self.decoder = Decoder(embed_dim, hidden_dim)

        self.latent_dim = latent_dim
        self.fixed_contexts = fixed_contexts
        self.fixed_llm_embeddings = fixed_llm_embeddings
        self.use_causal_lm = use_causal_lm

    def reparameterization(self, mean, std):
        epsilon = torch.randn_like(std).to(mean.device)  # sampling epsilon
        epsilon *= 1e-3     # TODO: set scale of variance here
        z = mean + std * epsilon                         # reparameterization trick
        z = F.normalize(z, p=2, dim=-1) * math.sqrt(z.shape[-1])
        return z

    def encode_pair(self, e_c, e_r):
        return self.pair_encoder(e_c, e_r)

    def encode_sequence(self, sequences, seq_start_end):
        return self.sequence_encoder(sequences, seq_start_end)

    def decode(self, e_c, e_r, z):
        return self.decoder(e_c, e_r, z)

    def forward(
        self,
        target_chosen,
        target_rejected,
        context_chosen,
        context_rejected,
        seq_start_end,
        user_type,
        ground_truth_user_vector=False
    ):
        # ipdb.set_trace()
        pair_embed = self.encode_pair(context_chosen, context_rejected)

        mean, log_var = self.encode_sequence(pair_embed, seq_start_end)
        mean = torch.clamp(mean, -1, 1)

        # Version 1
        _log_var = torch.clamp(log_var, -1, 1)
        if ground_truth_user_vector:
            z = (user_type * 2 - 1).unsqueeze(1).repeat(1, mean.shape[1])       # todo: change ground-truth implementation
            # z = torch.zeros_like(mean).reshape(mean.shape[0], 4, -1)
            # for idx in range(user_type.shape[0]):
            #     z[idx][int(user_type[idx])] += 1
            # z = z.reshape(mean.shape[0], -1)
        else:
            z = self.reparameterization(mean, torch.exp(0.5 * _log_var))

        # Version 2
        # act = torch.nn.Softplus()
        # std = act(log_var) * 1e-3
        # # std = torch.sigmoid(log_var) * 1e-3    # use log_var as std prediction
        # _log_var = torch.log(std ** 2)
        # z = self.reparameterization(mean, std)

        rc, rr = self.decode(target_chosen, target_rejected, z)

        return rc, rr, mean, _log_var, z

    def save_model(self, path):
        # state_dict = {
        #     "pair_encoder": self.pair_encoder.state_dict(),
        #     "sequence_encoder": self.sequence_encoder.state_dict(),
        #     "decoder": self.decoder.state_dict(),
        #     "latent_dim": self.latent_dim,
        # }
        torch.save(self, path)

    # def load_model(self, path, llm_encoder):
    #     state_dict = torch.load(path)
    #     self.pair_encoder.load_state_dict(state_dict["pair_encoder"])
    #     self.sequence_encoder.load_state_dict(state_dict["sequence_encoder"])
    #     self.decoder.load_state_dict(state_dict["decoder"])
    #     self.latent_dim = state_dict["latent_dim"]
    #     self.llm_encoder = llm_encoder


class VAETrainer(Trainer):
    def __init__(
        self, *args, lr_lambda=None, kl_loss_weight=None, use_annealing=False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lr_lambda = lr_lambda
        self.kl_loss_weight = kl_loss_weight
        self.use_annealing = use_annealing
        self.annealer = Annealer(
            total_steps=1e4, shape="cosine", baseline=0.1, cyclical=True    # todo: change total_step here
        )

    @classmethod
    def per_sample_loss(cls, rewards_chosen, rewards_rejected):
        return -nn.functional.logsigmoid(rewards_chosen - rewards_rejected)

    def loss(self, rewards_chosen, rewards_rejected):
        return torch.mean(self.per_sample_loss(rewards_chosen, rewards_rejected))

    def compute_loss(self, wrapped_model, inputs, return_outputs=False):
        model = wrapped_model  # .module
        device = model.llm_encoder.device
        batch_size = inputs["seq_start_end"].shape[0]
        if model.fixed_llm_embeddings:
            embeddings_chosen = torch.tensor(inputs["embeddings_chosen"]).to(device).bfloat16()
            embeddings_rejected = torch.tensor(inputs["embeddings_rejected"]).to(device).bfloat16()
        else:
            if model.use_causal_lm:
                input_ids = torch.concatenate(
                        [
                            inputs["input_ids_chosen"],
                            inputs["input_ids_rejected"],
                        ],
                        dim=0,
                    )
                attention_mask = torch.concatenate(
                        [
                            inputs["attention_mask_chosen"],
                            inputs["attention_mask_rejected"],
                        ],
                        dim=0,
                    )
                last_hidden_state = model.llm_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                ).hidden_states[-1]
                masked_last_hidden_state = last_hidden_state * attention_mask.unsqueeze(-1)
                token_length = torch.sum(attention_mask, dim=1)
                embeddings = torch.sum(masked_last_hidden_state, dim=1) / token_length.unsqueeze(-1)
            else:
                embeddings = model.llm_encoder(
                    torch.concatenate(
                        [
                            inputs["input_ids_chosen"],
                            inputs["input_ids_rejected"],
                        ],
                        dim=0,
                    ),
                    torch.concatenate(
                        [
                            inputs["attention_mask_chosen"],
                            inputs["attention_mask_rejected"],
                        ],
                        dim=0,
                    ),
                )[0]
            embeddings_chosen = embeddings[:batch_size]
            embeddings_rejected = embeddings[batch_size:]

        if model.fixed_contexts:
            contexts_embeddings_chosen = torch.tensor(inputs["contexts_embeddings_chosen"]).to(device).bfloat16()
            contexts_embeddings_rejected = torch.tensor(inputs["contexts_embeddings_rejected"]).to(device).bfloat16()
        else:
            if model.use_causal_lm:
                last_hidden_state_chosen = model.llm_encoder(
                    input_ids=inputs["contexts_input_ids_chosen"],
                    attention_mask=inputs["contexts_attention_mask_chosen"],
                    output_hidden_states=True
                ).hidden_states[-1]
                masked_last_hidden_state_chosen = last_hidden_state_chosen * inputs[
                    "contexts_attention_mask_chosen"].unsqueeze(-1)
                token_length_chosen = torch.sum(inputs["contexts_attention_mask_chosen"], dim=1)
                contexts_embeddings_chosen = torch.sum(masked_last_hidden_state_chosen,
                                                       dim=1) / token_length_chosen.unsqueeze(-1)

                last_hidden_state_rejected = model.llm_encoder(
                    input_ids=inputs["contexts_input_ids_rejected"],
                    attention_mask=inputs["contexts_attention_mask_rejected"],
                    output_hidden_states=True
                ).hidden_states[-1]
                masked_last_hidden_state_rejected = last_hidden_state_rejected * inputs[
                    "contexts_attention_mask_rejected"].unsqueeze(-1)
                token_length_rejected = torch.sum(inputs["contexts_attention_mask_rejected"], dim=1)
                contexts_embeddings_rejected = torch.sum(masked_last_hidden_state_rejected,
                                                         dim=1) / token_length_rejected.unsqueeze(-1)
            else:
                contexts_embeddings_chosen = model.llm_encoder(
                    inputs["contexts_input_ids_chosen"],
                    inputs["contexts_attention_mask_chosen"]
                )[0]
                contexts_embeddings_rejected = model.llm_encoder(
                    inputs["contexts_input_ids_rejected"],
                    inputs["contexts_attention_mask_rejected"]
                )[0]
        seq_start_end = inputs["seq_start_end"]
        user_type = torch.tensor(inputs["user_type"]).to(device).bfloat16()

        rewards_chosen, rewards_rejected, mean, log_var, z = model(
            embeddings_chosen,
            embeddings_rejected,
            contexts_embeddings_chosen,
            contexts_embeddings_rejected,
            seq_start_end,
            user_type,
            ground_truth_user_vector=False       # todo: set to True for debug usage
        )

        reproduction_loss = self.loss(rewards_chosen, rewards_rejected)
        if self.kl_loss_weight == 0:
            loss = reproduction_loss
            accuracy = torch.mean((rewards_chosen > rewards_rejected).float())
            if not return_outputs:
                self.log(
                    {
                        "train_loss": reproduction_loss.mean().item(),
                        "train_accuracy": accuracy.mean().item(),
                        "rewards_chosen": rewards_chosen.mean().item(),
                        "rewards_rejected": rewards_rejected.mean().item(),
                        "embeddings_chosen": embeddings_chosen.mean().item(),
                        "embeddings_rejected": embeddings_rejected.mean().item(),
                        "mean": mean.mean().item(),
                        "log_var": log_var.mean().item()
                    }
                )
        else:
            kld = -torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            if self.use_annealing:
                kld = self.annealer(kld)
                self.annealer.step()
            kld = self.kl_loss_weight * kld
            loss = reproduction_loss + kld
            accuracy = torch.mean((rewards_chosen > rewards_rejected).float())
            if not return_outputs:
                self.log(
                    {
                        "train_loss": reproduction_loss.mean().item(),
                        "train_kld": kld.mean().item(),
                        "train_accuracy": accuracy.mean().item(),
                        "rewards_chosen": rewards_chosen.mean().item(),
                        "rewards_rejected": rewards_rejected.mean().item(),
                        "embeddings_chosen": embeddings_chosen.mean().item(),
                        "embeddings_rejected": embeddings_rejected.mean().item(),
                        "mean": mean.mean().item(),
                        "log_var": log_var.mean().item()
                    }
                )
        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
                "mean": mean,
                "log_var": log_var,
                "z": z,
                "user_type": user_type,
            }
        return loss

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_lambda is not None:
            lr_lambda = partial(
                self.lr_lambda,
                num_training_steps=num_training_steps,
            )
            self.lr_scheduler = LambdaLR(optimizer, lr_lambda)
            return self.lr_scheduler
        else:
            return super().create_scheduler(num_training_steps, optimizer)

    @classmethod
    def compute_metrics(cls, eval_prediction: EvalPrediction):
        rewards_chosen, rewards_rejected, mean, log_var, z, user_type = (
            eval_prediction.predictions
        )
        rewards_chosen = torch.from_numpy(rewards_chosen)
        rewards_rejected = torch.from_numpy(rewards_rejected)
        mean = torch.from_numpy(mean)
        log_var = torch.from_numpy(log_var)

        loss = cls.per_sample_loss(rewards_chosen, rewards_rejected)
        kld = -torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        accuracy = torch.mean((loss < np.log(2)).float())

        return {
            "loss": loss.mean().item(),
            "accuracy": accuracy.item(),
            "kld": kld.item(),
            # # "total_loss": loss.mean().item() + kld.item(),
            # "user_embeddings": wandb.Table(columns=list(range(-1, z.shape[1])),
            #                                data=[[int(user_type[i])] + z[i].tolist() for i in range(z.shape[0])]),
        }


class Annealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    """

    def __init__(self, total_steps, shape, baseline=0.0, cyclical=False, disable=False):
        """
        Parameters:
            total_steps (int): Number of epochs to reach full KL divergence weight.
            shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
            baseline (float): Starting value for the annealing function [0-1]. Default is 0.0.
            cyclical (bool): Whether to repeat the annealing cycle after total_steps is reached.
            disable (bool): If true, the __call__ method returns unchanged input (no annealing).
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if disable:
            self.shape = "none"
            self.baseline = 0.0

    def __call__(self, kld):
        """
        Args:
            kld (torch.tensor): KL divergence loss
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
        """
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == "linear":
            y = self.current_step / self.total_steps
        elif self.shape == "cosine":
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == "logistic":
            exponent = (self.total_steps / 2) - self.current_step
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == "none":
            y = 1.0
        else:
            raise ValueError(
                "Invalid shape for annealing function. Must be linear, cosine, or logistic."
            )
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError(
                "Cyclical_setter method requires boolean argument (True/False)"
            )
        else:
            self.cyclical = value
        return


class VQVAE_Encoder(nn.Module):
    def __init__(self, n_embeddings, embed_dim, hidden_dim, llm_encoder, context_dim=None, commitment_cost=0.25, decay=0.999, epsilon=1e-5, fixed_contexts=False, fixed_llm_embeddings=False, use_causal_lm=False):
        super(VQVAE_Encoder, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        self.llm_encoder = llm_encoder
        self.pair_encoder = PairEncoder(context_dim, hidden_dim, embed_dim)
        self.sequence_encoder = SequenceEncoder(embed_dim, embed_dim)

        #TODO: initialise using llms
        mean_wte = llm_encoder.transformer.wte.weight.mean(0)
        weights = torch.randn(size=(n_embeddings, embed_dim), dtype=mean_wte.dtype) + mean_wte
        self.embedding = nn.Parameter(weights, requires_grad=True)
        # self.register_buffer("embedding", embedding)
        # self.register_buffer("ema_count", torch.zeros(n_embeddings))
        # self.register_buffer("ema_weight", self.embedding.clone())

        self.fixed_contexts = fixed_contexts
        self.fixed_llm_embeddings = fixed_llm_embeddings
        self.use_causal_lm = use_causal_lm

    def encode_pair(self, e_c, e_r):
        return self.pair_encoder(e_c, e_r)

    def encode_sequence(self, sequences, seq_start_end):
        e_z, _ = self.sequence_encoder(sequences, seq_start_end)
        return e_z
    
    # def discretize(self, x):
    #     M, D = self.embedding.size()
    #     x_flat = x.detach().reshape(-1, D)

    #     distances = (-torch.cdist(x_flat, self.embedding, p=2)) ** 2

    #     indices = torch.argmin(distances.float(), dim=-1)
    #     quantized = F.embedding(indices, self.embedding)
    #     quantized = quantized.view_as(x)
    #     return quantized, indices
    
    def retrieve_random_codebook(self, random_indices):
        quantized = F.embedding(random_indices, self.embedding)
        quantized = quantized.transpose(1, 3)
        
        return quantized

    def gt_forward(
        self,
        user_type,
        seq_start_end,
    ):
        quantized = self.embedding[user_type.long()]
        commitment_loss = torch.Tensor([0.0])
        codebook_loss = torch.Tensor([0.0])
        # import pdb; pdb.set_trace()
        return quantized, commitment_loss, codebook_loss, user_type #, perplexity
    def forward(
        self,
        context_chosen,
        context_rejected,
        seq_start_end,
        user_type,
        ground_truth_user_vector=True
    ):
       # import pdb; pdb.set_trace()
        if ground_truth_user_vector:
            return self.gt_forward(user_type, seq_start_end)
        pair_embed = self.encode_pair(context_chosen, context_rejected)
        x = self.encode_sequence(pair_embed, seq_start_end)
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)
        
        distances = (-torch.cdist(x_flat, self.embedding, p=2)) ** 2

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        
        #TODO: fix EMA loss
        # if self.training:
        #     self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
        #     n = torch.sum(self.ema_count)
        #     self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

        #     dw = torch.matmul(encodings.t(), x_flat)
        #     self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
        #     self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        #TODO: look at how losses flow? do we need to pass in gradients to the embeddings or the codebook loss works?
        codebook_loss = F.mse_loss(x_flat.detach(), quantized) * 0.1
        e_latent_loss = F.mse_loss(x_flat, quantized.detach())
        commitment_loss = self.commitment_cost * e_latent_loss * 0.1

        quantized = x + (quantized - x).detach()

        # avg_probs = torch.mean(encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        # import pdb; pdb.set_trace()
        return quantized, commitment_loss, codebook_loss, indices #, perplexity

    def save_model(self, path):
        torch.save(self, path)

class VQVAETrainer(VAETrainer):
    def __init__(
        self, *args, **kwargs
    ):
        super().__init__(*args, **kwargs) 
        self.pad_token = torch.tensor([50256])

    def compute_loss(self, wrapped_model, inputs, return_outputs=False):
        model = wrapped_model  # .module
        device = model.llm_encoder.device
        batch_size = inputs["seq_start_end"].shape[0]
        self.pad_token = self.pad_token.to(device)

        embeddings_chosen = model.llm_encoder.transformer.wte(inputs["input_ids_chosen"])
        embeddings_rejected = model.llm_encoder.transformer.wte(inputs["input_ids_rejected"])
        attention_mask_padding = torch.ones_like(inputs["attention_mask_chosen"][:, None, 0])
        attention_mask_chosen = torch.cat((attention_mask_padding, inputs["attention_mask_chosen"]), dim=-1)
        attention_mask_rejected = torch.cat((attention_mask_padding, inputs["attention_mask_rejected"]), dim=-1)

        seq_len_chosen = (inputs["input_ids_chosen"] != self.pad_token).sum(dim=1)
        seq_len_rejected = (inputs["input_ids_rejected"] != self.pad_token).sum(dim=1)
        seq_len = torch.cat([seq_len_chosen, seq_len_rejected])+1

        if model.fixed_contexts:
            contexts_embeddings_chosen = torch.tensor(inputs["contexts_embeddings_chosen"]).to(device).bfloat16()
            contexts_embeddings_rejected = torch.tensor(inputs["contexts_embeddings_rejected"]).to(device).bfloat16()
        else:
            if model.use_causal_lm:
                last_hidden_state_chosen = model.llm_encoder(
                    input_ids=inputs["contexts_input_ids_chosen"],
                    attention_mask=inputs["contexts_attention_mask_chosen"],
                    output_hidden_states=True
                ).hidden_states[-1]
                masked_last_hidden_state_chosen = last_hidden_state_chosen * inputs[
                    "contexts_attention_mask_chosen"].unsqueeze(-1)
                token_length_chosen = torch.sum(inputs["contexts_attention_mask_chosen"], dim=1)
                contexts_embeddings_chosen = torch.sum(masked_last_hidden_state_chosen,
                                                       dim=1) / token_length_chosen.unsqueeze(-1)

                last_hidden_state_rejected = model.llm_encoder(
                    input_ids=inputs["contexts_input_ids_rejected"],
                    attention_mask=inputs["contexts_attention_mask_rejected"],
                    output_hidden_states=True
                ).hidden_states[-1]
                masked_last_hidden_state_rejected = last_hidden_state_rejected * inputs[
                    "contexts_attention_mask_rejected"].unsqueeze(-1)
                token_length_rejected = torch.sum(inputs["contexts_attention_mask_rejected"], dim=1)
                contexts_embeddings_rejected = torch.sum(masked_last_hidden_state_rejected,
                                                         dim=1) / token_length_rejected.unsqueeze(-1)
            else:
                contexts_embeddings_chosen = model.llm_encoder(
                    inputs["contexts_input_ids_chosen"],
                    inputs["contexts_attention_mask_chosen"]
                )[0]
                contexts_embeddings_rejected = model.llm_encoder(
                    inputs["contexts_input_ids_rejected"],
                    inputs["contexts_attention_mask_rejected"]
                )[0]
        seq_start_end = inputs["seq_start_end"]
        user_type = torch.tensor(inputs["user_type"]).to(device).bfloat16()

        quantized, commitment_loss, codebook_loss, indices = model(
            contexts_embeddings_chosen,
            contexts_embeddings_rejected,
            seq_start_end,
            user_type,
            ground_truth_user_vector=False       # todo: set to True for debug usage
        )
        quantized = quantized.to(device).bfloat16()

        embeddings_chosen = torch.cat((quantized[:, None], embeddings_chosen), dim=1)
        embeddings_rejected = torch.cat((quantized[:, None], embeddings_rejected), dim=1)
        
        output_dict = model.llm_encoder(
            inputs_embeds=torch.concatenate(
                [
                    embeddings_chosen,
                    embeddings_rejected,
                ],
                dim=0,
            ),
            attention_mask=torch.concatenate(
                [
                    attention_mask_chosen,
                    attention_mask_rejected,
                ],
                dim=0,
            ),
            return_dict=True,
            output_hidden_states=True
        )

        batch_indices = torch.arange(len(seq_len)).to(device)
        hidden_states = output_dict["hidden_states"][-1][batch_indices, seq_len]
        rewards = model.llm_encoder.score(hidden_states)

        # rewards = rewards[0]
        rewards_chosen = rewards[:batch_size]
        rewards_rejected = rewards[batch_size:]

        reproduction_loss = self.loss(rewards_chosen, rewards_rejected)
        loss =  reproduction_loss # + commitment_loss + codebook_loss
        
        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
                "commitment_loss": commitment_loss,
                "codebook_loss": codebook_loss,
                "z": quantized,
                "user_type": user_type,
                "indices": indices,
                "embeddings": model.embedding
            }
        else:
            accuracy = torch.mean((rewards_chosen > rewards_rejected).float())
            self.log(
                {
                "rewards_chosen": rewards_chosen.mean().item(),
                "rewards_rejected": rewards_rejected.mean().item(),
                "train_commitment_loss": commitment_loss.item(),
                "train_codebook_loss": codebook_loss.item(),
                "train_loss": loss.item(),
                "train_reproduction_loss": reproduction_loss.item(),
                "train_accuracy": accuracy
            }   
            )
        return loss

    @classmethod
    def compute_metrics(cls, eval_prediction: EvalPrediction):
        rewards_chosen, rewards_rejected, commitment_loss, codebook_loss, z, user_type, indices, embeddings = (
            eval_prediction.predictions
        )
        rewards_chosen = torch.from_numpy(rewards_chosen)
        rewards_rejected = torch.from_numpy(rewards_rejected)

        loss = cls.per_sample_loss(rewards_chosen, rewards_rejected)
        accuracy = torch.mean((rewards_chosen > rewards_rejected).float())#torch.mean((loss < np.log(2)).float())
        
        # import pdb; pdb.set_trace()
        embeddings_table = wandb.Table(columns=list(range(z.shape[1])), data=embeddings)
        
        unique_users = np.unique(user_type)
        fig, axs = plt.subplots(1, len(unique_users), figsize=(20,5))
        for i, uid in enumerate(unique_users):
            user_indices = indices[np.argwhere(user_type == uid)]
            axs[i].hist(user_indices)
            axs[i].set_title(f"User {i}")
        im = wandb.Image(fig)

        return {
            "reproduction_loss": loss.mean().item(),
            "accuracy": accuracy.item(),
            "commitment_loss": commitment_loss.mean().item(),
            "codebook_loss": codebook_loss.mean().item(),
            "embeddings_table": embeddings_table,
            "latents": im
        }