import math
from functools import partial

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import torch.nn as nn
from transformers import Trainer, EvalPrediction


class PairEncoder(nn.Module):
    """
    Model to encode pair of accepted and rejected responses
    """

    def __init__(self, embed_dim, output_dim, hidden_dim):
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
        self.mean_layer = nn.Linear(input_dim, latent_dim)
        self.log_var_layer = nn.Linear(input_dim, latent_dim)

    def forward(
        self, sequences, seq_start_end
    ):  # (C_1+C_2+...+C_n, D), [(0, C_1), (C_1, C_1+C_2), ..., (C_1+...+C_n-1, C_1+...+C_n)]
        outputs = []
        for _, (start, end) in enumerate(seq_start_end):
            context = sequences[start:end]  # C_i x D
            transformed_seq = self.linear(context)  # C_i x D'
            attention_scores = torch.matmul(
                transformed_seq, transformed_seq.transpose(0, 1)
            )  # C_i x C_i
            attention_scores = attention_scores / (context.shape[-1] ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)  # C_i x C_i
            weighted_values = torch.matmul(attention_weights, context)  # C_i x D
            output = torch.sum(weighted_values, dim=0)  # D
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)  # n x D

        mean = self.mean_layer(outputs)
        log_var = self.log_var_layer(outputs)
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
        xc = torch.cat([xc, z], dim=1)
        xr = torch.cat([xr, z], dim=1)
        rc = self._model(xc)
        rr = self._model(xr)
        return rc, rr


class VAEModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, latent_dim, llm_encoder, use_fixed_contexts=False):
        super(VAEModel, self).__init__()
        self.llm_encoder = llm_encoder
        self.pair_encoder = PairEncoder(embed_dim, hidden_dim, latent_dim)
        self.sequence_encoder = SequenceEncoder(latent_dim, latent_dim)
        self.decoder = Decoder(embed_dim + latent_dim, hidden_dim)

        self.latent_dim = latent_dim
        self.use_fixed_contexts = use_fixed_contexts

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(mean.device)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
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
    ):
        # import pdb; pdb.set_trace()
        pair_embed = self.encode_pair(context_chosen, context_rejected)

        mean, log_var = self.encode_sequence(pair_embed, seq_start_end)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        rc, rr = self.decode(target_chosen, target_rejected, z)

        return rc, rr, mean, log_var

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
            total_steps=1e4, shape="cosine", baseline=0.1, cyclical=True
        )

    @classmethod
    def per_sample_loss(cls, rewards_chosen, rewards_rejected):
        return -nn.functional.logsigmoid(rewards_chosen - rewards_rejected)

    def loss(self, rewards_chosen, rewards_rejected):
        return torch.mean(self.per_sample_loss(rewards_chosen, rewards_rejected))

    def compute_loss(self, model, inputs, return_outputs=False):
        # import pdb; pdb.set_trace()
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

        batch_size = inputs["input_ids_chosen"].shape[0]
        target_chosen = embeddings[:batch_size]
        target_rejected = embeddings[batch_size:2*batch_size]

        if "embeddings_context_chosen" not in inputs.keys():
            # context = embeddings[2*batch_size:]
            # context_chosen = context[: len(context) // 2]
            # context_rejected = context[len(context) // 2 :]
            context_chosen = model.llm_encoder(
                inputs["input_ids_context_chosen"], inputs["attention_mask_context_chosen"])[0]
            context_rejected = model.llm_encoder(
                inputs["input_ids_context_rejected"], inputs["attention_mask_context_rejected"])[0]
        else:
            context_chosen = torch.tensor(inputs["embeddings_context_chosen"]).to(embeddings.device)
            context_rejected = torch.tensor(inputs["embeddings_context_rejected"]).to(embeddings.device)
        seq_start_end = inputs["seq_start_end"]

        rewards_chosen, rewards_rejected, mean, log_var = model(
            target_chosen,
            target_rejected,
            context_chosen,
            context_rejected,
            seq_start_end,
        )

        reproduction_loss = self.loss(rewards_chosen, rewards_rejected)
        kld = -torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        if self.use_annealing:
            kld = self.annealer(kld)
            self.annealer.step()
        kld = self.kl_loss_weight * kld
        loss = reproduction_loss + kld
        accuracy = torch.mean((rewards_chosen > rewards_rejected).float())
        self.log(
            {
                "train_loss": reproduction_loss.mean().item(),
                "train_kld": kld.mean().item(),
                "train_accuracy": accuracy.mean().item(),
            }
        )

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
                "mean": mean,
                "log_var": log_var,
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
        rewards_chosen, rewards_rejected, mean, log_var = (
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
            "total_loss": loss.mean().item() + kld.item(),
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
