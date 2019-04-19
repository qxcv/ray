from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools as it

from gym.spaces import Box
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import ray
from ray.rllib.evaluation.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.evaluation.policy_graph import PolicyGraph
from ray.rllib.evaluation.torch_policy_graph import TorchPolicyGraph


def make_fc_net(in_size, out_size, hiddens=[400, 300],
                inner_activation="relu"):
    layer_num = 1
    layers = collections.OrderedDict()
    in_dim = in_size
    for hidden in hiddens:
        layers['fc%d' % layer_num] = nn.Linear(in_dim, hidden)
        layers['relu%d' % layer_num] = nn.ReLU()
        in_dim = hidden
        layer_num += 1
    layers['fc%d' % layer_num] = nn.Linear(in_dim, out_size)
    return nn.Sequential(layers)


class PolicyNetwork(nn.Module):
    def __init__(self, dim_state, dim_action, max_action, hiddens=[400, 300]):
        super().__init__()
        self.max_action = max_action
        assert self.max_action > 0
        self.pi_unscaled = make_fc_net(dim_state,
                                       dim_action,
                                       hiddens,
                                       inner_activation="relu")

    def forward(self, state):
        pi_out = self.pi_unscaled.forward(state)
        # ensure action is in (-max_action, max_action)
        rescaled = pi_out.new_tensor(self.max_action) * torch.tanh(pi_out)
        return rescaled


class TwinQNetwork(nn.Module):
    def __init__(self, dim_state, dim_action, hiddens=[400, 300]):
        super().__init__()
        fc_args = dict(in_size=dim_state + dim_action,
                       out_size=1,
                       hiddens=hiddens,
                       inner_activation="relu")
        self.q1 = make_fc_net(**fc_args)
        self.q2 = make_fc_net(**fc_args)

    def forward(self, state, action):
        """Forward prop through both Q-networks, returning two Q-values."""
        state_action = torch.cat((state, action), -1)
        return self.q1(state_action), self.q2(state_action)

    def forward_q_pi(self, state, action):
        """Forward prop through only the first Q network. This is useful when
        doing policy updates."""
        state_action = torch.cat((state, action), -1)
        return self.q1(state_action)


class TD3Loss(nn.Module):
    def __init__(self, policy, q_networks, target_q_networks, target_policy,
                 target_noise, noise_clip, max_action, gamma):
        super().__init__()
        self.policy = policy
        self.q_networks = q_networks
        self.target_q_networks = target_q_networks
        self.target_policy = target_policy
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.max_action = max_action

    def forward(self, obs, act, rew, obs_next, dones):
        act_noise = torch.normal(torch.zeros_like(act), self.target_noise) \
            .clamp(-self.noise_clip, self.noise_clip)
        # TODO: before I was clamping target_acts to sit in valid range of
        # [-1,1]; was that a bad idea?
        target_acts = self.target_policy(obs_next) + act_noise
        target_next_q1, target_next_q2 = self.target_q_networks(
            obs_next, target_acts)
        min_q_next = torch.min(target_next_q1, target_next_q2)
        assert target_next_q1.shape == target_next_q2.shape
        assert min_q_next.shape == target_next_q2.shape
        dones_f = dones.float()
        targets = rew + self.gamma * (1 - dones_f) * min_q_next
        targets = targets.detach()
        next_q1, next_q2 = self.q_networks(obs, act)
        q_loss = F.mse_loss(next_q1, targets) + F.mse_loss(next_q2, targets)
        # FIXME: is there some way to prevent policy_loss.backward() from
        # touching grads for parameters of self.q_networks?
        policy_loss = -self.q_networks \
            .forward_q_pi(obs, self.policy(obs)).mean()
        return q_loss, policy_loss


class TD3TorchPolicyGraph(TorchPolicyGraph):
    # we don't need a TD3Postprocessing base class at the moment because we
    # don't support N-step returns or any of the other fancy things that
    # require post-processing
    def __init__(self, observation_space, action_space, config):
        config = dict(ray.rllib.agents.td3.td3.DEFAULT_CONFIG, **config)
        if not isinstance(action_space, Box):
            raise UnsupportedSpaceException(
                "TD3 only supports continuous vector-valued actions (i.e Box)")
        self.max_action = max(action_space.high)
        if not np.allclose(-action_space.low, self.max_action) \
           or not np.allclose(action_space.high, self.max_action):
            # FIXME: this is a dumb restriction so that max_action is valid; I
            # should fix that by having a pre/post processing step that
            # rescales all actions into [-1,1].
            raise UnsupportedSpaceException(
                "action space lower/upper bounds should be negative of each "
                "other, but lower is %s and upper is %s" %
                (action_space.low, action_space.high))
        self.config = config
        self.test_mode = self.config['test_mode']
        self.dim_action, = action_space.shape
        self.dim_state, = observation_space.shape
        # initially we do TOTAL EXPLORATION
        self.exploration_fraction = 1.0
        # we use this to decide whether to do a policy update
        self.num_q_updates = 0

        self.q_networks = TwinQNetwork(self.dim_state, self.dim_action)
        self.policy = PolicyNetwork(self.dim_state, self.dim_action,
                                    self.max_action)
        # targets get updated slooowly with Polyak
        self.target_q_networks = TwinQNetwork(self.dim_state, self.dim_action)
        self.target_policy = PolicyNetwork(self.dim_state, self.dim_action,
                                           self.max_action)
        self._copy_weights(self.target_q_networks, self.q_networks)
        self._copy_weights(self.target_policy, self.policy)
        for param in it.chain(self.target_q_networks.parameters(),
                              self.target_policy.parameters()):
            param.grad = None
            param.requires_grad = False

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(),
                                                 lr=self.config["pi_lr"])
        self.q_optimizer = torch.optim.Adam(self.q_networks.parameters(),
                                            lr=self.config["q_lr"])

        # TODO: remove the whole TD3Loss thing. It should just be a method of
        # this class, not a separate nn.Module (current design is artefact of
        # me aping the A3C implementation).
        loss = TD3Loss(policy=self.policy,
                       q_networks=self.q_networks,
                       target_q_networks=self.target_q_networks,
                       target_policy=self.target_policy,
                       target_noise=self.config['target_noise'],
                       noise_clip=self.config['target_noise_clip'],
                       gamma=self.config['gamma'],
                       max_action=self.max_action)
        combined_model = torch.nn.ModuleDict(
            collections.OrderedDict([
                ('q_networks', self.q_networks),
                ('policy', self.policy),
                ('target_q_networks', self.target_q_networks),
                ('target_policy', self.target_policy),
            ]))
        super().__init__(observation_space,
                         action_space,
                         combined_model,
                         loss,
                         loss_inputs=[
                             SampleBatch.CUR_OBS, SampleBatch.ACTIONS,
                             SampleBatch.REWARDS, SampleBatch.NEXT_OBS,
                             SampleBatch.DONES
                         ])

    @override(TorchPolicyGraph)
    def optimizer(self):
        # dummy; this doesn't do anything because we use self.policy_optimizer
        # and self.q_optimizer instead of self._optimizer
        return None

    @override(TorchPolicyGraph)
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        with self.lock:
            with torch.no_grad():
                # print('[ROLLOUTS] computing some actions (eps=%.3g)' %
                #       self.exploration_fraction)
                ob = torch.from_numpy(np.array(obs_batch)).float()
                # normally we just compute noisy actions
                actions = self.policy(ob).detach()
                action_noise = self.config['act_noise'] * torch.normal(
                    torch.zeros_like(actions), torch.ones_like(actions))
                if self.test_mode:
                    action_noise *= 0
                actions = actions + action_noise
                # sometimes we also do exploration by re-sampling a subset of
                # actions at uniform
                if self.exploration_fraction > 0 and not self.test_mode:
                    n_actions = actions.shape[0]
                    eps = self.exploration_fraction
                    rand_mask = np.random.randn(n_actions) < eps
                    to_randomise, = np.argwhere(rand_mask).T
                    if len(to_randomise) > 0:
                        # need len() > 0 check because np.stack() complains
                        # when you give it no arrays :(
                        random_acts = np.stack(
                            [self.action_space.sample() for i in to_randomise],
                            axis=0)
                        actions[to_randomise] = actions.new_tensor(random_acts)
                # TODO: support recurrent policies (I think that's what the
                # model output state thing is for; I've left it as empty list)
                return (actions.numpy(), [], self.extra_action_out(actions))

    @override(TorchPolicyGraph)
    def compute_gradients(self, postprocessed_batch):
        with self.lock:
            loss_in = []
            for key in self._loss_inputs:
                # TODO: push this fix into TorchPolicyGraph.py. It's necessary
                # to deal with bool-valued Numpy arrays, which
                # torch.from_numpy() will choke on.
                value = postprocessed_batch[key]
                if value.dtype.name == 'bool':
                    value = value.astype('uint8')
                loss_in.append(torch.from_numpy(value))
            q_loss, policy_loss = self._loss(*loss_in)
            grad_dict = {'policy': [], 'q_networks': []}
            # DANGER: return values are just references; calling zero_grad
            # later will modify the values.
            # Also, be careful with policy objective: it backprops though Q
            # network, so policy_loss.backward() will also change Q-network
            # gradients (unfortunately).
            self.policy_optimizer.zero_grad()
            self.q_optimizer.zero_grad()
            policy_loss.backward()
            for p in self.policy.parameters():
                grad_dict['policy'].append(p.grad.data.numpy())
            # now Q grad
            self.q_optimizer.zero_grad()
            q_loss.backward()
            for p in self.q_networks.parameters():
                grad_dict['q_networks'].append(p.grad.data.numpy())
            return grad_dict, {}

    @override(PolicyGraph)
    def apply_gradients(self, grad_dict):
        with self.lock:
            # XXX remove this later
            assert len(grad_dict['policy']) \
                == len(list(self.policy.parameters()))
            for g, p in zip(grad_dict['policy'], self.policy.parameters()):
                p.grad = torch.from_numpy(g)
            assert len(grad_dict['q_networks']) \
                == len(list(self.q_networks.parameters()))
            for g, p in zip(grad_dict['q_networks'],
                            self.q_networks.parameters()):
                p.grad = torch.from_numpy(g)
            # print('[CRITIC] updating')
            self.q_optimizer.step()
            self.num_q_updates += 1

            # policy update delay (the zero_grad() calls in compute_gradients()
            # should ensure that we don't accumulate gradients accidentally)
            if self._should_update_pol_and_target:
                # print('[POLICY] updating')
                self.policy_optimizer.step()

            return {}

    @property
    def _should_update_pol_and_target(self):
        return self.config["policy_delay"] == 0 \
            or self.num_q_updates % self.config["policy_delay"] == 0

    @override(TorchPolicyGraph)
    def get_initial_state(self):
        # TODO: support recurrent policies, as above
        return []

    def set_epsilon(self, eps):
        self.exploration_fraction = eps

    def set_test_mode(self, value):
        self.test_mode = value

    def _copy_weights(self, model_to_update, model_to_update_towards):
        # copy weights from one module (second argument) to another (first
        # argument)
        model_to_update.load_state_dict(model_to_update_towards.state_dict())

    def _do_target_update(self, current_target_params, new_params):
        polyak = self.config['polyak']
        for current, new in zip(current_target_params, new_params):
            # in-place is fine because we don't need grads on targets
            current.data = polyak * current.data + (1 - polyak) * new.data
            assert current.grad is None
            assert current.requires_grad is False

    def update_target(self):
        if self._should_update_pol_and_target:
            # print('[TARGET] updating')
            self._do_target_update(self.target_q_networks.parameters(),
                                   self.q_networks.parameters())
            self._do_target_update(self.target_policy.parameters(),
                                   self.policy.parameters())
