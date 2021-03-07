from src.models import Nature
from src.testfunc import test_func_bound
from src.testfunc import TestEnv
import tensorflow as tf
import numpy as np


nonlin_dict = {
    'elu': tf.nn.elu,
    'relu': tf.nn.relu,
    'sigmoid': tf.nn.sigmoid,
    'tanh': tf.nn.tanh
}

# If you add a new network you should add "string --> class" mapping here.
network_dict = {
    "Nature": Nature
}

class Policy(object):
    def __init__(self, env, network, nonlin_name):
        # Size of the virtual batch
        self.vb_size = 128
        # Maximum length of each episode (in steps)
        self.max_episode_len = 25000
        # Nonlinearity used in the network
        self.nonlin = nonlin_dict[nonlin_name]

        self.vb = None

        # Apply standard reinforcement learning preprocessing pipeline.
        # to theinput frames
        self.env = env

        # Shapes of the input and output of the network.
        self.in_shape = list(self.env.observation_space.shape)
        self.out_num = self.env.action_space.n

        # is_training is set to True when we forward virtual batch at the beginning of each iteration.
        # This updates normalization statistics, is_training is set to False during episode evaluation.
        self.is_training = tf.placeholder(tf.bool, [], name="is_training")

        # Placeholder for the input state
        self.input_placeholder = tf.placeholder(tf.float32, [None] + self.in_shape, name='Input')

        # Create session for 1 CPU
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        self.sess = tf.Session(config=tf_config)

        # Output from the network, computed values for each action.
        # In each step we will execute action with the maximum value.
        NetworkClass = network_dict[network]
        self.action_op = NetworkClass(self.input_placeholder, self.out_num, self.nonlin, self.is_training)

        # Tensorflow operation to compute and save virtual batch normalization statistics.
        self.vb_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        self.sess.run(tf.global_variables_initializer())

        # Those variables will be updated using ES algorithm.
        self.parameters = tf.trainable_variables()

        # We need to save parameter shapes. Those are used when extracting parameters from flat array.
        self.parameter_shapes = [Policy.shape2int(p) for p in self.parameters]

        # Operations to assign new values to the parameters.
        self.parameters_placeholders = [tf.placeholder(dtype=tf.float32, shape=s) for s in self.parameter_shapes]
        self.set_parameters_ops = [par.assign(placeholder) for par, placeholder in
                                   zip(self.parameters, self.parameters_placeholders)]

    @staticmethod
    def shape2int(x):
        s = x.get_shape()
        return [int(si) for si in s]

    def get_vb(self):
        # Plays the game using random actions,
        # after each step if game is not done there is 1% probability
        # that the state will be saved in the reference batch.
        # Ends when enough samples were collected (128 by default).
        vb = []
        self.env.reset()
        while len(vb) < self.vb_size:
            # Apply random action and with 1% chance save this state.
            state, _, done, _ = self.env.step(np.random.randint(self.out_num))
            if done:
                self.env.reset()
            elif np.random.rand() < 0.01:
                vb.append(np.array(state))

        vb = np.asarray(vb)
        self.set_vb(vb)
        return vb

    def set_vb(self, vb):
        self.vb = vb
        # Computes normalization statistics using virtual batch.
        self.sess.run(self.vb_op, feed_dict={self.input_placeholder: self.vb, self.is_training: True})

    def get_parameters(self):
        # Extracts parameters from the network and returns flat 1D array with parameter values.
        parameters = self.sess.run(self.parameters)
        return np.concatenate([p.flatten() for p in parameters])

    def set_parameters(self, parameters):
        # Sets network parameters from flat 1D array with parameter values.
        feed_dict = {}
        current_position = 0
        for parameter_placeholder, shape in zip(self.parameters_placeholders, self.parameter_shapes):
            length = np.prod(shape)
            feed_dict[parameter_placeholder] = parameters[current_position:current_position+length].reshape(shape)
            current_position += length
        self.sess.run(self.set_parameters_ops, feed_dict=feed_dict)

        # We need to update normalization statistics each time new parameters are set.
        if self.vb is not None:
            self.sess.run(self.vb_op, feed_dict={self.input_placeholder: self.vb, self.is_training: True})

    def rollout(self, seed=None,render=False):
        # Evaluates the policy for up to max_episode_len steps.
        if seed is not None:
            self.env.seed(int(seed))
        ob = self.env.reset()
        ob = np.asarray(ob)
        t = 0
        rew_sum = 0
        for _ in range(self.max_episode_len):
            ac = self.sess.run(self.action_op, feed_dict={self.input_placeholder: [ob], self.is_training: False})
            ob, rew, done, _ = self.env.step(np.argmax(ac))
            ob = np.asarray(ob)
            rew_sum += rew
            t += 1
            if render:
                self.env.render()
            if done:
                break

        return rew_sum, t
    

class FuncPolicy(object):
    """
    针对函数优化问题中的解表示，采用Policy这个名字只是
    为了和演化强化学习算法的代码相对照
    """
    def __init__(self, D, problem_t, rank, randomSeed):
        super().__init__()
        self.D = D
        self.problem_t = problem_t
        self.testenv = TestEnv(D, problem_t)
        bound = test_func_bound[problem_t]
        self.rs = np.random.RandomState(rank+randomSeed)
        self.parameters = self.rs.uniform(bound[0], bound[1], D)

    def set_parameters(self, params):
        self.parameters = params

    def get_parameters(self):
        return self.parameters

    def rollout(self, seed = None):
        fitness = self.testenv.get_fitness(self.parameters) + np.random.randn()
        return fitness , 1


class SupervisePolicy(Policy):
    '''for Pitfall game
    rollout function return precise of 1000 samples
    '''
    def __init__(self,env, network, nonlin_name, humanBuf):
        super().__init__(env, network, nonlin_name)
        self.humanBuf = humanBuf

    def rollout(self,point=2800,seed=None):
        # Evaluate the policy for 1000 step samples
        # return accuracy = nums of true samples
        if seed is not None:
            self.env.seed(seed)
        self.env.reset()
        acts = []
        accuracy = 0
        self.humanBuf.point = point
        for i in range(point,self.humanBuf.length):
            acts = self.humanBuf.act(i)
            ob, reward, done, info = self.humanBuf.step(i-1)
            # ob = np.asarray(ob)
            # # frameskip
            # obs = ob.copy()
            # for _ in range(3):
            #     obs = np.concatenate(obs,ob)
            ac = self.sess.run(self.action_op, feed_dict={self.input_placeholder: [ob], self.is_training: False})
            pred_act = np.argmax(ac)
            if pred_act == acts:
                accuracy += 1
        return accuracy, self.humanBuf.length-2800

    def set_seed(self,seed):
        self.env.seed(seed)

    def simulate(self,seed,point = 2800,render=False):
        # Evaluates the policy for up to max_episode_len steps.
        if seed is not None:
            self.env.seed(seed)
        ob = self.env.reset()
        ob = np.asarray(ob)
        rew_sum = 0
        self.humanBuf.point = 0
        action_list = []
        for t in range(self.max_episode_len):
            if t <= point:
                action = self.humanBuf.act(t)
            else:
                if t < len(self.humanBuf.data):
                    ob, rew1, done1, _1 = self.humanBuf.step(t)
                ac = self.sess.run(self.action_op, feed_dict={self.input_placeholder: [ob], self.is_training: False}
                )
                action = np.argmax(ac)
            action_list.append(action)
            ob, rew, done, _ = self.env.step(action)
            ob = np.asarray(ob)
            rew_sum += rew
            if rew_sum > 0:
                break
            if render:
                self.env.render()
            if done:
                break

        return rew_sum, t, action_list 
       
# Wrapper for setting some parameters to 0 in specified layers.
# Was used to test how well the network works if we decrease the number of parameters.
class DropoutPolicy(Policy):
    def __init__(self, env, network, nonlin_name, seed=1, keep=0.01, layers=(6,)):
        super().__init__(env, network, nonlin_name)
        state = np.random.RandomState(seed)

        # Layers for which we apply dropout (index).
        self.layers = layers

        parameters = self.sess.run(self.parameters)

        # Mask will contain indices in par vector for which we do not apply dropout
        self.mask = []
        offset = 0
        for i, p in enumerate(parameters):
            size = np.prod([int(s) for s in p.shape])

            if i in layers:
                indices = state.choice(range(offset, offset + size), size=int(size*keep), replace=False)
                self.mask.extend(indices)
            else:
                self.mask.extend(range(offset, offset + size))
            offset += size

        # Ignore some parameters (Will set them to 0 and never update)
        self._parameters = np.zeros(offset)
        self._parameters[self.mask] = super().get_parameters()[self.mask]
        super().set_parameters(self._parameters)

    def get_parameters(self):
        return self._parameters[self.mask]

    def set_parameters(self, parameters):
        self._parameters[self.mask] = parameters
        super().set_parameters(self._parameters)
