"""Main DQN agent."""
import shutil
import gym
import os
import time
import numpy as np
from gym import wrappers
from PIL import Image

from keras.models import Model
import tensorflow as tf
import math


class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """

    def __init__(self,
                 q_network,
                 q_target_network,
                 preprocessor,
                 test_preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 logdir,
                 save_freq,
                 evaluate_freq,
                 test_num_episodes,
                 env_name,
                 model_name,
                 z):
        self.model = q_network
        self.model_target = q_target_network
        self.preprocessor = preprocessor
        self.test_preprocessor = test_preprocessor
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.logdir = logdir
        self.save_freq = save_freq
        self.evaluate_freq = evaluate_freq
        self.test_num_episodes = test_num_episodes
        self.env_name = env_name
        self.model_name = model_name
        self.z = z


    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.

        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True

        # with tf.Session(config=config) as sess:

        self.sess = tf.Session(config=config)

        self.optimizer = optimizer
        self.loss_func = loss_func
        self.y_pred = self.model.outputs[0]
        self.input = self.model.inputs[0]
        self.y_true = tf.placeholder(tf.float32, shape=self.y_pred.get_shape(), name='y_true')
        self.mask = tf.placeholder(tf.float32, self.y_pred.get_shape(), name="mask")
        self.y_pred_target = self.model_target.outputs[0]
        self.input_target = self.model_target.inputs[0]

        self.loss = self.loss_func(self.y_true, tf.multiply(self.y_pred, self.mask))

        self.train_op = self.optimizer.minimize(self.loss)
        self.init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        # self.test_reward = tf.placeholder(tf.float32, shape=(), name='test_reward')

        self.reward_record = tf.Variable(0., name='global_r', trainable=False, dtype=tf.float32)
        self.set_reward = tf.placeholder(tf.float32, shape=[])

        self.assign_reward_op = tf.assign(self.reward_record, self.set_reward)

        self.sync_model_op = []
        self.sync_model_input = []
        for i in range(len(self.model.trainable_weights)):
            self.sync_model_input.append(tf.placeholder('float32', self.model.trainable_weights[i].get_shape().as_list()))
            self.sync_model_op.append(self.model_target.trainable_weights[i].assign(self.sync_model_input[i]))

        # init Q-network
        # check if checkpoint exists
        if not os.path.exists(self.logdir):
            os.mkdir(self.logdir)

        ckpt_dir = os.path.join(self.logdir, 'checkpoints')
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
            # self.sess.run(self.init_op)
            # print("init q - network")
        if len(os.listdir(ckpt_dir)) > 0:
            last_iter = 0
            for past_iter in os.listdir(ckpt_dir):
                if int(past_iter) > last_iter:
                    last_iter = int(past_iter)
            self.model.load_weights(filepath=os.path.join(ckpt_dir, str(last_iter)))
            # self.sess.run(self.init_op)
            print("Restore model from {0}".format(os.path.join(ckpt_dir, str(last_iter))))

        # copy model

        self.model_target.set_weights(self.model.get_weights())



        # set up logger
        # self.reward_summary = tf.summary.scalar('test_reward', self.test_reward)
        self.r_summary = tf.summary.scalar('reward_test', self.reward_record)
        self.loss_summary = tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge_all()
        self.merged_r = tf.summary.merge([self.r_summary])
        self.file_writer = tf.summary.FileWriter(self.logdir)

        self.model_saver = tf.train.Saver(self.model.trainable_weights)

        self.sess.run(self.init_op)
        print("compile finished")



    def calc_target_q_values(self, state):
        q_values = self.sess.run(self.y_pred_target, feed_dict={self.input_target: state})
        return q_values

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        # q_values = self.sess.run(self.y_pred, feed_dict={self.input: np.expand_dims(state, axis=0)})
        q_values = self.sess.run(self.y_pred, feed_dict={self.input: state})
        return q_values

    def select_action_train(self, q_values):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        return self.policy.select_action(q_values, True)

    def select_action(self, q_values):
        return self.policy.select_action(q_values, False)

    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        pass

    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment.
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        # init monitor for testing
        state = env.reset()
        iter_epi = 0
        print("burn in ....")
        for i in range(self.num_burn_in):
            processed_state = self.preprocessor.process_state_for_network(state)
            action = np.random.randint(0, self.policy.num_actions)
            next_state, reward, is_terminal, debug_info = env.step(action)
            reward = self.preprocessor.process_reward(reward)
            if iter_epi > 3:
                self.memory.append(state=processed_state.astype(np.uint8), action=action, reward=reward)

            if is_terminal or iter_epi >= max_episode_length:
                print('game ends! reset now.')
                processed_next_state = self.preprocessor.process_state_for_network(next_state)
                self.memory.end_episode(processed_next_state, is_terminal)
                state = env.reset()
                iter_epi = 0
                self.preprocessor.reset()
                continue
            if i % (self.num_burn_in / 10) == 0:
                print("Burn in : {0}% done".format(i / self.num_burn_in))
            iter_epi = iter_epi+1
            state = next_state


        for i in range(1,num_iterations):
            start_time = time.time()

            processed_state = self.preprocessor.process_state_for_network(state)

            q_values = self.calc_q_values(np.expand_dims(processed_state/self.z, axis=0))
            action = self.select_action_train(q_values)

            next_state, reward, is_terminal, debug_info = env.step(action)
            if reward > 0 or reward<0:
                print('reward: {0}'.format(reward))
            # env.render()
            reward = self.preprocessor.process_reward(reward)
            # todo put into memory
            if iter_epi>3:
                self.memory.append(state=processed_state.astype(np.uint8), action=action, reward=reward)

            if is_terminal or iter_epi >= max_episode_length:
                print('game ends! reset now.')
                processed_next_state = self.preprocessor.process_state_for_network(next_state)
                self.memory.end_episode(final_state=processed_next_state, is_terminal=is_terminal)
                state = env.reset()
                iter_epi = 0
                self.preprocessor.reset()
                continue

            if i%self.train_freq==0:
                batch = self.memory.sample(self.batch_size)

                next_target_q_value = self.calc_target_q_values(batch["next_state"]/self.z)
                # print(next_target_q_value)

                mask = np.zeros([self.batch_size, env.action_space.n])
                for x in range(self.batch_size):
                    if batch["is_terminal"][x]==0:
                        mask[x,batch["action"][x]] = 1

                if self.model_name == "ddqn" or self.model_name == "linear_ddqn":
                    next_q_value = self.calc_q_values(batch["next_state"] / self.z)
                    next_actions = np.argmax(next_q_value,1)
                    # print("next actions")
                    # print(next_actions)
                    target_rewards = batch["reward"]
                    # print(target_rewards)
                    target_is_terminal = batch["is_terminal"]
                    q_values_target = np.zeros([self.batch_size, env.action_space.n])
                    # print("target values")
                    for x in range(self.batch_size):
                        q_values_target[x, batch["action"][x]] = (target_rewards[x]
                        + self.gamma * (1-target_is_terminal[x]) * next_target_q_value[x,next_actions[x]])
                        # print(q_values_target[x, batch["action"][x]])

                else:
                    target = batch["reward"] + self.gamma * np.multiply(1 - batch["is_terminal"],
                                                                        next_target_q_value.max(axis=1))
                    q_values_target = np.zeros([self.batch_size, env.action_space.n])
                    for x in range(self.batch_size):
                        q_values_target[x, batch["action"][x]] = target[x]

                loss_summary, loss, _ = self.sess.run([self.loss_summary, self.loss, self.train_op],
                                                        feed_dict={self.y_true: q_values_target,
                                                             self.input: batch["state"]/self.z,
                                                             self.mask: mask})
                self.file_writer.add_summary(loss_summary, i)
            # update target policy
            if i % self.target_update_freq == 0:
                print("update model")
                # print(self.model_target.get_weights()[2][1])
                for k in range(len(self.model.trainable_weights)):
                    self.sync_model_op[k].eval(session=self.sess, feed_dict={self.sync_model_input[k]: self.model.trainable_weights[k].eval(session=self.sess)})

            iter_epi = iter_epi+1
            duration = time.time() - start_time

            if i % 50 == 0:
                print('q_values: {0}'.format(q_values))
                # print('q_values - q_target: {0}'.format(max(abs(q_values - q_values_target))))
                print('iter= {0}, loss = {1:.4f}, ({2:.2f} sec/iter)'.format(i, loss, duration))
            if i > 0 and i % self.save_freq == 0:
                save_dir = os.path.join(self.logdir, 'checkpoints', str(i))
                # self.model.save_weights(save_dir)
                save_path = self.model_saver.save(self.sess, save_dir)
                print("Saving model at {0}".format(save_path))
            if i > 0 and i % self.evaluate_freq == 0:

                average_test_reward = self.evaluate(env=gym.make(self.env_name),
                                                    num_episodes=self.test_num_episodes, iter=i)
                print('Evaluation at iter {0}: average reward for 20 episodes: {1}'.format(i, average_test_reward))
            state = next_state
            iter_epi += 1

    def evaluate(self, env, num_episodes, iter):
        """Test your agent with a provided environment.

        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        monitor_dir = os.path.join(self.logdir, 'gym_monitor', str(iter))
        print("Monitored evaluation video saved at {0}".format(monitor_dir))
        if os.path.exists(monitor_dir):
            shutil.rmtree(monitor_dir)
        env = wrappers.Monitor(env, monitor_dir)
        total_reward = 0
        for i in range(num_episodes):
            state = env.reset()
            self.test_preprocessor.reset()
            is_terminal = 0
            while not is_terminal:
                processed_state = self.test_preprocessor.process_state_for_network(state)
                q_values = self.calc_q_values(np.expand_dims(processed_state/self.z, axis=0))
                # print('q_values: {0}'.format(q_values))
                action = self.select_action(q_values)
                # print('action: {0}'.format(action))
                next_state, reward, is_terminal, debug_info = env.step(action)
                total_reward += reward
                state = next_state
                # env.render()
            self.test_preprocessor.reset()
            print(total_reward)
        average_reward = total_reward / num_episodes
        self.sess.run(self.assign_reward_op, feed_dict={self.set_reward: average_reward})
        merged_r = self.sess.run(self.merged_r)
        self.file_writer.add_summary(merged_r, iter)
        env.close()
        return average_reward

    def compile_test_model(self, iter):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True

        # with tf.Session(config=config) as sess:

        self.sess = tf.Session(config=config)

        self.y_pred = self.model.outputs[0]
        self.input = self.model.inputs[0]

        self.init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        # self.test_reward = tf.placeholder(tf.float32, shape=(), name='test_reward')

        # init Q-network
        # check if checkpoint exists
        self.model_saver = tf.train.Saver(self.model.trainable_weights)
        self.sess.run(self.init_op)


        print("compile finished")

        save_dir = os.path.join(self.logdir, 'checkpoints', str(iter))
        self.model_saver.restore(self.sess, save_dir)
        print("load model finished")


    def test_trained_model(self, num_episodes, iter):
        """Test your agent with a provided environment.

        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        monitor_dir = os.path.join(self.logdir, 'gym_monitor', 'test_'+str(iter))
        print("Monitored evaluation video saved at {0}".format(monitor_dir))
        if os.path.exists(monitor_dir):
            shutil.rmtree(monitor_dir)
        env = gym.make(self.env_name)
        env = wrappers.Monitor(env, monitor_dir)
        total_reward = 0
        max_reward = 0
        min_reward = 10000000
        rewards = []
        for i in range(num_episodes):
            state = env.reset()
            self.test_preprocessor.reset()
            is_terminal = 0
            episode_reward = 0
            while not is_terminal:
                processed_state = self.test_preprocessor.process_state_for_network(state)
                q_values = self.calc_q_values(np.expand_dims(processed_state/self.z, axis=0))
                # print('q_values: {0}'.format(q_values))
                action = self.select_action(q_values)
                # print('action: {0}'.format(action))
                next_state, reward, is_terminal, debug_info = env.step(action)
                episode_reward += reward
                state = next_state
                # env.render()
            self.test_preprocessor.reset()
            print(episode_reward)
            rewards.append(episode_reward)
            total_reward += episode_reward
            max_reward = max(max_reward, episode_reward)
            min_reward = min(min_reward, episode_reward)

        env.close()
        average_reward = total_reward / num_episodes
        std_reward = 0
        for i in range(rewards):
            std_reward += (rewards[i]-average_reward)**2
        std_reward = math.sqrt(std_reward/num_episodes)

        return average_reward, max_reward, min_reward, std_reward

