import tensorflow.keras.layers as kl
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.backend import one_hot
from tensorflow.python.keras.layers import Concatenate, ZeroPadding1D
import tensorflow.keras.backend as kb
from util import DLogger
import pandas as pd
from tensorflow.python.keras.models import load_model

"""
This class implements an RNN learner which can be trained on a task to predict next actions.

reset_after is to GRU use the implementation compatible with Tensorflow JS
"""
class RNNAgent:
    def __init__(self, n_actions, state_size, n_cells, reset_after=True, model_path=None):
        self.n_actions = n_actions
        self.n_cells = n_cells

        if model_path is None:
            # defining model through functional API
            original_inputs = tf.keras.Input(shape=(None, 1 + n_actions + state_size), name='action_reward')
            initial_state = tf.keras.Input(shape=(n_cells,), name='initial_state')
            rnn_out = kl.GRU(n_cells, return_sequences=True, name='GRU', reset_after=reset_after)(original_inputs, initial_state=initial_state)
            policy = kl.Dense(n_actions, activation='softmax', name='policy')(rnn_out)
            self.model = tf.keras.Model(inputs=[original_inputs, initial_state], outputs=[rnn_out, policy], name='encoder')
        else:
            self.model = load_model(model_path)
            DLogger.logger().debug('Model loaded from ' + model_path)
        DLogger.logger().debug("Model created with {} actions and {} cells".format(n_actions, n_cells))

    def train(self, reward, action, state,
              test_reward=None, test_action=None, test_state=None,
              output_path=None, lr=1e-3):
        action, inputs = self._make_model_input(action, reward, state)


        sh = action.shape
        DLogger.logger().debug('Training data dims: ' + str(sh))

        if action is not None:
            sh = test_action.shape
            DLogger.logger().debug('Test data dims: ' + str(sh))


        test_inputs = None
        if test_reward is not None:
            test_action, test_inputs = self._make_model_input(test_action, test_reward, test_state)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        save_path = output_path + 'model-init.h5'
        DLogger.logger().debug('Init model saved to: ' + save_path)
        self.model.save(save_path)

        events = []

        try:
            # Iterate over epochs.
            for epoch in range(50000):

                if test_reward is not None and epoch % 100 == 0:
                    _, test_pred_pol = self.model(
                        [test_inputs, np.zeros((test_inputs.shape[0], self.n_cells,), dtype=np.float32)])
                    test_actions_onehot_corrected = (1 - tf.reduce_sum(test_action, axis=2))[:, :, np.newaxis] + test_action
                    test_loss = tf.reduce_sum(kb.log(tf.reduce_sum(test_pred_pol * test_actions_onehot_corrected, axis=2)),
                                              axis=[1])
                    mean_test_loss = -kb.sum(test_loss) / tf.reduce_sum(test_action)

                    correct_percent = kb.sum(tf.cast(kb.sum(test_pred_pol * test_action, axis=2) > 1. / self.n_actions, tf.float32)) \
                                      / kb.sum(test_action)

                    DLogger.logger().debug('step %s: mean test loss = %s, %%correct = %s' %
                                           (epoch, mean_test_loss.numpy(), correct_percent.numpy()))

                    events.append({'epoch': epoch,
                                   'loss': mean_test_loss.numpy(),
                                   'n actions': tf.reduce_sum(test_action).numpy(),
                                   'sum loss': -kb.sum(test_loss).numpy(),
                                   '% correct': correct_percent.numpy()
                                   })

                    pd.DataFrame(events).to_csv(output_path + "events.csv")
                    save_path = output_path + 'model-' + str(epoch) + '.h5'
                    DLogger.logger().debug('Trained model saved to: ' + save_path)
                    self.model.save(save_path)

                with tf.GradientTape() as tape:
                    _, pred_pol = self.model([inputs, np.zeros((inputs.shape[0], self.n_cells,), dtype=np.float32)])

                    # TODO check this

                    # this is correction for the missing actions for which the one hot is [0,0, ... 0]
                    actions_onehot_corrected = (1 - tf.reduce_sum(action, axis=2))[:, :, np.newaxis] + action
                    loss = tf.reduce_sum(kb.log(tf.reduce_sum(pred_pol * actions_onehot_corrected, axis=2)), axis=[1])
                    loss = -kb.sum(loss)

                    correct_percent = kb.sum(tf.cast(kb.sum(pred_pol * action, axis=2) > 1. / self.n_actions, tf.float32)) \
                                      / kb.sum(action)

                grads = tape.gradient(loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                DLogger.logger().debug('step %s: mean loss = %.3f correct = %.3f'
                                       % (epoch, loss.numpy(), correct_percent.numpy()))
        except KeyboardInterrupt:
            DLogger.logger().debug('Training interrupted at trial ' + str(epoch))
            save_path = output_path + 'model-final.h5'
            DLogger.logger().debug('Trained model saved to: ' + save_path)
            self.model.save(save_path)



    def _make_model_input(self, action, reward, state):
        action, inputs = self.make_model_input(action, reward, state, self.n_actions)
        return action, inputs

    @classmethod
    def make_model_input(cls, action, reward, state, n_actions):
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.int32)
        state = tf.convert_to_tensor(state, dtype=tf.float32) if state is not None else None
        # changing action to one-hot encoding
        action = one_hot(action, n_actions)
        action_reward = Concatenate(axis=2)([reward[:, :, np.newaxis], action])
        # added dummy zero to the beginning
        action_reward = ZeroPadding1D(padding=[1, 0])(action_reward)
        if state is not None:
            inputs = Concatenate(axis=2)([action_reward[:, :-1, :, ], state])
        else:
            inputs = action_reward[:, :-1, :]
        return action, inputs

