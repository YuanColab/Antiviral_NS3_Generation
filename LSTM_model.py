import json
import os
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold

def _onehotencode(s, vocab=None):
    if vocab is None:
        vocab = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', ' ']

    to_one_hot = np.eye(len(vocab), dtype=int)
    vocab_dict = {char: code for char, code in zip(vocab, to_one_hot)}

    result = [vocab_dict[l] for l in s]
    result = np.array(result)
    return np.reshape(result, (1, result.shape[0], result.shape[1])), vocab_dict, vocab

def _sample_with_temp(preds, temp=1.25):
    log_preds = np.log(preds) / temp
    log_preds -= np.max(log_preds)  # Numerical stability
    stretched_probs = np.exp(log_preds) / np.sum(np.exp(log_preds))
    return np.random.choice(len(stretched_probs), p=stretched_probs)

def load_model_instance(filename):
    modfile = os.path.dirname(filename) + '/model.p'
    mod = pickle.load(open(modfile, 'rb'))
    hdf5_file = ''.join(modfile.split('.')[:-1]) + '.hdf5'
    mod.model = load_model(hdf5_file)
    return mod

def save_model_instance(mod):
    tmp = mod.model
    tmp.save(mod.checkpointdir + 'model.hdf5')
    mod.model = None
    pickle.dump(mod, open(mod.checkpointdir + 'model.p', 'wb'))
    mod.model = tmp

class Model:
    def __init__(self, n_vocab, outshape, session_name, cell="LSTM", n_units=256, batch=64, layers=2,
                 learning_rate=0.001,
                 dropoutfract=0.1, loss='categorical_crossentropy', l2_reg=None, ask=True, seed=42):
        random.seed(seed)
        self.seed = seed
        self.dropout = dropoutfract
        self.inshape = (None, n_vocab)
        self.outshape = outshape
        self.neurons = n_units
        self.layers = layers
        self.batchsize = batch
        self.learning_rate = learning_rate
        self.losses = []
        self.val_losses = []
        self.cell = cell
        self.losstype = loss
        self.session_name = session_name
        self.logdir = f'./{session_name}'
        self.l2 = l2_reg

        self._setup_session_dir(ask)
        self.checkpointdir = os.path.join(self.logdir, 'checkpoint/')
        os.makedirs(self.checkpointdir, exist_ok=True)

        _, self.to_one_hot, self.vocab = _onehotencode('A')
        self.model = self._build_model()

    def _setup_session_dir(self, ask):
        if ask and os.path.exists(self.logdir):
            decision = input('\nSession folder already exists!\nDo you want to overwrite the previous session? [y/n] ')
            if decision.lower() not in ['y', 'yes']:
                self.logdir = f'./{input("Enter new session name: ")}'
                os.makedirs(self.logdir, exist_ok=True)

    def _build_model(self):
        weight_init = RandomNormal(mean=0.0, stddev=0.05, seed=self.seed)
        optimizer = Adam(learning_rate=self.learning_rate)

        l2reg = l2(self.l2) if self.l2 else None

        model = Sequential()
        for l in range(self.layers):
            layer_params = {
                'units': self.neurons,
                'return_sequences': True,
                'kernel_initializer': weight_init,
                'kernel_regularizer': l2reg,
                'dropout': self.dropout * (l + 1),
                'recurrent_dropout': self.dropout * (l + 1)
            }
            if l == 0:
                layer_params['input_shape'] = self.inshape  # Ensure input_shape is only set for the first layer

            if self.cell == "GRU":
                model.add(GRU(name=f'GRU{l + 1}', **layer_params))
            else:
                model.add(LSTM(name=f'LSTM{l + 1}', **layer_params))
        model.add(Dense(self.outshape, activation='softmax', kernel_initializer=weight_init, kernel_regularizer=l2reg))
        model.compile(loss=self.losstype, optimizer=optimizer)

        with open(os.path.join(self.checkpointdir, "model.json"), 'w') as f:
            json.dump(model.to_json(), f)
        model.summary()

        return model

    def train(self, x, y, epochs=100, valsplit=0.2, sample=0, patience=5, print_interval=1):
        writer = tf.summary.create_file_writer(f'./logs/{self.session_name}')
        min_val_loss_epoch = None
        min_val_loss = float('inf')
        self.losses = []
        self.val_losses = []

        callbacks = self._create_callbacks(valsplit, patience)

        with writer.as_default():
            for e in range(epochs):
                train_history = self.model.fit(x, y, epochs=1, batch_size=self.batchsize,
                                               validation_split=valsplit,
                                               shuffle=False, callbacks=callbacks)
                self._log_metrics(writer, e, train_history, valsplit)
                if sample:
                    sampled = self.sample(sample)
                    for s in sampled:
                        print(s)
                if e % print_interval == 0:
                    print(f"Epoch {e + 1}")  # 打印出当前的epoch编号，从1开始
                val_loss = train_history.history['val_loss'][-1]
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    min_val_loss_epoch = e + 1  # 保存的epoch编号从1开始

        # 打印保存模型的轮数，验证损失和学习率
        print("Best model saved at epoch: ", min_val_loss_epoch)
        print("Validation loss: ", min_val_loss)
        print("Learning rate: ", self.model.optimizer.lr.numpy())

        writer.close()
        return min_val_loss_epoch, min_val_loss, train_history
    def _create_callbacks(self, valsplit, patience):
        def scheduler(epoch, learning_rate):
            if epoch % 20 == 0 and epoch != 0:
                return learning_rate * 0.8
            else:
                return learning_rate

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='min', restore_best_weights=True),
            LearningRateScheduler(scheduler)
        ]
        if valsplit > 0.:
            callbacks.append(ModelCheckpoint(filepath=os.path.join(self.checkpointdir, 'best_model_epochs.hdf5'),
                                             monitor='val_loss', save_best_only=True, verbose=0, mode='min'))
        return callbacks

    def _log_metrics(self, writer, epoch, train_history, valsplit):
        tf.summary.scalar('loss', train_history.history['loss'][-1], step=epoch)
        self.losses.append(train_history.history['loss'][-1])

        if valsplit > 0.:
            val_loss = train_history.history['val_loss'][-1]
            self.val_losses.append(val_loss)
            tf.summary.scalar('val_loss', val_loss, step=epoch)

    def cross_val(self, x, y, epochs=100, cv=5, plot=True):
        self.losses = []
        self.val_losses = []
        kf = KFold(n_splits=cv)

        for cntr, (train_idx, test_idx) in enumerate(kf.split(x), start=1):
            print(f"\nFold {cntr}")
            self._initialize_model(cntr)
            train_history = self.model.fit(x[train_idx], y[train_idx], epochs=epochs, batch_size=self.batchsize,
                                           validation_data=(x[test_idx], y[test_idx]))
            self.losses.append(train_history.history['loss'])
            self.val_losses.append(train_history.history['val_loss'])

        self._compute_cv_metrics(plot)

    def _initialize_model(self, seed):
        self.__init__(self.inshape[1], self.outshape, self.session_name, self.cell, self.neurons, self.batchsize,
                      self.layers, self.learning_rate, self.dropout, self.losstype, self.l2, ask=False, seed=seed)

    def _compute_cv_metrics(self, plot):
        self.cv_loss = np.mean(self.losses, axis=0)
        self.cv_loss_std = np.std(self.losses, axis=0)
        self.cv_val_loss = np.mean(self.val_losses, axis=0)
        self.cv_val_loss_std = np.std(self.val_losses, axis=0)

        if plot:
            self.plot_losses(cv=True)

        best_epoch = np.argmin(self.cv_val_loss)
        minloss = self.cv_val_loss[best_epoch]
        print(
            f"\n{len(self.losses)}-fold cross-validation result:\n\nBest epoch:\t{best_epoch}\nVal_loss:\t{minloss:.4f}")
        with open(os.path.join(self.logdir, f'{self.session_name}_best_epoch.txt'), 'w') as f:
            f.write(
                f"{len(self.losses)}-fold cross-validation result:\n\nBest epoch:\t{best_epoch}\nVal_loss:\t{minloss:.4f}")

    def plot_losses(self, show=False, cv=False):
        fig, ax = plt.subplots()
        ax.set_title('LSTM Categorical Crossentropy Loss Plot', fontweight='bold', fontsize=16)
        filename = os.path.join(self.logdir, f'{self.session_name}_{"cv_" if cv else ""}loss_plot.pdf')
        x = range(1, len(self.cv_loss) + 1) if cv else range(1, len(self.losses) + 1)

        if cv:
            ax.plot(x, self.cv_loss, '-', color='#FE4365', label='Training')
            ax.plot(x, self.cv_val_loss, '-', color='k', label='Validation')
            ax.fill_between(x, self.cv_loss + self.cv_loss_std, self.cv_loss - self.cv_loss_std,
                            facecolors='#FE4365', alpha=0.5)
            ax.fill_between(x, self.cv_val_loss + self.cv_val_loss_std, self.cv_val_loss - self.cv_val_loss_std,
                            facecolors='k', alpha=0.5)
            minloss = np.min(self.cv_val_loss)
            plt.text(x=0.5, y=0.5,
                     s=f'best epoch: {np.where(minloss == self.cv_val_loss)[0][0]}, val_loss: {minloss:.4f}',
                     transform=ax.transAxes)
        else:
            ax.plot(x, self.losses, '-', color='#FE4365', label='Training')
            if self.val_losses:
                ax.plot(x, self.val_losses, '-', color='k', label='Validation')

        ax.set_xlim([0.5, len(x) + 0.5])
        ax.set_ylabel('Loss', fontweight='bold', fontsize=14)
        ax.set_xlabel('Epoch', fontweight='bold', fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.legend(loc='best')
        if show:
            plt.show()
        else:
            plt.savefig(filename)

    def sample(self, num=100, minlen=4, maxlen=42, start=None, temp=1.25, show=True):
        print("\nSampling...\n")
        sampled = []
        lcntr = 0
        if maxlen == 0:
            maxlen = np.random.randint(4, 42)
        if minlen >= maxlen:
            raise ValueError(f"minlen ({minlen}) must be less than maxlen ({maxlen})")

        for rs in range(num):
            random.seed(rs)
            longest = np.random.randint(minlen, maxlen) if maxlen == 0 else maxlen
            start_aa = start if start else 'B'
            sequence = start_aa

            while sequence[-1] != ' ' and len(sequence) <= longest:
                x, _, _ = _onehotencode(sequence)
                preds = self.model.predict(x)[0][-1]
                next_aa = _sample_with_temp(preds, temp=temp)
                sequence += self.vocab[next_aa]

            sequence = sequence[1:].rstrip() if start_aa == 'B' else sequence.rstrip()
            if len(sequence) < minlen:
                lcntr += 1
                continue

            sampled.append(sequence)
            if show:
                print(sequence)

        print(f"\t{lcntr} sequences were shorter than {minlen}")
        return sampled

        ax.yaxis.set_ticks_position('left')
        plt.legend(loc='best')
        if show:
            plt.show()
        else:
            plt.savefig(filename)

    def load_model(self, filename):
        self.model.load_weights(filename)
