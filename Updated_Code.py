#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1. Import packages

import pandas as pd
import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np
import random

import warnings
warnings.filterwarnings('ignore')

import datetime
import shutil
import csv


# In[2]:


#2. Technicalities

FLAGS = tf.compat.v1.app.flags.FLAGS
tf.compat.v1.app.flags.DEFINE_string('f', '', 'kernel')


# In[3]:


#3. Source: main.py

"""Set up dataset."""
tf.compat.v1.app.flags.DEFINE_string(
    'dataset', 'poly2d', 'dataset/scenario: sin, step, abs, linear, poly2d, poly3d, mnistz, mnistc, mnistzc')
tf.compat.v1.app.flags.DEFINE_integer('num_train', 500, 'training data')
tf.compat.v1.app.flags.DEFINE_integer('num_valid', 500, 'validation data')
tf.compat.v1.app.flags.DEFINE_integer('num_test', 500, 'test data')

"""Set up training."""
tf.compat.v1.app.flags.DEFINE_integer(
    'rep_net_layer', 2, 'layers of representation networks')
tf.compat.v1.app.flags.DEFINE_integer(
    'x_net_layer', 2, 'layers of treatment prediction network')
tf.compat.v1.app.flags.DEFINE_integer(
    'emb_net_layer', 2, 'layers of embedding network')
tf.compat.v1.app.flags.DEFINE_integer(
    'y_net_layer', 2, 'layers of outcome prediction network')
tf.compat.v1.app.flags.DEFINE_integer('emb_dim', 4, 'embedding dimension')
tf.compat.v1.app.flags.DEFINE_integer('rep_dim', 4, 'representation dimension')
tf.compat.v1.app.flags.DEFINE_float(
    'lrate', 1e-3, 'learning rate of optimizer')
tf.compat.v1.app.flags.DEFINE_float('dropout', 0., 'drop rate of dropout')
tf.compat.v1.app.flags.DEFINE_integer(
    'epochs', 1000, 'training epochs of AutoIV')
tf.compat.v1.app.flags.DEFINE_integer(
    'opt_lld_step', 1, 'steps of likelihood optimizer')
tf.compat.v1.app.flags.DEFINE_integer(
    'opt_bound_step', 1, 'steps of bound optimizer')
tf.compat.v1.app.flags.DEFINE_integer(
    'opt_2stage_step', 1, 'steps of two-stage optimizer')
tf.compat.v1.app.flags.DEFINE_float(
    'sigma', 0.1, 'hyper-parameter sigma of RBF kernel')
tf.compat.v1.app.flags.DEFINE_integer(
    'interval', 2, 'print times during training')
tf.compat.v1.app.flags.DEFINE_integer('exp_num', 20, 'experiment runs')

""" Set up experiments. """
tf.compat.v1.app.flags.DEFINE_boolean(
    'gen_new', True, 'whether generate new data') #Change: From False to True
tf.compat.v1.app.flags.DEFINE_boolean(
    'del_res', True, 'whether delete all the previous results') #Change: From False to True
tf.compat.v1.app.flags.DEFINE_string(
    'res_path', 'AutoIV-results/', 'result path')
tf.compat.v1.app.flags.DEFINE_string(
    'res_file', 'summary.csv', 'result summary csv file')
tf.compat.v1.app.flags.DEFINE_string('gpu', '0', 'which GPU to use')
tf.compat.v1.app.flags.DEFINE_integer('seed', 0, 'seed')

#Change: os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#Change: seed = FLAGS.seed
seed = 0
random.seed(seed)
np.random.seed(seed)
#Change: tf.set_random_seed(seed) TF update
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


# In[4]:


#4. Source: main_run

def main_run(FLAGS):
    """Set hyper-parameters."""
    coefs = {'coef_cx2y': 1, 'coef_zc2x': 1, 'coef_lld_zx': 1,
             'coef_lld_zy': 1, 'coef_lld_cx': 1,  'coef_lld_cy': 1,
             'coef_lld_zc': 1, 'coef_bound_zx': 1, 'coef_bound_zy': 1,
             'coef_bound_cx': 1, 'coef_bound_cy': 1, 'coef_bound_zc': 1, 'coef_reg': 0.001}

    num = {'train': FLAGS.num_train,
           'valid': FLAGS.num_valid, 'test': FLAGS.num_test}

    layers = {'rep_net_layer': FLAGS.rep_net_layer, 'x_net_layer': FLAGS.x_net_layer,
              'emb_net_layer': FLAGS.emb_net_layer, 'y_net_layer': FLAGS.y_net_layer}

    opt_steps = {'opt_lld_step': FLAGS.opt_lld_step, 'opt_bound_step': FLAGS.opt_bound_step,
                 'opt_2stage_step': FLAGS.opt_2stage_step}

    data_path = 'data/{}/{}-train_{}/'.format(
        FLAGS.dataset, FLAGS.dataset, FLAGS.num_train)

    all_paras = {'dataset': FLAGS.dataset, 'data_path': data_path, 'coefs': coefs,
                 'rep_dim': FLAGS.rep_dim, 'lrate': FLAGS.lrate, 'dropout': FLAGS.dropout,
                 'emb_dim': FLAGS.emb_dim, 'epochs': FLAGS.epochs, 'interval': FLAGS.interval,
                 'exp_num': FLAGS.exp_num, 'gen_new': FLAGS.gen_new, 'del_res': FLAGS.del_res,
                 'sigma': FLAGS.sigma, 'res_path': FLAGS.res_path, 'visible_gpu': FLAGS.gpu}

    all_paras.update(num)
    all_paras.update(layers)
    all_paras.update(opt_steps)

    print('\n\n' + '=' * 50)
    print('Run experiment.\n')
    _, _ = run(all_paras)


# In[5]:


def run(all_paras):
    """Run AutoIV."""

    """Create result files."""
    log = Log(all_paras)

    """Get data."""
    data = gen_load_data(all_paras, get_data=True)

    """Get model."""
    tf.compat.v1.reset_default_graph()
    dim_x, dim_v, dim_y = data[0]['x'].shape[1], data[0]['v'].shape[1], data[0]['y'].shape[1]
    model = AutoIV(all_paras, dim_x, dim_v, dim_y)

    """Get trainable variables."""
    zx_vars = get_tf_var(['zx'])
    zy_vars = get_tf_var(['zy'])
    cx_vars = get_tf_var(['cx'])
    cy_vars = get_tf_var(['cy'])
    zc_vars = get_tf_var(['zc'])
    rep_vars = get_tf_var(['rep/rep_z', 'rep/rep_c'])
    x_vars = get_tf_var(['x'])
    emb_vars = get_tf_var(['emb'])
    y_vars = get_tf_var(['y'])

    vars_lld = zx_vars + zy_vars + cx_vars + cy_vars + zc_vars
    vars_bound = rep_vars
    vars_2stage = rep_vars + x_vars + emb_vars + y_vars

    """Set optimizer."""
    train_opt_lld = get_opt(lrate=all_paras['lrate'], NUM_ITER_PER_DECAY=100,
                            lrate_decay=0.95, loss=model.loss_lld, _vars=vars_lld)

    train_opt_bound = get_opt(lrate=all_paras['lrate'], NUM_ITER_PER_DECAY=100,
                              lrate_decay=0.95, loss=model.loss_bound, _vars=vars_bound)

    train_opt_2stage = get_opt(lrate=all_paras['lrate'], NUM_ITER_PER_DECAY=100,
                               lrate_decay=0.95, loss=model.loss_2stage, _vars=vars_2stage)

    train_opts = [train_opt_lld, train_opt_bound, train_opt_2stage]
    train_steps = [all_paras['opt_lld_step'],
                   all_paras['opt_bound_step'], all_paras['opt_2stage_step']]

    ''' Run experiments '''
    train_sess = TrainSess(
        model, train_opts, train_steps, data, all_paras, log)
    result, _ = train_sess.train()

    return result, log


# In[6]:


class Log(object):
    def __init__(self, all_paras):
        """Result log."""

        self.coefs = all_paras['coefs']
        self.date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        self.res_path = all_paras['res_path']
        self.res_detail_path = self.res_path +             '{}-train_{}-'.format(all_paras['dataset'],
                                  all_paras['train']) + self.date + '/'

        self.res_summary_file = self.res_path + 'result-sum.txt'
        self.res_detail_file = self.res_detail_path + 'result-det.txt'

        self.all_paras = all_paras

        """Delete and create directory."""
        self.del_cre_directory()

    def del_cre_directory(self):
        """Delete previous result and create directory."""

        """If remove all the detail results."""
        if self.all_paras['del_res'] and os.path.exists(self.res_path):
            shutil.rmtree(self.res_path)

        """Remove summary results."""
        if os.path.exists(self.res_summary_file):
            os.remove(self.res_summary_file)

        """Create directory."""
        create_path(self.res_detail_path)

        self.write('both', self.date)

    def write(self, _file, _str, _print_flag=False):
        """Write str in summary file, or detail file, or both."""

        if _print_flag:
            print(_str)
        if _file == 'summary':
            with open(self.res_summary_file, 'a') as f:
                f.write(_str + '\n')
        elif _file == 'detail':
            with open(self.res_detail_file, 'a') as f:
                f.write(_str + '\n')
        elif _file == 'both':
            with open(self.res_summary_file, 'a') as f:
                f.write(_str + '\n')
            with open(self.res_detail_file, 'a') as f:
                f.write(_str + '\n')
        else:
            raise Exception('Wrong value for writing log file!')

    def get_path(self, _file):
        """Get path of summary path or detail path."""

        if _file == 'summary':
            return self.res_path
        elif _file == 'detail':
            return self.res_detail_path
        else:
            raise Exception('Wrong value for getting file path!')


# In[7]:


def create_path(path):
    """Create path like: 'a/b/c/'."""
    path_split = path.split('/')
    temp = path_split[0] + '/'
    for i in range(1, len(path_split)):
        if not os.path.exists(temp):
            os.mkdir(temp)
        temp = temp + path_split[i] + '/'


# In[8]:


def gen_load_data(all_paras, get_data=False, autoiv_gen=False):
    """Generate and load data."""

    data_path = all_paras['data_path']
    exp_num = all_paras['exp_num']
    da = all_paras['dataset']

    if (da == 'sin') or (da == 'step') or (da == 'abs') or (da == 'linear') or (da == 'poly2d') or (da == 'poly3d'):
        data = 'toy'
    else:
        data = 'mnist'

    """If gen_flag is True or data is not exist, generate data; else load data."""
    if all_paras['gen_new'] or (not os.path.exists(data_path)):
        if data == 'toy':
            data_or_path = toy(all_paras, get_data)
        else:
            data_or_path = mnist(all_paras, get_data)
    else:
        if autoiv_gen:
            variable = ['v', 'z', 'c', 'x', 'y', 'ye', 'v_c0',
                        'z_c0', 'c_c0', 'exp_num', 'train', 'valid', 'test']
        else:
            variable = ['v', 'z', 'c', 'x', 'y', 'ye',
                        'exp_num', 'train', 'valid', 'test']

        data_or_path = load_data(
            variable, variable, data_path, exp_num, get_data=get_data)

    print('\nUse data: {}\n'.format(data_path))

    return data_or_path


# In[9]:


def toy(all_paras, get_data):
    """Generate toy (low-dimensinal scenarios) data."""

    """Set paras."""
    num = all_paras['train'] + all_paras['valid'] + all_paras['test']
    exp_num = all_paras['exp_num']

    """Create fold for data."""
    create_path(all_paras['data_path'])

    # A folder for saving information.
    info_path = all_paras['data_path'] + 'info/'
    if not os.path.exists(info_path):
        os.mkdir(info_path)

    # A folder for saving csv and npz data.
    data_path = all_paras['data_path'] + 'data/'
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    datas = []
    print('=' * 50 + '\nStart to generate new data.')
    for exp_i in range(exp_num):
        """Generate Z ~ Uniform([-3, 3]^2)."""
        z = np.random.uniform(low=-3., high=3., size=(num, 2))

        """Generate e ~ N(0, 1)."""
        e = np.random.normal(0., 1., size=(num, 1))

        """Generate gamma ~ N(0, 0.1)."""
        gamma = np.random.normal(0., .1, size=(num, 2))

        """Generate x = z1 + e + gamma."""
        two_gps = False
        iv_strength = 0.9 #0.5
        if two_gps:
            x = 2 * z[:, 0:1] * (z[:, 0:1] > 0) * iv_strength + 2 * z[:, 1:2] * (z[:, 1] < 0) * iv_strength +                 2 * e * (1 - iv_strength) + gamma[:, 0:1]
        else:
            x = 2 * z[:, 0:1] * iv_strength + 2 *                 e * (1 - iv_strength) + gamma[:, 0:1]

        """Generate y = g0(x) + e + delta."""
        if all_paras['dataset'] == 'sin':
            g0 = np.sin(x)
        elif all_paras['dataset'] == 'step':
            g0 = - np.heaviside(x, 1.0)
        elif all_paras['dataset'] == 'abs':
            g0 = np.abs(x)
        elif all_paras['dataset'] == 'linear':
            g0 = -1.0 * x
        elif all_paras['dataset'] == 'poly2d':
            g0 = -0.4 * x - 0.1 * (x**2)
        elif all_paras['dataset'] == 'poly3d':
            g0 = -0.8 * x + 0.1 * (x**2) + 0.05 * (x ** 3)

        y = g0
        ye = g0 + gamma[:, 1:2] + 2. * e

        y = (y - np.mean(y)) / np.std(y)
        ye = (ye - np.mean(ye)) / np.std(ye)

        """Draw pictures."""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x_draw, y_draw = get_line(x, y)
        ax.plot(x_draw, y_draw, label='structural function',
                linewidth=5, c='royalblue', zorder=10)
        ax.scatter(x, ye, label='data', s=6, c='limegreen', zorder=1)
        plt.xlim(-4, 4)
        plt.ylim(-3, 3)
        plt.savefig(info_path + 'y_{}-train_{}-distribution{}.png'
                    .format(all_paras['dataset'], all_paras['train'], exp_i + 1))
        plt.close(fig)

        """Write data in csv."""
        # Write data distribution.
        with open(info_path + 'x_y_ye-train_{}-distribution_{}.txt'.format(all_paras['train'], exp_i + 1), 'w') as f:
            f.write('\nmean_std_x: {}, {}\n'.format(np.mean(x), np.std(x)))
            f.write('\nmedian_min_max_x: {}, {}, {}\n'.format(
                np.median(x), np.min(x), np.max(x)))
            f.write('\nmean_std_y: {}, {}\n'.format(np.mean(y), np.std(y)))
            f.write('\nmedian_min_max_y: {}, {}, {}\n'.format(
                np.median(y), np.min(y), np.max(y)))
            f.write('\nmean_std_ye: {}, {}'.format(np.mean(ye), np.std(ye)))
            f.write('\nmedian_min_max_ye: {}, {}, {}\n'.format(
                np.median(ye), np.min(ye), np.max(ye)))

        # Write data.
        with open(data_path + 'exp{}.csv'.format(exp_i), 'w', newline='') as f:
            f.write('x, y, ye\n')
            csv_writer = csv.writer(f, delimiter=',')
            for j in range(num):
                temp = [x[j][0], y[j][0], ye[j][0]]
                csv_writer.writerow(temp)

        v = np.concatenate([z, gamma], axis=1)

        """Write data in npz."""
        np.savez(data_path + 'exp{}.npz'.format(exp_i), v=v, z=z, c=gamma, x=x, y=y, ye=ye,
                 train=all_paras['train'], valid=all_paras['valid'], test=all_paras['test'], exp_num=exp_num)

        if get_data:
            datas = datas + [{'v': v, 'z': z, 'c': gamma, 'x': x, 'y': y, 'ye': ye, 'exp_num': exp_num,
                              'train': all_paras['train'], 'valid': all_paras['valid'], 'test': all_paras['test']}]

    print('Data size:\n\tz: {}\n\tc: {}\n\tx: {}\n\ty: {}'.format(
        z.shape, gamma.shape, x.shape, y.shape))
    print('Finish generating and saving data.\n' + '=' * 50)

    """Return data or path."""
    if get_data:
        return datas
    else:
        return all_paras['data_path']


# In[10]:


def get_line(x, y, x_int_times=50, x_min=-5, x_max=5):
    interval = (x_max - x_min) / x_int_times
    x_new, y_new = [], []
    for int_i in range(x_int_times):
        start = x_min + interval * int_i
        end = x_min + interval * (int_i + 1)
        get_data = np.where((x > start) & (x < end))
        x_new = x_new + [(start + end) / 2]
        y_new = y_new + [np.mean(y[get_data])]
    return np.array(x_new), np.array(y_new)


# In[11]:


class AutoIV(object):
    def __init__(self, all_paras, dim_x, dim_v, dim_y):
        """Build AutoIV model."""

        """Get sess and placeholder."""
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        self.dim_x, self.dim_v, self.dim_y = dim_x, dim_v, dim_y
        
        #Change: TF Update
        # Disable eager execution
        tf.compat.v1.disable_eager_execution()
        self.x = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.dim_x], name='x')
        self.v = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.dim_v], name='v')
        self.y = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.dim_y], name='y')

        """Set up parameters."""
        self.emb_dim = all_paras['emb_dim']
        self.rep_dim = all_paras['rep_dim']
        self.num = all_paras['train']
        self.coefs = all_paras['coefs']
        self.dropout = all_paras['dropout']
        self.train_flag = tf.compat.v1.placeholder(tf.bool, name='train_flag')
        self.all_paras = all_paras
        self.get_flag = True  # tf.Variable or tf.get_variable
        #self.init = tf.contrib.layers.xavier_initializer() TF Update
        self.init = tf.keras.initializers.GlorotUniform() 

        """Build model and get loss."""
        self.build_model()
        self.calculate_loss()

    def build_model(self):
        """Build model."""

        """Build representation network."""
        with tf.compat.v1.variable_scope('rep'):
            rep_net_layer = self.all_paras['rep_net_layer']
            self.rep_z, self.w_z, self.b_z = self.rep_net(inp=self.v,
                                                          dim_in=self.dim_v,
                                                          dim_out=self.rep_dim,
                                                          layer=rep_net_layer,
                                                          name='rep_z')
            self.rep_c, self.w_c, self.b_c = self.rep_net(inp=self.v,
                                                          dim_in=self.dim_v,
                                                          dim_out=self.rep_dim,
                                                          layer=rep_net_layer,
                                                          name='rep_c')
        self.rep_zc = tf.concat([self.rep_z, self.rep_c], 1)

        """Build treatment prediction network."""
        with tf.compat.v1.variable_scope('x'):
            self.x_pre, self.w_x, self.b_x = self.x_net(inp=self.rep_zc,
                                                        dim_in=self.rep_dim * 2,
                                                        dim_out=self.dim_x,
                                                        layer=self.all_paras['x_net_layer'])

        """Build embedding network."""
        with tf.compat.v1.variable_scope('emb'):
            self.x_emb, self.w_emb, self.b_emb = self.emb_net(inp=self.x_pre,
                                                              dim_in=self.dim_x,
                                                              dim_out=self.emb_dim,
                                                              layer=self.all_paras['emb_net_layer'])
        self.rep_cx = tf.concat([self.rep_c, self.x_emb], 1)

        """Build outcome prediction network."""
        with tf.compat.v1.variable_scope('y'):
            self.y_pre, self.w_y, self.b_y = self.y_net(inp=self.rep_cx,
                                                        dim_in=self.rep_dim + self.emb_dim,
                                                        dim_out=self.dim_y,
                                                        layer=self.all_paras['y_net_layer'])

        """Maximize MI between z and x."""
        with tf.compat.v1.variable_scope('zx'):
            self.lld_zx, self.bound_zx, self.mu_zx, self.logvar_zx, self.ws_zx = self.mi_net(
                inp=self.rep_z,
                outp=self.x,
                dim_in=self.rep_dim,
                dim_out=self.dim_x,
                mi_min_max='max')

        """Minimize MI between z and y given x."""
        with tf.compat.v1.variable_scope('zy'):
            self.lld_zy, self.bound_zy, self.mu_zy, self.logvar_zy, self.ws_zy = self.mi_net(
                inp=self.rep_z,
                outp=self.y,
                dim_in=self.rep_dim,
                dim_out=self.dim_y,
                mi_min_max='min',
                name='zy')

        """Maximize MI between c and x."""
        with tf.compat.v1.variable_scope('cx'):
            self.lld_cx, self.bound_cx, self.mu_cx, self.logvar_cx, self.ws_cx = self.mi_net(
                self.rep_c,
                outp=self.x,
                dim_in=self.rep_dim,
                dim_out=self.dim_x,
                mi_min_max='max')

        """Maximize MI between c and y."""
        with tf.compat.v1.variable_scope('cy'):
            self.lld_cy, self.bound_cy, self.mu_cy, self.logvar_cy, self.ws_cy = self.mi_net(
                inp=self.rep_c,
                outp=self.y,
                dim_in=self.rep_dim,
                dim_out=self.dim_y,
                mi_min_max='max')

        """Minimize MI between z and c."""
        with tf.compat.v1.variable_scope('zc'):
            self.lld_zc, self.bound_zc, self.mu_zc, self.logvar_zc, self.ws_zc = self.mi_net(
                inp=self.rep_z,
                outp=self.rep_c,
                dim_in=self.rep_dim,
                dim_out=self.rep_dim,
                mi_min_max='min')

    def calculate_loss(self):
        """Get loss."""

        """Loss of y prediction."""
        self.loss_cx2y = tf.reduce_mean(tf.square(self.y - self.y_pre))

        """Loss of t prediction."""
        self.loss_zc2x = tf.reduce_mean(tf.square(self.x - self.x_pre))

        """Loss of network regularization."""
        def w_reg(w):
            """Calculate l2 loss of network weight."""
            w_reg_sum = 0
            for w_i in range(len(w)):
                w_reg_sum = w_reg_sum + tf.nn.l2_loss(w[w_i])
            return w_reg_sum
        self.loss_reg = (w_reg(self.w_z) + w_reg(self.w_c) +
                         w_reg(self.w_emb) + w_reg(self.w_x) + w_reg(self.w_y)) / 5.

        """Losses."""
        self.loss_lld = self.coefs['coef_lld_zy'] * self.lld_zy +             self.coefs['coef_lld_cx'] * self.lld_cx +             self.coefs['coef_lld_zx'] * self.lld_zx +             self.coefs['coef_lld_cy'] * self.lld_cy +             self.coefs['coef_lld_zc'] * self.lld_zc

        self.loss_bound = self.coefs['coef_bound_zy'] * self.bound_zy +             self.coefs['coef_bound_cx'] * self.bound_cx +             self.coefs['coef_bound_zx'] * self.bound_zx +             self.coefs['coef_bound_cy'] * self.bound_cy +             self.coefs['coef_bound_zc'] * self.bound_zc +             self.coefs['coef_reg'] * self.loss_reg
        #Note: 

        self.loss_2stage = self.coefs['coef_cx2y'] * self.loss_cx2y +             self.coefs['coef_zc2x'] * self.loss_zc2x +             self.coefs['coef_reg'] * self.loss_reg

    def layer_out(self, inp, w, b, flag):
        """Set up activation function and dropout for layers."""
        out = tf.matmul(inp, w) + b
        if flag:
            #return tf.layers.dropout(tf.nn.elu(out), rate=self.dropout, training=self.train_flag) TF Update
            return tf.keras.layers.Dropout(rate=self.dropout)(tf.nn.elu(out), training=self.train_flag)
        else:
            return out

    def rep_net(self, inp, dim_in, dim_out, layer, name):
        """Representation network."""
        rep, w_, b_ = [inp], [], []
        with tf.compat.v1.variable_scope(name):
            for i in range(layer):
                dim_in_net = dim_in if (i == 0) else dim_out
                dim_out_net = dim_out
                w_.append(get_var(dim_in_net, dim_out_net, 'w_' +
                          name + '_%d' % i, get_flag=self.get_flag))
                b_.append(tf.Variable(
                    tf.zeros([1, dim_out_net]), name='b_' + name + '_%d' % i))
                rep.append(self.layer_out(
                    rep[i], w_[i], b_[i], flag=(i != layer - 1)))
        return rep[-1], w_, b_

    def x_net(self, inp, dim_in, dim_out, layer):
        """Treatment prediction network."""
        x_pre, w_x, b_x = [inp], [], []
        for i in range(layer):
            dim_in_net = dim_in if (i == 0) else dim_in // (i * 2)
            dim_out_net = dim_in // ((i + 1) *
                                     2) if i != (layer - 1) else dim_out
            dim_in_net = dim_in_net if dim_in_net > 0 else 1
            dim_out_net = dim_out_net if dim_out_net > 0 else 1
            w_x.append(get_var(dim_in_net, dim_out_net, 'w_x' +
                       '_%d' % i, get_flag=self.get_flag))
            b_x.append(tf.Variable(
                tf.zeros([1, dim_out_net]), name='b_x' + '_%d' % i))
            x_pre.append(self.layer_out(
                x_pre[i], w_x[i], b_x[i], flag=(i != layer - 1)))
        return x_pre[-1], w_x, b_x

    def emb_net(self, inp, dim_in, dim_out, layer):
        """Treatment embedding network."""
        x_emb, w_emb, b_emb = [inp], [], []
        for i in range(layer):
            dim_in_net = dim_in if (i == 0) else dim_out
            dim_out_net = dim_out
            w_emb.append(get_var(dim_in_net, dim_out_net,
                         'w_emb_%d' % i, get_flag=self.get_flag))
            b_emb.append(tf.Variable(
                tf.zeros([1, dim_out_net]), name='b_emb_%d' % i))
            x_emb.append(self.layer_out(
                x_emb[i], w_emb[i], b_emb[i], flag=(i != layer - 1)))
        return x_emb[-1], w_emb, b_emb

    def y_net(self, inp, dim_in, dim_out, layer):
        """Outcome prediction network."""
        y_pre, w_y, b_y = [inp], [], []
        for i in range(layer):
            dim_in_net = dim_in if (i == 0) else dim_in // (i * 2)
            dim_out_net = dim_in // ((i + 1) *
                                     2) if i != (layer - 1) else dim_out
            dim_in_net = dim_in_net if dim_in_net > 0 else 1
            dim_out_net = dim_out_net if dim_out_net > 0 else 1
            w_y.append(get_var(dim_in_net, dim_out_net, 'w_y' +
                       '_%d' % i, get_flag=self.get_flag))
            b_y.append(tf.Variable(
                tf.zeros([1, dim_out_net]), name='b_y' + '_%d' % i))
            y_pre.append(self.layer_out(
                y_pre[i], w_y[i], b_y[i], flag=(i != layer - 1)))
        return y_pre[-1], w_y, b_y

    def fc_net(self, inp, dim_out, act_fun, init):
        """Fully-connected network."""
        #return layers.fully_connected(inputs=inp,
        #                              num_outputs=dim_out,
        #                              activation_fn=act_fun,
        #                              weights_initializer=init)
        return tf.keras.layers.Dense(units=dim_out, activation=act_fun, kernel_initializer=init)(inp)

    def mi_net(self, inp, outp, dim_in, dim_out, mi_min_max, name=None):
        """Mutual information network."""
        h_mu = self.fc_net(inp, dim_in // 2, tf.nn.elu, self.init)
        mu = self.fc_net(h_mu, dim_out, None, self.init)
        h_var = self.fc_net(inp, dim_in // 2, tf.nn.elu, self.init)
        logvar = self.fc_net(h_var, dim_out, tf.nn.tanh, self.init)

        #new_order = tf.random_shuffle(tf.range(self.num)) TF Update
        new_order = tf.random.shuffle(tf.range(self.num))
        outp_rand = tf.gather(outp, new_order)

        """Get likelihood."""
        loglikeli = -             tf.reduce_mean(tf.reduce_sum(-(outp - mu) ** 2 /
                           tf.exp(logvar) - logvar, axis=-1))

        """Get positive and negative U."""
        pos = - (mu - outp) ** 2 / tf.exp(logvar)
        neg = - (mu - outp_rand) ** 2 / tf.exp(logvar)

        if name == 'zy':
            x_rand = tf.gather(self.x, new_order)

            # Using RBF kernel to measure distance.
            sigma = self.all_paras['sigma']
            w = tf.exp(-tf.square(self.x - x_rand) / (2 * sigma ** 2))
            w_soft = tf.nn.softmax(w, axis=0)
        else:
            w_soft = 1. / self.num

        """Get estimation of mutual information."""
        if mi_min_max == 'min':
            pn = 1.
        elif mi_min_max == 'max':
            pn = -1.
        else:
            raise ValueError
        bound = pn * tf.reduce_sum(w_soft * (pos - neg))

        return loglikeli, bound, mu, logvar, w_soft


# In[12]:


def get_var(_dim_in, _dim_out, _name, get_flag=False):
    if get_flag:
        #var = tf.get_variable(name=_name, shape=[_dim_in, _dim_out],
        #                      initializer=tf.contrib.layers.xavier_initializer())
        var = tf.Variable(initial_value=tf.keras.initializers.GlorotUniform()([_dim_in, _dim_out]),
                  name=_name)
    else:
        var = tf.Variable(tf.random.normal(
            [_dim_in, _dim_out], stddev=0.1 / np.sqrt(_dim_out)), name=_name)
    return var


# In[13]:


def get_tf_var(names):
    """ Get all trainable variables. """

    _vars = []
    for na_i in range(len(names)):
        _vars = _vars +             tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=names[na_i])
    return _vars


# In[14]:


def get_opt(lrate, NUM_ITER_PER_DECAY, lrate_decay, loss, _vars):
    global_step = tf.Variable(0, trainable=False)
    #lr = tf.train.exponential_decay(
    #    lrate, global_step, NUM_ITER_PER_DECAY, lrate_decay, staircase=True) TF Update
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lrate, NUM_ITER_PER_DECAY, lrate_decay, staircase=True)
    lr = lr_schedule(global_step)
    opt = tf.compat.v1.train.AdamOptimizer(lr)
    train_opt = opt.minimize(loss, global_step=global_step, var_list=_vars)
    return train_opt


# In[15]:


class TrainSess(object):
    def __init__(self, model, train_opts, train_steps, data, all_paras, log):
        """Train tensorflow model."""

        self.model, self.train_opts, self.train_steps = model, train_opts, train_steps
        self.data = data
        self.all_paras = all_paras
        self.log = log
        self.epochs, self.int = all_paras['epochs'], all_paras['epochs'] // all_paras['interval']
        self.train_mses, self.valid_mses, self.test_mses = [], [], []
        self.detail_file = self.log.get_path('detail')

    def train(self):
        exp_log(sta_end_flag='start', log=self.log, all_paras=self.all_paras)
        for exp in range(self.all_paras['exp_num']):
            self.model.sess.run(tf.compat.v1.global_variables_initializer())
            self.train_mse_int, self.valid_mse_int, self.test_mse_int = [], [], []

            """Load data."""
            data = self.data[exp]
            v, x, y, ye = data['v'], data['x'], data['y'], data['ye']
            data_split = [self.all_paras['train'],
                          self.all_paras['valid'], self.all_paras['test']]
            train_range = range(0, data_split[0])
            valid_range = range(data_split[0], data_split[0] + data_split[1])
            test_range = range(
                data_split[0] + data_split[1], data_split[0] + data_split[1] + data_split[2])

            """Training, validation, and test data."""
            v_train, x_train, ye_train = v[train_range,
                                           :], x[train_range, :], ye[train_range, :]
            v_valid, x_valid, ye_valid = v[valid_range,
                                           :], x[valid_range, :], ye[valid_range, :]
            v_test, x_test, y_test = v[test_range,
                                       :], x[test_range, :], y[test_range, :]

            """Training, validation, and test dict."""
            dict_train_true = {self.model.v: v_train, self.model.x: x_train, self.model.y: ye_train,
                               self.model.train_flag: True}
            dict_train = {self.model.v: v_train, self.model.x: x_train, self.model.x_pre: x_train,
                          self.model.y: ye_train, self.model.train_flag: False}
            dict_valid = {self.model.v: v_valid, self.model.x: x_valid, self.model.x_pre: x_valid,
                          self.model.y: ye_valid, self.model.train_flag: False}
            dict_test = {self.model.v: v_test, self.model.x_pre: x_test, self.model.y: y_test,
                         self.model.train_flag: False}

            """Train model."""
            self.log.write(
                'detail', '=' * 50 + '\nStart {}th experiment.'.format(exp + 1), _print_flag=True)
            for ep_th in range(self.epochs):
                if (ep_th % self.int == 0) or (ep_th == self.epochs - 1):
                    loss = self.model.sess.run([self.model.loss_cx2y,
                                                self.model.loss_zc2x,
                                                self.model.lld_zx,
                                                self.model.lld_zy,
                                                self.model.lld_cx,
                                                self.model.lld_cy,
                                                self.model.lld_zc,
                                                self.model.bound_zx,
                                                self.model.bound_zy,
                                                self.model.bound_cx,
                                                self.model.bound_cy,
                                                self.model.bound_zc,
                                                self.model.loss_reg],
                                               feed_dict=dict_train)

                    self.log.write('detail', 'Epoch {}th:'.format(
                        str(ep_th).zfill(4)), _print_flag=True)
                    coef_name = [key for key in self.all_paras['coefs']]
                    for i in range(len(loss)):
                        self.log.write('detail', '\tLoss_{}: %.6f'.format(
                            coef_name[i][5:]) % loss[i], _print_flag=True)

                    """Get train and valid mse."""
                    y_pre_train = self.model.sess.run(
                        self.model.y_pre, feed_dict=dict_train)
                    y_pre_valid = self.model.sess.run(
                        self.model.y_pre, feed_dict=dict_valid)
                    y_pre_test = self.model.sess.run(
                        self.model.y_pre, feed_dict=dict_test)

                    mse_train = np.mean(np.square(y_pre_train - ye_train))
                    mse_valid = np.mean(np.square(y_pre_valid - ye_valid))
                    mse_test = np.mean(np.square(y_pre_test - y_test))

                    """Save mse."""
                    self.log.write('detail', '-' * 50 + '\n\ttrain: %.4f | valid: %.4f | test: %.4f\n'
                                   % (float(mse_train), float(mse_valid), float(mse_test)), _print_flag=True)

                    self.train_mse_int = np.append(
                        self.train_mse_int, mse_train)
                    self.valid_mse_int = np.append(
                        self.valid_mse_int, mse_valid)
                    self.test_mse_int = np.append(self.test_mse_int, mse_test)

                for i in range(len(self.train_opts)):  # optimizer to train
                    for j in range(self.train_steps[i]):  # steps of optimizer
                        self.model.sess.run(
                            self.train_opts[i], feed_dict=dict_train_true)

            """Save final MSE results."""
            self.train_mses = np.append(self.train_mses, mse_train)
            self.valid_mses = np.append(self.valid_mses, mse_valid)
            self.test_mses = np.append(self.test_mses, mse_test)

            """Save variables after training."""
            z, c = data['z'], data['c']
            v_c0 = np.concatenate(
                [z, np.zeros((c.shape[0], c.shape[1]))], axis=1)
            dict_all = {self.model.v: v, self.model.x: x,
                        self.model.y: y, self.model.train_flag: False}
            dict_all_c0 = {self.model.v: v_c0, self.model.train_flag: False}
            res_val_save(self.model, self.all_paras, [
                         dict_all, dict_all_c0], exp)

        exp_log('end', self.log, self.all_paras)

        return [self.train_mses, self.valid_mses, self.test_mses], loss


# In[16]:


def exp_log(sta_end_flag, log, all_paras):
    time_now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if sta_end_flag == 'start':
        log.write('both', '=' * 50)
        log.write('both', 'Start training: ' + time_now)
        log.write('both', 'data_train: {}; data_valid: {}; data_test: {}'.format(
            all_paras['train'], all_paras['valid'], all_paras['test']))
        """ Log hyper-parameters. """
        log.write('both', '-' * 50)
        log.write('both', 'rep_net_layer: {}'.format(
            all_paras['rep_net_layer']))
        log.write('both', 'x_net_layer: {}'.format(all_paras['x_net_layer']))
        log.write('both', 'emb_net_layer: {}'.format(
            all_paras['emb_net_layer']))
        log.write('both', 'y_net_layer: {}'.format(all_paras['y_net_layer']))
        log.write('both', 'emb_dim: {}'.format(all_paras['emb_dim']))
        log.write('both', 'rep_dim: {}'.format(all_paras['rep_dim']))
        log.write('both', 'learning rate: {}'.format(all_paras['lrate']))
        log.write('both', 'dropout: {}'.format(all_paras['dropout']))
        log.write('both', 'epochs: {}'.format(all_paras['epochs']))
        log.write('both', 'opt_lld_step: {}'.format(all_paras['opt_lld_step']))
        log.write('both', 'opt_bound_step: {}'.format(
            all_paras['opt_bound_step']))
        log.write('both', 'opt_2stage_step: {}'.format(
            all_paras['opt_2stage_step']))
        log.write('both', 'sigma: {}'.format(all_paras['sigma']))
        for key in all_paras['coefs']:
            log.write('both', key + ': ' + str(all_paras['coefs'][key]))
        log.write('both', '-' * 50)

    elif sta_end_flag == 'end':
        log.write('both', 'Finish training: ' + time_now)
        log.write('both', '=' * 50)


# In[17]:


def res_val_save(model, all_para, dicts, exp):
    """Save variables after training."""

    """Get feed_dict."""
    dict_all, dict_all_c0 = dicts[0], dicts[1]

    z, c, x, y, x_pre, y_pre = model.sess.run(
        [model.rep_z, model.rep_c, model.x, model.y, model.x_pre, model.y_pre], feed_dict=dict_all)
    z_c0, c_c0 = model.sess.run(
        [model.rep_z, model.rep_c], feed_dict=dict_all_c0)

    """Data path."""
    path_split = all_para['data_path'].split('/')
    gen_dciv = 'autoiv-' + all_para['dataset']
    data_path = path_split[0] + '/' + path_split[1] + '/' + gen_dciv + '/' +         gen_dciv +         '-train_{}-rep_{}/data/'.format(all_para['train'], all_para['rep_dim'])
    create_path(data_path)

    """Save data in csv.""" #Here is where they export to CSV
    num = all_para['train'] + all_para['valid'] + all_para['test']
    with open(data_path + 'exp{}.csv'.format(exp), 'w', newline='') as f:
        f.write('x,x_pre, y, y_pre\n')
        csv_writer = csv.writer(f, delimiter=',')
        for j in range(num):
            temp = [x[j][0], x_pre[j][0], y[j][0], y_pre[j][0]]
            temp.extend(z[j, :]) #IV
            temp.extend(c[j, :])
            csv_writer.writerow(temp)

    """Save data in npz."""
    v = np.concatenate([z, c], axis=1)
    v_c0 = np.concatenate([z_c0, c_c0], axis=1)
    np.savez(data_path + 'exp{}.npz'.format(exp),
             v=v, z=z, c=c, x=x, y=y, ye=y, v_c0=v_c0, z_c0=z_c0, c_c0=c_c0,
             train=all_para['train'], valid=all_para['valid'], test=all_para['test'], exp_num=all_para['exp_num'])

    print(data_path + 'exp{}.npz'.format(exp))
    


# In[18]:


main_run(FLAGS)


# In[ ]:





# In[ ]:





# In[ ]:




