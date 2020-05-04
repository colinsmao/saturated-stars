import os
import sys
import argparse
import pandas as pd
import numpy as np
import h5py

import torch
import torch.utils.data

import models


class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--num_epochs', type=int, default=8,
                                 help='# of training epochs')
        self.parser.add_argument('--batch_size', type=int, default=8,
                                 help='batch size during training')
        self.parser.add_argument('--test_batch_size', type=int, default=128,
                                 help='batch size for testing, default to batch_size')
        self.parser.add_argument('--num_workers', type=int, default=0,
                                 help='number of workers to use in dataloader')
        self.parser.add_argument('--train_path', type=str,
                                 default='./data/VISTA_cutouts/HDF5_F4_s64_k4-10_unique_norm6_train.h5',
                                 help="string path to HDF5 file containing training cutouts")
        self.parser.add_argument('--test_path', type=str,
                                 default='./data/VISTA_cutouts/HDF5_F4_s64_k4-10_unique_norm6_test.h5',
                                 help="string path to HDF5 file containing testing cutouts")
        self.parser.add_argument('--dataset_noise', type=float, default=0.0001,
                                 help='sigma of gaussian noise to add to dataset (training)')
        self.parser.add_argument('--draw_dist', type=str,
                                 default='./labels/HDF5_F4_s64_k4-10_unique_norm6.pkl',
                                 help="path to pandas DataFrame containing normalized metadata, for drawing samples")
        self.parser.add_argument('--limit_len_train', type=int, default=None,
                                 help="limit size of training dataset - set to None for the full dataset")
        self.parser.add_argument('--limit_len_test', type=int, default=None,
                                 help="limit size of the testing dataset - set to None for the full dataset")
        self.parser.add_argument('--save_dir', type=str, default='./runs/',
                                 help="directory to save model checkpoints")
        self.parser.add_argument('--save_freq', type=int, default=10000,
                                 help="number of iters between saving checkpoints")
        self.parser.add_argument('--save_num', type=int, default=3,
                                 help="number of checkpoints to keep - if 0, keep all")
        self.parser.add_argument('--test_freq', type=int, default=2500,
                                 help="number of iters between between evaluation model on testing set")
        self.parser.add_argument('--print_freq', type=int, default=1250,
                                 help="number of iters between between printing state to terminal")
        self.parser.add_argument('--image_freq', type=int, default=2500,
                                 help="number of iters between saving generated images")
        self.parser.add_argument('--load_name', type=int, default=None,
                                 help="unique name of model to resume training from")
        self.parser.add_argument('--load_iter', type=int, default=None,
                                 help="iteration to resume training from")

    def get_default_opts(self):
        return self.parser.parse_args([])

    def get_opts(self, opts):
        return self.parser.parse_known_args(opts)[0]


class StarsCutoutDataset(torch.utils.data.Dataset):
    """Characterizes a PyTorch dataset"""

    def __init__(self, HDF5_path, cutout_size, size, nz, dataset_noise, limit_len=None):
        """Initialize the dataset.
        Args:
            * HDF5_path (str)       -- string path to HDF5 file containing cutouts
            * cutout_size (int)     -- cutout size in the HDF5 file
            * size (int)            -- image input size for model
            * nz (int)              -- length of noise vector
            * dataset_noise (float) -- sigma of gaussian noise to add to dataset
            * limit_len (int)       -- limit size of dataset, None for entire dataset
        """

        self.HDF5_path = HDF5_path
        self.data = h5py.File(self.HDF5_path, 'r')

        self.cutout_size = cutout_size
        self.size = size
        self.nz = nz

        self.dataset_noise = dataset_noise

        if limit_len is None:
            self.len = len(self.data)
        else:
            self.len = min(limit_len, len(self.data))

    def close(self):
        self.data.close()

    def reopen(self):
        self.data = h5py.File(self.HDF5_path, 'r')

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """Generates one set of data from an image"""
        group = self.data[str(index)]

        # Raw cutout and label from hdf5 file
        cutout = np.array(group['data'])[None, :, :]
        label = group.attrs['metadata']

        if self.size < self.cutout_size:
            # Centered star
            x = (self.cutout_size - self.size) // 2
            cutout = cutout[:, x:(x + self.size), x:(x + self.size)]

        # Add some Gaussian noise to the cutout and labels
        if self.dataset_noise > 0:
            cutout = cutout + np.random.randn(*cutout.shape) * self.dataset_noise
            cutout = np.clip(cutout, -1., 1.)

            label_noise = np.random.randn(*label.shape) * self.dataset_noise
            label_noise[-1] = 0  # no noise to chipno
            label = label + label_noise
            label = np.clip(label, -1., 1.)

        # Noise
        noise = np.random.rand(self.nz) * 2 - 1

        return cutout.astype(np.float32), label.astype(np.float32), noise.astype(np.float32)


def make_fixed_label_noise(numeric_features, nz, batch_size=64, draw_dist=None):
    """Generate batch_size fixed 2D inputs to generator - not necessarily opt.batch_size"""
    if draw_dist is None:
        labels = torch.rand(batch_size, numeric_features) * 2 - 1
    else:
        from scipy import stats
        metadata = pd.read_pickle(draw_dist)
        assert numeric_features == len(metadata.columns)
        labels = torch.empty(batch_size, numeric_features)
        for col in range(numeric_features):
            col_name = metadata.columns[col]

            if col_name == 'k_m':
                lin = 8
                thresh = 0.5
                tot = 64
                power = 4
                k_ms = np.concatenate((np.linspace(0, thresh - (thresh / (lin + 1)), lin),
                                       np.power(np.linspace(np.power(thresh, power), 1, tot - lin), 1 / power)))
                labels[:, col] = torch.from_numpy(k_ms) * 2 - 1
            else:
                kernel = stats.gaussian_kde(metadata[col_name])
                labels[:, col] = torch.from_numpy(kernel.resample(batch_size)[0]).clamp(-1, 1)

    noise = torch.rand(batch_size, nz) * 2 - 1
    return labels, noise


def save_model_checkpoint(model, topt):
    """Save a model checkpoint.
    Args:
        * model: model to save. Must have .save(PATH) function
        * topt: TrainOptions() object, containing:
            * save_dir: parent directory within which to save the models
            * save_num: number of checkpoints to keep
    """
    # Generate unique path from model name and iters
    SAVE_DIR = os.path.join(topt.save_dir, model.unique_name)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    SAVE_PATH = os.path.join(SAVE_DIR, 'model_{}.pt'.format(model.iters))

    # Save model
    if os.path.exists(SAVE_PATH):
        print("Warning: checkpoint at '{}' already exists, overwriting.".format(SAVE_PATH))
        os.remove(SAVE_PATH)
    model.save(SAVE_PATH)

    # Delete excess checkpoints
    if topt.save_num == 0:
        pass
    else:
        # Get files in directory
        files = os.listdir(SAVE_DIR)
        # Filter files to only show checkpoints
        files = [f for f in files if f[:6] == 'model_']
        files = [f for f in files if f[-3:] == '.pt']
        # If number of checkpoints > save_num, delete the oldest checkpoints
        if len(files) > topt.save_num:
            iters = [int(f[6:-3]) for f in files]  # Get iters from filenames
            low_iters = sorted(iters)[:-topt.save_num]  # Get lowest iters
            for i in low_iters:
                os.remove(os.path.join(SAVE_DIR, 'model_{}.pt'.format(i)))


def get_opts(argv):
    model_opt = models.ModelOptions().get_opts(argv)
    train_opt = TrainOptions().get_opts(argv)

    # Set D_levels and G_levels based on opt.size
    d_ = np.round(np.log2(model_opt.size) - 1).astype(np.int32)
    g_ = d_ + 1
    model_opt.D_levels = d_
    model_opt.G_levels = g_

    # Mode 1 requires 2^N-1 size, other modes require 2^N size
    if model_opt.sampling_mode in [0]:
        assert np.equal(np.mod(np.log2(model_opt.size + 1), 1), 0)
    elif model_opt.sampling_mode in [1, 2]:
        assert np.equal(np.mod(np.log2(model_opt.size), 1), 0)

    print('Options:')
    print(model_opt)
    print(train_opt)

    return model_opt, train_opt


def get_dataloaders(model_opt, train_opt):
    print('Initializing datasets and dataloaders...', end=' ', flush=True)

    # Training data
    train_dataset = StarsCutoutDataset(HDF5_path=train_opt.train_path,
                                       cutout_size=64,
                                       size=model_opt.size,
                                       nz=model_opt.nz,
                                       dataset_noise=train_opt.dataset_noise,
                                       limit_len=train_opt.limit_len_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=train_opt.batch_size,
                                                   shuffle=True,
                                                   num_workers=train_opt.num_workers)

    # Testing data
    test_dataset = StarsCutoutDataset(HDF5_path=train_opt.test_path,
                                      cutout_size=64,
                                      size=model_opt.size,
                                      nz=model_opt.nz,
                                      dataset_noise=0.,
                                      limit_len=train_opt.limit_len_test)
    test_batch_size = train_opt.test_batch_size if train_opt.test_batch_size is not None else train_opt.batch_size
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=test_batch_size,
                                                  shuffle=True,
                                                  num_workers=train_opt.num_workers)
    print('Done.', flush=True)
    print('Training set: {} cutouts, batch size: {}'.format(len(train_dataset), train_opt.batch_size), flush=True)
    print('Testing set: {} cutouts, batch size: {}'.format(len(test_dataset), test_batch_size), flush=True)

    return train_dataloader, test_dataloader


def get_model(model_opt, save_dir, fixed_label_noise, n=1):
    print('Creating model...', end=' ', flush=True)
    # model = models.ConditionalModel(opt, dataset.fixed_label_noise)
    model = models.AuxiliaryModel(opt=model_opt,
                                  num_pred=n,
                                  fixed_label_noise=fixed_label_noise,
                                  save_dir=save_dir,
                                  training=True)
    print('Done.', flush=True)
    return model


def train(model, train_dataloader, test_dataloader, train_opt):
    extra_print = [1, 2, 5, 10, 20, 50, 100, 200, 400, 800]
    extra_image = [1, 10, 50, 100, 200, 400, 800]
    extra_test = [1, 10, 50, 100, 200, 400, 800]

    # Training Loop
    if model.iters == 0:
        print("Starting Training Loop...", flush=True)
    else:
        print("Continuing Training Loop...", flush=True)
    model.set_start_time()
    # For each epoch
    for epoch in range(train_opt.num_epochs):
        # For each batch in the dataloader
        for data in iter(train_dataloader):
            # Train on the batch
            model.set_input(data)
            # model.optimize_G_only(1)
            model.optimize_parameters()

            # Save losses for plotting later
            model.save_losses()
            # model.save_grads()

            # Output training stats
            if model.iters in extra_print or model.iters % train_opt.print_freq == 0:
                print(model.get_formatted_time(epoch, train_opt.num_epochs),
                      model.get_formatted_state(),
                      flush=True)

            # Check how the generator is doing by saving G's output on fixed_in
            if model.iters in extra_image or model.iters % train_opt.image_freq == 0:
                model.save_images()

            # Save checkpoint of model
            if model.iters % train_opt.save_freq == 0:
                save_model_checkpoint(model, train_opt)

            # Evaluate model with testing set
            if model.iters in extra_test or model.iters % train_opt.test_freq == 0:
                print(model.get_formatted_time(epoch, train_opt.num_epochs),
                      model.evaluate_for_dataloader(test_dataloader),
                      flush=True)

    # Finally
    if model.iters % train_opt.print_freq != 0:
        print(model.get_formatted_time(train_opt.num_epochs-1, train_opt.num_epochs),
              model.get_formatted_state(),
              flush=True)
    if model.iters % train_opt.image_freq != 0:
        model.save_images()
    if model.iters % train_opt.save_freq != 0:
        save_model_checkpoint(model, train_opt)
    if model.iters % train_opt.test_freq != 0:
        print(model.get_formatted_time(train_opt.num_epochs-1, train_opt.num_epochs),
              model.evaluate_for_dataloader(test_dataloader),
              flush=True)

    print('Done.')


if __name__ == '__main__':
    model_opt_, train_opt_ = get_opts(sys.argv)
    train_dataloader_, test_dataloader_ = get_dataloaders(model_opt_, train_opt_)
    fixed_label_noise_ = make_fixed_label_noise(model_opt_.numeric_features, model_opt_.nz, 64, train_opt_.draw_dist)
    model_ = get_model(model_opt_, train_opt_.save_dir, fixed_label_noise_)
    if train_opt_.load_name is not None:
        model_.load(os.path.join(train_opt_.save_dir, str(train_opt_.load_name), 'model_{}.pt'.format(train_opt_.load_iter)))
    train(model_, train_dataloader_, test_dataloader_, train_opt_)
