from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.vrp.state_cvrp import StateCVRP
from problems.vrp.state_sdvrp import StateSDVRP
from utils.beam_search import beam_search
import math
from numpy.random import default_rng
from numpy import meshgrid, array, random
# from sklearn.datasets.samples_generator import make_blobs
import scipy.stats
import numpy as np
from utils.data_utils import check_extension, save_dataset



class CVRP(object):

    NAME = 'cvrp'  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -CVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        for i in range(pi.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= CVRP.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = CVRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class SDVRP(object):

    NAME = 'sdvrp'  # Split Delivery Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()

        # Each node can be visited multiple times, but we always deliver as much demand as possible
        # We check that at the end all demand has been satisfied
        demands = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -SDVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        rng = torch.arange(batch_size, out=demands.data.new().long())
        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        a_prev = None
        for a in pi.transpose(0, 1):
            assert a_prev is None or (demands[((a_prev == 0) & (a == 0)), :] == 0).all(), \
                "Cannot visit depot twice if any nonzero demand"
            d = torch.min(demands[rng, a], SDVRP.VEHICLE_CAPACITY - used_cap)
            demands[rng, a] -= d
            used_cap += d
            used_cap[a == 0] = 0
            a_prev = a
        assert (demands == 0).all(), "All demand must be satisfied"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateSDVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"
        assert not compress_mask, "SDVRP does not support compression of the mask"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = SDVRP.make_state(input)

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size
    }


class VRPDataset(Dataset):
    def process_cvrp_lib(self, file_path):
        torch_array = []

        if(os.path.isdir(file_path)):
            files = os.listdir(file_path)
            for file in files:
                file_open = open(file_path+'/'+file, 'rb')
                # file_open = open(file, 'rb')
                pickle_file = pickle.load(file_open)
                depot = pickle_file[0]
                loc = pickle_file[1]
                demand = pickle_file[2]
                cap = pickle_file[3]

                torch_array.append(
                    {
                        'loc': torch.FloatTensor(loc),
                        # Uniform 1 - 9, scaled by capacities
                        'demand': (torch.FloatTensor(demand).float() / cap),
                        'depot': torch.FloatTensor(depot)
                    })
        else:
            file_open = open(file_path , 'rb')
            pickle_file = pickle.load(file_open)
            depot = pickle_file[0]
            loc = pickle_file[1]
            demand = pickle_file[2]
            cap = pickle_file[3]

            torch_array.append(
                {
                    'loc': torch.FloatTensor(loc),
                    # Uniform 1 - 9, scaled by capacities
                    'demand': (torch.FloatTensor(demand).float() / cap),
                    'depot': torch.FloatTensor(depot)
                })


        return torch_array


    def __init__(self, filename=None, num_samples=1000000, offset=0, distribution=None, task=None):
        super(VRPDataset, self).__init__()
        print("VRP")

        self.data_set = []
        if filename is not None:
            #
            if (task['variation_type'] == 'cvrplib'):
                #         num_points = int(split_line[1])
                self.data = self.process_cvrp_lib(filename)
                # print(self.data[0])
                task['graph_size'] = len(self.data[0]['loc'])+1
                pass
                # save_dataset(self.data, filename + '.pkl')
                # save_dataset(self.data, filename + '.pkl')

            #
            else:

                assert os.path.splitext(filename)[1] == '.pkl'

                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                self.data = data#[make_instance(args) for args in data[offset:offset+num_samples]]
                print("self. data vrp ", self.data)

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                10: 20.,
                20: 30.,
                30: 35,
                40: 38, # 40 FOR CAPACITY VARIATION:
                50: 40.,

                80: 45 ,

                100: 50.,
                120:54,
                150:60,
                200:70

            }



            print("Task", task)

            if (task['variation_type'] == 'graph_size'):
                self.data = [
                    {
                        'loc': torch.FloatTensor(task['graph_size'], 2).uniform_(task['low'], task['high']),
                        # Uniform 1 - 9, scaled by capacities
                        'demand': (torch.FloatTensor(task['graph_size']).uniform_(0, 9).int() + 1).float() / CAPACITIES[
                            task['graph_size']],
                        'depot': torch.FloatTensor(2).uniform_(task['low'], task['high'])
                    }
                    for i in range(num_samples)
                ]


            if (task['variation_type'] == 'cap_vrp'):
                self.data = [
                    {
                        'loc': torch.FloatTensor(task['graph_size'], 2).uniform_(task['low'], task['high']),
                        # Uniform 1 - 9, scaled by capacities
                        'demand': (torch.FloatTensor(task['graph_size']).uniform_(0, 9).int() + 1).float() /
                            task['vrp_capacity'],
                        'depot': torch.FloatTensor(2).uniform_(task['low'], task['high'])
                    }
                    for i in range(num_samples)
                ]


            if (task['variation_type'] == 'scale'):
                self.data = [
                    {
                        'loc': torch.FloatTensor(task['graph_size'], 2).uniform_(task['low'], task['high']),
                        # Uniform 1 - 9, scaled by capacities
                        'demand': (torch.FloatTensor(task['graph_size']).uniform_(0, 9).int() + 1).float() / CAPACITIES[
                            task['graph_size']],
                        'depot': torch.FloatTensor(2).uniform_(task['low'], task['high'])
                    }
                    for i in range(num_samples)
                ]

            if (task['variation_type'] == 'distribution'):
                self.data = self.generate_GM_vrp_data(num_samples, task['graph_size'], task['num_modes'], CAPACITIES)


            if (task['variation_type'] == 'mix_distribution_size'):
                self.data = self.generate_GM_vrp_data(num_samples, task['graph_size'], task['num_modes'], CAPACITIES)




            # print("print sample ", self.data[5])

        self.size = len(self.data)

    def generate_GM_vrp_data(self, dataset_size, vrp_size, num_modes=-1,CAPACITIES=None, low=0, high=1, ):
        # "# GMM-9: each mode with N points; overall clipped to the 0-1 square\n",
        # "# sc: propto stdev of modes arounf the perfect grid; sc1: stdev at each mode\n",
        print("num modes ", num_modes)

        dataset = []

        remaining_elements = vrp_size

        for i in range(dataset_size):

            # dataset
            cur_gauss = np.empty([0, 2])
            remaining_elements = vrp_size

            modes_done = 0

            sc = 1. / 9.
            sc1 = .045

            elements_in_this_mode = remaining_elements

            rng = default_rng()
            z = array((1., 3., 5.)) / 6
            z = array(meshgrid(z, z))  # perfect grid\n",
            z += rng.uniform(-sc, sc, size=z.shape)  # shake it a bit\n",

            z = z.reshape(2, 9)

            cells_chosen = np.random.choice(9, num_modes, replace=False)

            mu_x_array = []
            mu_y_array = []
            for mode in cells_chosen:
                # grid_x = mode//3
                # grid_y = mode % 3
                mu_x = z[0][mode]
                mu_y = z[1][mode]
                mu_x_array.append(mu_x)
                mu_y_array.append(mu_y)

                elements_in_this_mode = int(remaining_elements / (num_modes - modes_done))

                samples_x = scipy.stats.truncnorm.rvs(
                    (low - mu_x) / sc1, (high - mu_x) / sc1, loc=mu_x, scale=sc1, size=elements_in_this_mode)

                samples_y = scipy.stats.truncnorm.rvs(
                    (low - mu_y) / sc1, (high - mu_y) / sc1, loc=mu_y, scale=sc1, size=elements_in_this_mode)
                #

                samples = np.stack((samples_x, samples_y), axis=1)

                cur_gauss = np.concatenate((cur_gauss, samples))

                # elements_in_this_mode = int(elements_in_this_mode)
                remaining_elements = remaining_elements - elements_in_this_mode
                modes_done += 1

                # print(cur_gauss)

            data = torch.Tensor(cur_gauss)
            loc = data.reshape(vrp_size, 2)

            data_append = {'loc': loc,
            # Uniform 1 - 9, scaled by capacities
            'demand': (torch.FloatTensor(vrp_size).uniform_(0, 9).int() + 1).float() / CAPACITIES[
               vrp_size],
            'depot': torch.FloatTensor(2).uniform_(low, high)}

            dataset.append(data_append)

        return dataset

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
