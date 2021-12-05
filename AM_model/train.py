import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to
import copy
import torch.optim as optim
from datetime import datetime
import pickle


def get_vrp_costs_norm(dataset, pi):
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
            torch.full_like(dataset['demand'][:, :1], -1.0),
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
        assert (used_cap <= 1.0+ 1e-5).all(), "Used more than capacity"

    # Gather dataset in order of tour
    loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
    d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

    # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
    return (
                   (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
                   + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
                   + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
           ), None

def get_costs_norm_tsp(dataset, pi):
    # Check that tours are valid, i.e. contain 0 to n -1
    assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
    ).all(), "Invalid tour"

    # Gather dataset in order of tour
    d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

    # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
    return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts, return_all_costs=False, return_pi = False):
    # Validate
    print('Validating...')
    if(return_pi):
        cost, pi = rollout(model, dataset, opts, return_pi = return_pi)
    else:
        cost = rollout(model, dataset, opts, return_pi = return_pi)

    avg_cost = cost.mean()

    print('Overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) ))

    if(return_all_costs ==True and return_pi):
        return avg_cost, cost, pi

    if(return_all_costs==True and return_pi==False):
        return avg_cost, cost

    return avg_cost


def rollout_for_plot(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            graph_embed = model(move_to(bat, opts.device), return_emb=True)
        return graph_embed.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def rollout(model, dataset, opts, return_pi=False):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _, pi = model(move_to(bat, opts.device), return_pi=True)
        return cost.data.cpu(), pi.cpu()


    if(return_pi ==False):
        return torch.cat([
            eval_model_bat(bat)[0]
            for bat
            in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
        ], 0)

    else:
        # torch.cat([
        cost_array = []
        pi_array  = []

        for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar):
            cost_, pi_ = eval_model_bat(bat)
            cost_array.append(cost_)
            pi_array.append(pi_)

        return torch.cat(cost_array,0), torch.cat(pi_array, 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [

        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped



def plot_tune_and_test(task, model_meta, baseline, epoch, val_dataset, problem, tb_logger, opts,fine_tuning_dataset=None, dict_results_task_sample_iter_wise=None):
    sequence_updated_reward = []

    print("task size ", task)
    step = 0
    start_time = time.time()

    COUNTER_FINE_TUNE = 0

    # TESTING
    
    print("fine tuning dataset size ", len(fine_tuning_dataset))

    if (opts.longer_fine_tune == 1):
        print("created new big fine tuning dataset")

        training_dataset = baseline.wrap_dataset(
            problem.make_dataset(num_samples=256000, distribution=opts.data_distribution, task=task))
        num_fine_tune_step_epochs = opts.test_num_step_epochs * 100
        num_batch_size = 512
        if (task['graph_size'] >= 150):
            num_batch_size = 256
    else:
        training_dataset = baseline.wrap_dataset(fine_tuning_dataset)
        num_fine_tune_step_epochs = opts.test_num_step_epochs  # not 30; it depends upon (fine tuning dataset used)
        num_batch_size = 256

    print("size of fine tuning dataset ", len(fine_tuning_dataset))

    print("nbs ", num_batch_size)

    rand_sampler = torch.utils.data.RandomSampler(training_dataset, num_samples=len(training_dataset), replacement=True)
    training_dataloader = DataLoader(training_dataset, batch_size=num_batch_size, num_workers=1, sampler = rand_sampler )

    # Put model in train mode!

    model_task = copy.deepcopy(model_meta)

    print(" BEFORE TUNING size  ", task)
    graph_embed_before = rollout_for_plot(model_task, val_dataset, opts)
    print("EMB VAL BEFORE TUNE ", graph_embed_before)

    model_task.train()

    set_decode_type(model_task, "sampling")


    optimizer = optim.Adam(model_task.parameters(), lr=opts.lr_model*(0.1))

    print("num_fine_tune_step_epochs", num_fine_tune_step_epochs)
    for outer_step_id in range(num_fine_tune_step_epochs):
        print("outer_step_id", outer_step_id)

        for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):


            print("COUNTER FINE TUNE STEP ", COUNTER_FINE_TUNE)

            train_batch(
                model_task,
                optimizer,
                baseline,
                epoch,
                batch_id,
                step,
                batch,
                tb_logger,
                opts
            )

            step += 1

            print("COUNTER FINE TUNE ", COUNTER_FINE_TUNE)

            COUNTER_FINE_TUNE += 1

            model_task.train()

            set_decode_type(model_task, "sampling")

            cur_time = datetime.now()

    print(" AFTER TUNING size ", task)
    graph_embed_after = rollout_for_plot(model_task, val_dataset, opts)
    print(" EMBED AFTER TUNING ", graph_embed_after)


    print(" length ", task, " ::: ")
    for index, x in enumerate( sequence_updated_reward):

        print(x.item())

    return graph_embed_before, graph_embed_after



def tune_and_test(task, model_meta, baseline, epoch, val_dataset, problem, tb_logger, opts,fine_tuning_dataset=None, dict_results_task_sample_iter_wise=None):
    sequence_updated_reward = []
    sequence_updated_all_costs = []

    print("task  ", task)
    step=0
    start_time = time.time()

    COUNTER_FINE_TUNE = 0

    # TESTING

    if(opts.longer_fine_tune==1):
        print("created new big fine tuning dataset")

        training_dataset = baseline.wrap_dataset(
            problem.make_dataset(num_samples=256000, distribution=opts.data_distribution, task=task))
        num_fine_tune_step_epochs = opts.test_num_step_epochs*100
        num_batch_size = 512
        if(task['graph_size'] >=150):
            num_batch_size = 256
    else:
        training_dataset = baseline.wrap_dataset(fine_tuning_dataset)
        num_fine_tune_step_epochs = opts.test_num_step_epochs # not 30; it depends upon (fine tuning dataset used)
        num_batch_size = 256

        if (task['graph_size'] >= 150):
            num_batch_size = 128

        if('cvrplib' in task):
            num_batch_size=1


    print(" fine tune loaded from file")
    print("size of fine tuning dataset ", len(fine_tuning_dataset))


    print("nbs ", num_batch_size)


    rand_sampler = torch.utils.data.RandomSampler(training_dataset, num_samples=len(training_dataset), replacement=True)
    training_dataloader = DataLoader(training_dataset, batch_size=num_batch_size, num_workers=1, sampler = rand_sampler )


    model_task = copy.deepcopy(model_meta)

    print(" BEFORE TUNING size  ", task)
    # avg_reward, all_costs, pi_before = validate(model_task, val_dataset, opts,  return_all_costs=True, return_pi=True)
    avg_reward, all_costs = validate(model_task, val_dataset, opts,  return_all_costs=True)
    print(" BEFORE TUNING ", avg_reward)

    if('miplib' in task):
        original_tsp_instance = pickle.load(open('data/tsplib/tsp/'+task['original'],'rb'))
        original_tsp_instance = torch.from_numpy(original_tsp_instance)
        original_tsp_instance = original_tsp_instance.reshape(1,-1,2)
        tsp_tour_cost_original = get_costs_norm_tsp(original_tsp_instance, pi_before)[0]
        avg_reward=tsp_tour_cost_original


    if('cvrplib' in task):
        test_path_original = opts.test_file_original_full_path_cvrplib
        print(" test_path original unnormalized file ", test_path_original)
        print(" test_file unnormalized", test_path_original)
        original_cvrp_instance = pickle.load(open(test_path_original,'rb'))
        depot = original_cvrp_instance[0]
        loc = original_cvrp_instance[1]
        demand = original_cvrp_instance[2]
        cap = original_cvrp_instance[3]

        torch_instance_cvrp=            {
                'loc': torch.FloatTensor(loc),
                # Uniform 1 - 9, scaled by capacities
                'demand': (torch.FloatTensor(demand).float() / cap),
                'depot': torch.FloatTensor(depot)
            }
        torch_instance_cvrp['demand'] = torch_instance_cvrp['demand'][None,:]
        torch_instance_cvrp['loc'] = torch_instance_cvrp['loc'][None,:]
        torch_instance_cvrp['depot'] = torch_instance_cvrp['depot'][None,:]

        vrp_tour_cost_original = get_vrp_costs_norm(torch_instance_cvrp, pi_before)[0]
        avg_reward=vrp_tour_cost_original



    begin_time = datetime.now()
    cur_time = datetime.now()
    dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE] = {}
    dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE]['cost'] = all_costs
    if(opts.rescale_for_testing is not None): # only for scratch part since we didn't want to train again.
        dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE]['cost'] = dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE]['cost']*(task['rescale_for_testing']/3.0)

    if(COUNTER_FINE_TUNE%50==0):
        if('miplib' in task or 'cvrplib' in task):
            dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE]['pi'] = pi_before
        dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE]['unnormalized'] = avg_reward.item()
        print(" COUNTER FINE TUNE ", COUNTER_FINE_TUNE ,"   ", avg_reward.item())

    dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE]['current_time'] = cur_time

    sequence_updated_reward.append(avg_reward)

    model_task.train()

    set_decode_type(model_task, "sampling")

    optimizer = optim.Adam(model_task.parameters(), lr=opts.lr_model*(0.1))


    print("num_fine_tune_step_epochs", num_fine_tune_step_epochs)
    time_spent_in_fine_tuning = 0


    for outer_step_id in range(num_fine_tune_step_epochs):
        print("outer_step_id", outer_step_id)

        if (time_spent_in_fine_tuning > 180 or (COUNTER_FINE_TUNE >= 100 and opts.longer_fine_tune == 0)):
            break

        for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

            print("COUNTER FINE TUNE ", COUNTER_FINE_TUNE)

            time_before_update = datetime.now()

            train_batch(
                model_task,
                optimizer,
                baseline,
                epoch,
                batch_id,
                step,
                batch,
                tb_logger,
                opts
            )
            time_after_update = datetime.now()
            time_taken_for_update = (time_after_update - time_before_update).total_seconds() / 60.0

            time_spent_in_fine_tuning += time_taken_for_update

            step += 1

            print("COUNTER FINE TUNE STEP :: ", COUNTER_FINE_TUNE)

            if(COUNTER_FINE_TUNE%10==0 or COUNTER_FINE_TUNE==1):
            # if(COUNTER_FINE_TUNE%1==0 or COUNTER_FINE_TUNE==1):

                if ('miplib' in task or 'cvrplib' in task):
                    updated_reward, updated_all_costs,updated_pi = validate(model_task, val_dataset, opts, return_all_costs=True, return_pi=True)
                else:
                    updated_reward, updated_all_costs = validate(model_task, val_dataset, opts, return_all_costs=True,
                                                             return_pi=False)
                print(" REWARD(tour length) AFTER TUNING ", updated_reward)

                #
                if('miplib' in task):
                    print("MIPLIB FILE")
                    tsp_tour_cost_original = get_costs_norm_tsp(original_tsp_instance, updated_pi)[0]
                    print(" tsp_tour_cost_original ", tsp_tour_cost_original)
                    updated_reward = tsp_tour_cost_original

                if ('cvrplib' in task):

                    # original_cvrp_instance = original_cvrp_instance.reshape(1,-1,2)
                    vrp_tour_cost_original = get_vrp_costs_norm(torch_instance_cvrp, updated_pi)[0]
                    print(" vrp_tour_cost_original ", vrp_tour_cost_original)
                    updated_reward = vrp_tour_cost_original
                #normalized
                #

                sequence_updated_reward.append(updated_reward)

                if(dict_results_task_sample_iter_wise is not None):
                    dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE] = {}
                    dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE]['cost'] = updated_all_costs

                    dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE]['pi'] = None
                    dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE]['current_time'] = datetime.now()
                    dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE]['time_spent_in_fine_tuning'] = time_spent_in_fine_tuning

                    #
                    dict_results_task_sample_iter_wise[COUNTER_FINE_TUNE]['unnormalized'] = updated_reward.item()
                    #

            COUNTER_FINE_TUNE += 1
            model_task.train()

            set_decode_type(model_task, "sampling")

            cur_time = datetime.now()
            time_diff = (cur_time - begin_time).total_seconds() / 60.0

            print("time diff since first fine tune step time_spent_in_fine_tuning", time_spent_in_fine_tuning, " minutes ")

            if (time_spent_in_fine_tuning > 180 or ( COUNTER_FINE_TUNE ==250000 and opts.longer_fine_tune ==0 )):
                return updated_reward

        if ('cvrplib' in task):
            if (COUNTER_FINE_TUNE > 100):
                break


    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))



    updated_reward = validate(model_task, val_dataset, opts)

    if (num_fine_tune_step_epochs == 0):
        print("****** No fine tuning done **** ")
    else:
        print(" AFTER TUNING size ", task)
        print(" REWARD AFTER TUNING ", updated_reward)


    print(" Task description ", task, " ::: ")
    for index, x in enumerate( sequence_updated_reward):
        print(x.item())


    # print("dict_results_task_sample_iter_wise",dict_results_task_sample_iter_wise)
    return updated_reward


def tune_and_validate(task, model_meta, baseline, epoch, val_dataset, problem, tb_logger, opts, cost_ins_heur):
    sequence_updated_opt_gap_wrt_ins = []
    step=0

    print("task size ", task)
    start_time = time.time()
    begin_time = datetime.now()

    COUNTER_FINE_TUNE = 0
    num_samples_query = 2560
    num_fine_tune_steps = opts.k_tune_steps//10
    

    print("num_fine_tune_steps = ",num_fine_tune_steps )

    if(num_fine_tune_steps ==0):
        num_fine_tune_steps =1
    print("nsq ", num_samples_query)

    training_dataset = baseline.wrap_dataset(
        problem.make_dataset(num_samples=num_samples_query, distribution=opts.data_distribution, task=task))
    print(" gen fine tune")

    num_batch_size = 256

    print("nbs ", num_batch_size)


    rand_sampler = torch.utils.data.RandomSampler(training_dataset, num_samples=len(training_dataset), replacement=True)
    training_dataloader = DataLoader(training_dataset, batch_size=num_batch_size, num_workers=1, sampler = rand_sampler )

    # # Put model in train mode!
    model_task = copy.deepcopy(model_meta)

    print(" BEFORE TUNING size  ", task)
    avg_reward, all_costs = validate(model_task, val_dataset, opts,  return_all_costs=True, return_pi=False)
    print(" BEFORE TUNING ", avg_reward)

    scale_multiply = 1
    if (opts.variation_type == 'scale'):
        scale_multiply = task['high'] - task['low']


    opt_gap_wrt_ins_heu_cur_task = 0
    for before_fine_tune_cost, ins_cost in zip(all_costs, cost_ins_heur):
        opt_gap_wrt_ins_heu_cur_task += (before_fine_tune_cost.item() - (ins_cost*scale_multiply)) * 100.0 / ((ins_cost*scale_multiply))

    opt_gap_wrt_ins_heu_cur_task = opt_gap_wrt_ins_heu_cur_task/len(cost_ins_heur)


    print("opt_gap_wrt_ins_heu_cur_task before", opt_gap_wrt_ins_heu_cur_task)



    sequence_updated_opt_gap_wrt_ins.append(avg_reward)

    model_task.train()
    set_decode_type(model_task, "sampling")


    optimizer = optim.Adam(model_task.parameters(), lr=opts.lr_model*(0.1))
    print("optimizer ", optimizer)

    opt_gap_wrt_ins_heu_cur_task_updated =None


    for outer_step_id in range(num_fine_tune_steps):

        for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):


            # for less than 10 steps hack
            if(opts.k_tune_steps <10):
                if(batch_id == opts.k_tune_steps):
                    break

            print("batch size ", len(batch))
            print( "TIME " ,time.time())
            print("COUNTER FINE TUNE ", COUNTER_FINE_TUNE)

            train_batch(
                model_task,
                optimizer,
                baseline,
                epoch,
                batch_id,
                step,
                batch,
                tb_logger,
                opts
            )

            step += 1

            print("COUNTER FINE TUNE ", COUNTER_FINE_TUNE)
            if (COUNTER_FINE_TUNE % 10 == 0):

                updated_reward, updated_all_costs = validate(model_task, val_dataset, opts,
                                                                         return_all_costs=True, return_pi=False)
                print(" REWARD AFTER TUNING ", updated_reward)

                opt_gap_wrt_ins_heu_cur_task_updated = 0

                for meta_tune_cost, ins_cost in zip(updated_all_costs, cost_ins_heur):
                    opt_gap_wrt_ins_heu_cur_task_updated += (meta_tune_cost.item() - (
                                ins_cost * scale_multiply)) * 100.0 / ((ins_cost * scale_multiply))

                opt_gap_wrt_ins_heu_cur_task_updated = opt_gap_wrt_ins_heu_cur_task_updated / len(updated_all_costs)

                print("opt_gap_wrt_ins_heu_cur_task after fine tune step ", COUNTER_FINE_TUNE,
                      opt_gap_wrt_ins_heu_cur_task_updated)

                sequence_updated_opt_gap_wrt_ins.append(opt_gap_wrt_ins_heu_cur_task_updated)

            COUNTER_FINE_TUNE += 1

            model_task.train()
            set_decode_type(model_task, "sampling")

            cur_time = datetime.now()
            time_diff = (cur_time - begin_time).total_seconds() / 60.0
            if (time_diff > 60):
                return updated_reward


    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))


    print(" AFTER TUNING size ", task)
    updated_reward = validate(model_task, val_dataset, opts)
    print(" REWARD AFTER TUNING ", updated_reward)


    print(" length ", task, " ::: ")
    for index, x in enumerate( sequence_updated_opt_gap_wrt_ins):
        print(x.item())


    return opt_gap_wrt_ins_heu_cur_task_updated



def train_epoch(task, model_meta, baseline, epoch, val_dataset, problem, tb_logger, opts, alpha, test=False):
    print("task size ", task)
    step=0
    start_time = time.time()

    # Generate new training data for each epoch

    epoch_size = opts.batch_size*opts.k_tune_steps
    print("epoch size for fine tuning ", epoch_size)
    training_dataset = baseline.wrap_dataset(problem.make_dataset(num_samples=epoch_size, distribution=opts.data_distribution, task=task))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)


    # alpha =opts.alpha
    print("alpha is ", alpha)
    # Put model in train mode!
    current_weights = copy.deepcopy(model_meta.state_dict())
    # model_task = copy.deepcopy(model_meta)

    if(epoch%20==0):
        print(" BEFORE TUNING size  ", task)
        avg_reward = validate(model_meta, val_dataset, opts)
        print(" BEFORE TUNING ", avg_reward)

    model_meta.train()

    set_decode_type(model_meta, "sampling")

    optimizer = optim.Adam(model_meta.parameters(), lr=opts.lr_model)

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(
            model_meta,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        step += 1

    candidate_weights = model_meta.state_dict()

    state_dict = {candidate: (current_weights[candidate] + alpha *
                              (candidate_weights[candidate] - current_weights[candidate]))
                  for candidate in candidate_weights}
    
    # print(" baseline every {} epoch ".format(opts.baseline_every_Xepochs_for_META))
    if(epoch%opts.baseline_every_Xepochs_for_META==0):
        print(" AFTER TUNING size ", task)
        avg_reward = validate(model_meta, val_dataset, opts)
        print(" REWARD AFTER TUNING ", avg_reward)

        baseline.epoch_callback(model_meta, epoch)
        
        
    model_meta.load_state_dict(state_dict)

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))




def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss
    # print(" inner loss ", loss)
    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()




def query_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    print("qyer los ", loss)

    return loss
