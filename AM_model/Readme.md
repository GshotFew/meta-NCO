## Dependencies
The code was tested using Python 3.7.8 on CentOS Linux 7. The dependencies can be installed using 

`python3 -m pip install -r requirements.txt` 

## Steps to train/test different models.

### Training scripts:
* `run.py` to train **meta-AM** model
* `run_multi.py` used to train **multi-AM** model
* `run_single.py` to train **original-AM** model

### Testing scripts:
* `testing_with_tuning.py` for testing **meta-AM/multi-AM** model after fine tuning
* `test_single.py` for testing the **original-AM** model without fine-tuning

### Parameters:
Check `options.py` in which each parameter has a 'help' attribute that gives its precise describtion

## Examples:
* To train the **meta-AM** model on small graphs (N=10,20,30,50) and test on unseen larger graphs for **CVRP**:
```sh
python3 run.py --problem cvrp --graph_size -1  --run_name <NAME_OF_MODEL>  --variation_type graph_size --baseline_every_Xepochs_for_META 7 
python3 testing_with_tuning.py --problem cvrp --graph_size -1  --run_name temp_val --load_path outputs/cvrp/SIZE/<NAME_OF_MODEL>/checkpoint.pt --variation_type graph_size --test_result_pickle_file <NAME_OF_MODEL>.pkl  
```
where <NAME_OF_MODEL> could be "SIZE_META". Please keep the name without spaces. 

* To train the **meta-AM** model on graphs of different scales (L=1,2,4) and test on unseen scales for **TSP**:
```sh
python3 run.py --graph_size 40  --run_name <NAME_OF_MODEL>  --variation_type scale --baseline_every_Xepochs_for_META 7
python3 testing_with_tuning.py --graph_size 40  --run_name temp_val --load_path outputs/tsp/SCALE/<NAME_OF_MODEL>/checkpoint.pt  --variation_type scale --test_result_pickle_file <NAME_OF_MODEL>.pkl  
```
where <NAME_OF_MODEL> could be "SCALE_META". Please keep the name without spaces. 

* To train the **multi-AM** on various numbers of modes M and test on an unseen mode for **TSP**:
```sh
python3 run_multi.py --graph_size 40  --run_name <NAME_OF_MODEL> --variation_type distribution
python3 testing_with_tuning.py --graph_size 40   --run_name temp_val   --variation_type distribution --load_path outputs/tsp/MODE/<NAME_OF_MODEL>/checkpoint.pt --test_result_pickle_file GRID_MULTI_40_3_modes.pkl `
```

* To train and test the **original-AM** model on graphs of size N=80 for **CVRP**:
```sh
python3 run_single.py --problem cvrp  --graph_size 80  --run_name <NAME_OF_MODEL> --variation_type graph_size 
python3 test_single.py --problem cvrp --graph_size 80  --run_name temp_val  --variation_type graph_size --load_path outputs/cvrp/SIZE/<NAME_OF_MODEL>/checkpoint.pt  --test_result_pickle_file GSIZE_SINGLE_80.pkl 
```

**original-AM** for scale L = 5 for **TSP**
* ##### Train:
`python3 run_single.py --graph_size 40  --run_name <NAME_OF_MODEL> --variation_type scale --scale 5 `


## Using pre-trained models
We also provide the models (**meta-AM**, **multi-AM** and **original-AM**) that we have trained on our side (for 24 hours using 1 GPU). They are stored in the `pretrained_models` folder.

### Example to test the meta-model for TSP graph size variation using the pre-trained meta-model
```sh
python3 testing_with_tuning.py --graph_size -1  --run_name temp_val --load_path pretrained_models/tsp/SIZE/META_10_20_30_50/checkpoint.pt  --variation_type graph_size --test_result_pickle_file temp.pkl 
```


## Credits 

Our code is built upon code provided by Kool  et al. https://github.com/wouterkool/attention-learn-to-route
