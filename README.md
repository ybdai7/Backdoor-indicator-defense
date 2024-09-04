# BackdoorIndiator

Official code implementation of **BackdoorIndicator: Leveraging OOD Data for
Proactive Backdoor Detection in Federated
Learning**(https://www.usenix.org/conference/usenixsecurity24/presentation/li-songze)

# Initialization
You first need to install relevant packages using:

    pip install -r requirement.txt

For the version of important packages:
    
    Python==3.7.15
    torch==1.13.0
    torchvision==0.14.0

For the edge-case datasets, you can acquire them following the instructions of https://github.com/ksreenivasan/OOD_Federated_Learning, 
which is the official repo of the Yes-you-can-really-backdoor-FL paper.

# First Run
The code trains an Federated Learning global model from scratch when you run the
code for the first time. It is recommended that do not apply any defense
mechanism, and only saves a few checkpoints of the global model for the first
run. To do this, you may set the follow parameters in
`utils/yamls/params_vanilla_indicator.yaml`:

    poisoned_start_round: 10000 #larger than the biggest global round index you want to save
    global_watermarking_start_round: 10000 

A recording folder will then be created based on the launching time under
`saved_models`, where checkpoints will be saved. Then, you may choose any
checkpoint, and put the path in the `resumed_model`:

    resumed_model: "Jun.05_06.09.03/saved_model_global_model_1200.pt.tar"
    save_on_round: [xxx, yyy, zzz] #any round you like

To launch any defense mechanism, you may then put the corresponding yaml file in
the command line. For example, to implement BackdoorIndicator, you may first
want to check `global_watermarking_start_round` and `poisoned_start_round`, as
these two parameters determine the round where BackdoorIndicator begins and the
poisoning begins. Then you run the code

    python main.py --GPU_id "x" --params utils/yamls/indicator/params_vanilla_Indicator.yaml

The results are recorded in the corresponding `saved_models` folder.

# Hyperparameters
Please feel free to change the following parameters in corresponding yamls to
see their influence to the proposed method, as it is discussed in the paper:

    ood_data_source
    ood_data_sample_lens
    global_retrain_no_times
    watermarking_mu

# Citation
We appreciate it if you would please cite the following paper if you found the
repository useful for your work:

    @misc{li2024backdoorindicator,
      title={BackdoorIndicator: Leveraging OOD Data for Proactive Backdoor Detection in Federated Learning}, 
      author={Songze Li and Yanbo Dai},
      year={2024},
      eprint={2405.20862},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
    }







