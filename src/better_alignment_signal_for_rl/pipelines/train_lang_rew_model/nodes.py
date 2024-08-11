"""
This is a boilerplate pipeline 'train_lang_rew_model'
generated using Kedro 0.19.3
"""
from copy import deepcopy
from pathlib import Path
from tqdm.auto import tqdm 
import torch as th
from icecream import ic 
from better_alignment_signal_for_rl.lang_rew_model_backbone.model.base import BaseModel
from better_alignment_signal_for_rl.lang_rew_model_backbone.model.traj_recog_model import TrajRecogLangRewModel
from better_alignment_signal_for_rl.lang_rew_model_backbone.model.cosine_sim_model import CosineSimLangRewModel
from better_alignment_signal_for_rl.agent_components.optimizer import get_optims_and_scheduler
import wandb 
import os 
from natsort import natsorted
from glob import glob

# ! we need to 
# 1. setup the model
# 2. iterate over the data and train the model
# 3. save the model when the epoch ends 
# 4. logging the model performance
# 5. early stopping according to performance in validation set


def generate_tag_lst_and_str(lang_rew_model_cfg, general_cfg):
    has_hard_signal = lang_rew_model_cfg['cosine_sim_based_params']['has_hard_signal']
    cls_weight = lang_rew_model_cfg['recognition_based_params']['cls_weight']
    minigrid_no_pretrain = lang_rew_model_cfg['model_kwargs']['minigrid_no_pretrain']
    env_name = general_cfg['env_name']

    mix_config = dict()
    mix_config.update(lang_rew_model_cfg)
    mix_config.update(general_cfg)
    tag_lst = [f"is_mark_{lang_rew_model_cfg['is_markovian']}", f"type_{lang_rew_model_cfg['lang_rew_type']}", f"extra_mnpl_{lang_rew_model_cfg['has_extra_data_manipulation']}", f"traj_l_{lang_rew_model_cfg['traj_length']}"]
    
    if lang_rew_model_cfg['lang_rew_type'] == "cosine_similarity":
        tag_lst.append(f"hard_{has_hard_signal}")
    elif lang_rew_model_cfg['lang_rew_type'] == "trajectory_recognition":
        tag_lst.append(f"cls_w_{cls_weight}")
        
    if env_name == "minigrid":
        if minigrid_no_pretrain:
            tag_lst.append("no_pretrain")
        
    tag_lst = natsorted(tag_lst)
    # ! tag_lst_str will not record has_hard_signal
    tag_lst_cache_for_str = deepcopy(tag_lst)
    tag_lst_cache_for_str.remove(f"hard_{has_hard_signal}")
    tag_lst_str = "-".join(tag_lst_cache_for_str)
    return tag_lst, tag_lst_str, mix_config

def setup_lang_rew_model(lang_rew_model_cfg, general_cfg):
    # setup the model and set it to train 
    
    is_markovian = lang_rew_model_cfg['is_markovian']
    traj_length = lang_rew_model_cfg['traj_length']
    
    if is_markovian:
        assert traj_length == 2
    else:
        assert traj_length > 2 
    
    has_data_augmentation = lang_rew_model_cfg['has_data_augmentation']
    assert has_data_augmentation
    has_extra_data_manipulation = lang_rew_model_cfg['has_extra_data_manipulation']
    lang_rew_type = lang_rew_model_cfg['lang_rew_type']
    
    model_kwargs = lang_rew_model_cfg['model_kwargs']
    env_name = general_cfg['env_name']
    pretrained_model_cls = model_kwargs['pretrained_model_cls']
    pretrained_model_output_size = general_cfg['constant']['clip_embeds_size']
    is_pretrained_module_freezed = model_kwargs['is_pretrained_module_freezed']
    minigrid_no_pretrain = model_kwargs['minigrid_no_pretrain'] 
    assert is_pretrained_module_freezed
    rnn_model_cls = model_kwargs['rnn_model_cls']
    rnn_kwargs = model_kwargs['rnn_kwargs']
    dense_kwargs = model_kwargs['dense_kwargs']
    
    if lang_rew_type == "cosine_similarity":
        has_hard_signal = lang_rew_model_cfg['cosine_sim_based_params']['has_hard_signal']
        alpha = lang_rew_model_cfg['cosine_sim_based_params']['alpha']
        lang_model = CosineSimLangRewModel(
            pretrained_model_cls = pretrained_model_cls, 
            pretrained_model_output_size = pretrained_model_output_size, 
            is_pretrained_module_freezed = is_pretrained_module_freezed, 
            is_markovian = is_markovian, 
            rnn_model_cls = rnn_model_cls, 
            rnn_kwargs = rnn_kwargs, 
            dense_kwargs = dense_kwargs, 
            has_extra_data_manipulation = has_extra_data_manipulation, 
            traj_length = traj_length,
            has_hard_signal = has_hard_signal, 
            alpha=alpha,
            env_name = env_name, 
            minigrid_no_pretrain = minigrid_no_pretrain,
        )
        
    elif lang_rew_type == "trajectory_recognition":
        cls_weight = lang_rew_model_cfg['recognition_based_params']['cls_weight']
        
        lang_model = TrajRecogLangRewModel(
            pretrained_model_cls = pretrained_model_cls, 
            pretrained_model_output_size = pretrained_model_output_size, 
            is_pretrained_module_freezed = is_pretrained_module_freezed, 
            is_markovian = is_markovian, 
            rnn_model_cls = rnn_model_cls, 
            rnn_kwargs = rnn_kwargs, 
            dense_kwargs = dense_kwargs, 
            has_extra_data_manipulation = has_extra_data_manipulation, 
            cls_weight = cls_weight, 
            minigrid_no_pretrain = minigrid_no_pretrain,
            env_name=env_name,
            traj_length=traj_length,
        )
    
    else:
        raise NotImplementedError(f"lang_rew_type: {lang_rew_type} is not implemented")
    return lang_model




def process_and_forward_helper(data, lang_model, has_extra_data_manipulation, device):
    """
    Process and forward the data through the language reward model.

    Args:
        data (tuple): A tuple containing trajectory data and instruction data.
        lang_model (object): The language reward model.
        has_extra_data_manipulation (bool): Indicates whether extra data manipulation is required.
        device (str): The device to use for computation.

    Returns:
        list: A list of loss values for each processed data.

    Raises:
        NotImplementedError: If the lang_rew_type is not implemented.
    """
    
    traj_d, instr_d = data
    # traj_d shape (batch, seq_len, 3, H, W)
    if isinstance(lang_model, CosineSimLangRewModel):
        target_cls = None
    elif isinstance(lang_model, TrajRecogLangRewModel):
        target_cls = lang_model.get_raw_target_cls(
            vision_input=traj_d, 
        )
    if has_extra_data_manipulation:
        if isinstance(lang_model, CosineSimLangRewModel):
            data_lst = lang_model.generate_extra_hard_negatives(traj_d, instr_d)
        elif isinstance(lang_model, TrajRecogLangRewModel):
            data_lst = lang_model.generate_extra_hard_negatives(
                traj_d, instr_d, target_cls
            )
    else:
        data_lst = [(traj_d, instr_d, target_cls)]
        
    loss_val_lst = []
    cp_threshold_lst = [] 
    f1_lst = []
    precision_lst = []
    recall_lst = [] 
    accuracy_lst = [] 
    for traj_d, instr_d, target_cls in data_lst:

        inputs = lang_model.clip_format_prepare(traj_d, instr_d)
        # to device
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        if target_cls is not None:
            target_cls = target_cls.to(device, non_blocking=True)
            
        # forward
        outputs = lang_model(inputs)
        if isinstance(lang_model, CosineSimLangRewModel):
            vision_embeds, text_embeds = outputs
            # ic(vision_embeds.shape) # torch.Size([32, 512])
            # ic(text_embeds.shape)   # torch.Size([32, 512]) 
            # ! calculate loss 
            loss_info = lang_model.compute_losses(
                vision_embeds=vision_embeds,
                text_embeds=text_embeds,
                instr_d = instr_d,
            )
            loss = loss_info['loss']
            if lang_model.training: # only do backpropagation during training
                loss.backward()
                
            if "cp_threshold" in loss_info: # Cosine model may have cp_threshold key 
                cp_threshold = loss_info['cp_threshold']
                cp_threshold_lst.append(cp_threshold)
                
            for metric in ["f1", "precision", "recall"]:
                if metric in loss_info:
                    metric_val = loss_info[metric]
                    if metric == "f1":
                        f1_lst.append(metric_val)
                    elif metric == "precision":
                        precision_lst.append(metric_val)
                    elif metric == "recall":
                        recall_lst.append(metric_val)
                    
            
        elif isinstance(lang_model, TrajRecogLangRewModel):
            # ! calculate loss 
            loss_info = lang_model.compute_losses(
                outputs=outputs,
                target_cls=target_cls,
            )
            loss = loss_info['loss']
            if lang_model.training:
                loss.backward() 
                
        else:
            raise NotImplementedError(f"The lang_rew_type is not implemented")
        
        loss_val_lst.append(loss.item())
        if "accuracy" in loss_info: # accuracy may apply to all models 
            accuracy_lst.append(loss_info['accuracy'])
        
    output = dict()
    output['loss_val_lst'] = loss_val_lst
    if len(cp_threshold_lst) > 0:
        output['cp_threshold_lst'] = cp_threshold_lst
        
    if len(f1_lst) > 0:
        output['f1_lst'] = f1_lst
    if len(precision_lst) > 0:
        output['precision_lst'] = precision_lst
    if len(recall_lst) > 0:
        output['recall_lst'] = recall_lst
    if len(accuracy_lst) > 0:
        output['accuracy_lst'] = accuracy_lst
    
    return output 
        
def logging_performance_helper(status_name, metric_stats):
    loss_epoch = 0 
    precision_epoch = 0
    recall_epoch = 0 
    f1_epoch = 0 
    accuracy_epoch = 0 
    nupdate = 0 
    
    loss_val_lst = metric_stats['loss_val_lst']
            
    for loss in loss_val_lst:
        loss_epoch += loss
        nupdate += 1
        
    pbar_desc = f"{status_name} Loss: {round(loss_epoch /nupdate, 4)}"
    
        
    if 'precision_lst' in metric_stats:
        precision_lst = metric_stats['precision_lst']
        for precision in precision_lst:
            if isinstance(precision, dict): # in case we have multiple cp_threshold and the precision is recorded in a dictionary 
                value = precision[list(precision.keys())[0]]
                precision_epoch += value
            else:
                precision_epoch += precision
                
        pbar_desc += f" Precision: {round(precision_epoch /nupdate, 4)}"
        
    if 'recall_lst' in metric_stats:
        recall_lst = metric_stats['recall_lst']
        for recall in recall_lst:
            if isinstance(recall, dict):
                value = recall[list(recall.keys())[0]]
                recall_epoch += value
            else:
                recall_epoch += recall
                
        pbar_desc += f" Recall: {round(recall_epoch /nupdate, 4)}"
                
    if 'f1_lst' in metric_stats:
        f1_lst = metric_stats['f1_lst']
        for f1 in f1_lst:
            if isinstance(f1, dict):
                value = f1[list(f1.keys())[0]]
                f1_epoch += value
            else:
                f1_epoch += f1
                
        pbar_desc += f" F1: {round(f1_epoch /nupdate, 4)}"
        
    if "accuracy_lst" in metric_stats:
        accuracy_lst = metric_stats['accuracy_lst']
        for accuracy in accuracy_lst:
            if isinstance(accuracy, dict):
                value = accuracy[list(accuracy.keys())[0]]
                accuracy_epoch += value
            else:
                accuracy_epoch += accuracy
                
        pbar_desc += f" Accuracy: {round(accuracy_epoch /nupdate, 4)}"

    return loss_epoch, precision_epoch, recall_epoch, f1_epoch, accuracy_epoch, pbar_desc, nupdate
    
def accumulate_stats_dict_helper(mdict, stats_lst):
    for ele in stats_lst:
        if isinstance(ele, dict):
            for k, v in ele.items():
                if k not in mdict:
                    mdict[k] = v 
                else:
                    mdict[k] += v
        else:
            if "default" not in mdict:
                mdict['default'] = ele
            else:
                mdict['default'] += ele

# ! NODE
def setup_n_train_lrm(
    train_dataloader,
    validate_dataloader,
    test_dataloader,
    lang_rew_model_cfg, 
    general_cfg
    ):
    
    env_name = general_cfg['env_name']
    device = general_cfg['device']
    # * add epoch number for montezuma env because it has limited number of training data
    if env_name == "montezuma":
        factor = 10
        lang_rew_model_cfg['algorithm_kwargs']['nepoch'] = lang_rew_model_cfg['algorithm_kwargs']['nepoch'] * factor
        lang_rew_model_cfg['algorithm_kwargs']['save_interval'] = lang_rew_model_cfg['algorithm_kwargs']['save_interval'] * factor

    lang_model : BaseModel = setup_lang_rew_model(lang_rew_model_cfg, general_cfg)
    has_extra_data_manipulation = lang_rew_model_cfg['has_extra_data_manipulation']
    device = general_cfg['device']
    
    # algo kwargs 
    algorithm_kwargs = lang_rew_model_cfg['algorithm_kwargs']
    nepoch = algorithm_kwargs['nepoch']
    optimizer_cls = algorithm_kwargs['optimizer_cls']
    wd = algorithm_kwargs['wd']
    lr = algorithm_kwargs['lr']
    save_interval = algorithm_kwargs['save_interval']
    continue_training = algorithm_kwargs['continue_training']
    steps_per_epoch = len(train_dataloader)
    
    # ! wandb setup 
    tag_lst, tag_lst_str, mix_config = generate_tag_lst_and_str(lang_rew_model_cfg, general_cfg)
    save_dir = os.path.join(os.environ['PWD'], "data/04_lang_rew_model_checkpoint", "lang_rew_models",f"{env_name}",tag_lst_str)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    wandb.init(
        project="Better Vision Language Alignment Signal for RL",
        name=f"Train Lang Rew Model on {env_name}",
        config=mix_config,
        tags=tag_lst
    )
    
    wandb.watch(lang_model)
    
    # load model if continue_training is True
    if continue_training:
        checkpoint_paths = glob(os.path.join(save_dir, "checkpoint_*.pth"))
        if len(checkpoint_paths) == 0:
            raise FileNotFoundError(f"No checkpoint found in {save_dir}")
        checkpoint_paths = natsorted(checkpoint_paths)
        load_path = checkpoint_paths[-1]
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model checkpoint not found at {load_path}")
        lang_model.load_state_dict(th.load(load_path))
    
    lang_model = lang_model.to(device)
    
    
    # ! setup optimizer
    optimizer, lr_scheduler = get_optims_and_scheduler(
        model = lang_model , 
        optimizer_cls = optimizer_cls , 
        wd = wd , 
        lr = lr , 
        n_epochs = nepoch , 
        steps_per_epoch = steps_per_epoch , 
    )

    best_train_loss = 99999999
    best_train_precision = 0
    best_train_recall = 0
    best_train_f1 = 0 
    best_train_accuracy = 0 
    
    
    best_val_loss = 99999999
    best_val_precision = 0 
    best_val_recall = 0
    best_val_f1 = 0
    best_val_accuracy = 0
    
    best_epoch_for_loss = 0 
    best_epoch_for_precision = 0  
    best_epoch_for_f1 = 0 
    best_epoch_for_accuracy = 0
    
    best_test_loss = 99999999
    best_test_precision = 0
    best_test_recall = 0
    best_test_f1 = 0 
    best_test_accuracy = 0 
    
    early_break_signal = False 
    
    for epoch in tqdm(range(nepoch), desc='Epoch'):
        # ! train the model 
        lang_model.train()
        train_loss_epoch = 0 
        train_precision_epoch = 0
        train_recall_epoch = 0 
        train_f1_epoch = 0 
        train_accuracy_epoch = 0 
        
        nupdate = 0 
        
        optimizer.zero_grad()
        for data in (trainpbar:=tqdm(train_dataloader, desc='Train')):
            metric_stats = process_and_forward_helper(data, lang_model, has_extra_data_manipulation, device)
            
            loss_epoch, precision_epoch, recall_epoch, f1_epoch, accuracy_epoch, pbar_desc, local_nupdate = logging_performance_helper("Train", metric_stats)
            train_loss_epoch += loss_epoch
            train_precision_epoch += precision_epoch
            train_recall_epoch += recall_epoch
            train_f1_epoch += f1_epoch
            train_accuracy_epoch += accuracy_epoch
            nupdate += local_nupdate
                    
            # ic(outputs.shape) # torch.Size([32, 9, 3]) GPU mem ~= 6300MB, CPU mem requirement ~= 60GB, training time ~= 1.0 hour per epoch, if batch size = 64, GPU mem ~= 11185MB
            
            trainpbar.set_description(pbar_desc) 
                
            # ! update optimizer        
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            
        train_loss_epoch /= nupdate
        train_precision_epoch /= nupdate
        train_recall_epoch /= nupdate
        train_f1_epoch /= nupdate
        train_accuracy_epoch /= nupdate
        
        if train_loss_epoch < best_train_loss:
            best_train_loss = train_loss_epoch
        if train_precision_epoch > best_train_precision:
            best_train_precision = train_precision_epoch
        if train_recall_epoch > best_train_recall:
            best_train_recall = train_recall_epoch
        if train_f1_epoch > best_train_f1:
            best_train_f1 = train_f1_epoch
        if train_accuracy_epoch > best_train_accuracy:
            best_train_accuracy = train_accuracy_epoch
        
        # ! eval validation 
        lang_model.eval()
        val_loss_epoch = 0
        val_precision_epoch = 0
        val_recall_epoch = 0 
        val_f1_epoch = 0 
        val_accuracy_epoch = 0 
        
        cp_threshold_dict = dict()
        
        
        nupdate = 0
        
        for data in (valpbar:=tqdm(validate_dataloader, desc='Val')):
            metric_stats = process_and_forward_helper(data, lang_model, False, device)
            loss_epoch, precision_epoch, recall_epoch, f1_epoch, accuracy_epoch, pbar_desc, local_nupdate = logging_performance_helper("Validate", metric_stats)
            
            if "cp_threshold_lst" in metric_stats:
                cp_threshold_lst = metric_stats['cp_threshold_lst']
                accumulate_stats_dict_helper(cp_threshold_dict, cp_threshold_lst)
            
            val_loss_epoch += loss_epoch
            val_precision_epoch += precision_epoch
            val_recall_epoch += recall_epoch
            val_f1_epoch += f1_epoch
            val_accuracy_epoch += accuracy_epoch
            nupdate += local_nupdate
            
            valpbar.set_description(pbar_desc)
            
        val_loss_epoch /= nupdate
        val_precision_epoch /= nupdate
        val_recall_epoch /= nupdate
        val_f1_epoch /= nupdate
        val_accuracy_epoch /= nupdate
        
        for k in cp_threshold_dict:
            cp_threshold_dict[k] = cp_threshold_dict[k] / nupdate
            
        # update the model's cp_threshold
        if len(cp_threshold_dict) > 0 and isinstance(lang_model, CosineSimLangRewModel):
            lang_model.cp_thred_dict = cp_threshold_dict
            
            
        # --- save best ---
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_epoch_for_loss = epoch
            best_save_path = os.path.join(save_dir, "best_for_loss.pth")
            th.save(lang_model.state_dict(), best_save_path)
        if val_precision_epoch > best_val_precision:
            best_val_precision = val_precision_epoch
            best_epoch_for_precision = epoch
            best_save_path = os.path.join(save_dir, "best_for_precision.pth")
            th.save(lang_model.state_dict(), best_save_path)
        
        if val_recall_epoch > best_val_recall:
            best_val_recall = val_recall_epoch
            
        if val_f1_epoch > best_val_f1:
            best_val_f1 = val_f1_epoch
            best_epoch_for_f1 = epoch
            best_save_path = os.path.join(save_dir, "best_for_f1.pth")
            th.save(lang_model.state_dict(), best_save_path)
        if val_accuracy_epoch > best_val_accuracy:
            best_val_accuracy = val_accuracy_epoch
            best_epoch_for_accuracy = epoch
            best_save_path = os.path.join(save_dir, "best_for_accuracy.pth")
            th.save(lang_model.state_dict(), best_save_path)
            
        # --- early stop ---
        
        if epoch - best_epoch_for_loss > (nepoch//5): # early stop if the model does not improve for 1/5 of the total epoch
            early_break_signal = True
        
        
        # ! eval testing 
        lang_model.eval()
        test_loss_epoch = 0
        nupdate = 0
        
        precision_dict = dict()
        recall_dict = dict()
        f1_dict = dict()
        accuracy_dict = dict()
        
        for data in (testpbar:=tqdm(test_dataloader, desc='Test')):
            metric_stats = process_and_forward_helper(data, lang_model, False, device) # we will not generate extra hard negatives for testing, also in lrm eval, we also use this function setup
            
                
            if "precision_lst" in metric_stats:
                precision_lst = metric_stats['precision_lst']
                accumulate_stats_dict_helper(precision_dict, precision_lst)
                
            if "recall_lst" in metric_stats:
                recall_lst = metric_stats['recall_lst']
                accumulate_stats_dict_helper(recall_dict, recall_lst)
                
            if "f1_lst" in metric_stats:
                f1_lst = metric_stats['f1_lst']
                accumulate_stats_dict_helper(f1_dict, f1_lst)
                
            if "accuracy_lst" in metric_stats:
                accuracy_lst = metric_stats['accuracy_lst']
                accumulate_stats_dict_helper(accuracy_dict, accuracy_lst)
                
                            
            loss_epoch, precision_epoch, recall_epoch, f1_epoch, accuracy_epoch, pbar_desc, local_nupdate = logging_performance_helper("Testing", metric_stats)
            
            test_loss_epoch += loss_epoch
            nupdate += local_nupdate
                
            testpbar.set_description(pbar_desc)
        
        test_loss_epoch /= nupdate
        test_precision_epoch = 0
        test_recall_epoch = 0
        test_f1_epoch = 0
        test_accuracy_epoch = 0
        
        
        for k in precision_dict:
            precision_dict[k] = precision_dict[k] / nupdate
            if precision_dict[k] > test_precision_epoch:
                test_precision_epoch = precision_dict[k]
        for k in recall_dict:
            recall_dict[k] = recall_dict[k] / nupdate
            if recall_dict[k] > test_recall_epoch:
                test_recall_epoch = recall_dict[k]
        for k in f1_dict:
            f1_dict[k] = f1_dict[k] / nupdate
            if f1_dict[k] > test_f1_epoch:
                test_f1_epoch = f1_dict[k]
        for k in accuracy_dict:
            accuracy_dict[k] = accuracy_dict[k] / nupdate
            if accuracy_dict[k] > test_accuracy_epoch:
                test_accuracy_epoch = accuracy_dict[k]
        
        if test_loss_epoch < best_test_loss:
            best_test_loss = test_loss_epoch
        if test_precision_epoch > best_test_precision:
            best_test_precision = test_precision_epoch
        if test_recall_epoch > best_test_recall:
            best_test_recall = test_recall_epoch
        if test_f1_epoch > best_test_f1:
            best_test_f1 = test_f1_epoch
        if test_accuracy_epoch > best_test_accuracy:
            best_test_accuracy = test_accuracy_epoch
            
        
        # ! save model
        if (epoch+1) % save_interval == 0:
            # save model
            temp_save_path = os.path.join(save_dir, f"checkpoint_{epoch+1}.pth")
            th.save(lang_model.state_dict(), temp_save_path)
            
        wandb_stats_dict = {
            "train_loss": train_loss_epoch,
            "val_loss": val_loss_epoch,
            "test_loss": test_loss_epoch,
            "best_train_loss": best_train_loss,
            "best_val_loss": best_val_loss,
            "best_test_loss": best_test_loss,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]['lr'],
        }
        # prec, recall, f1 and acc for train 
        if best_train_precision > 0:
            wandb_stats_dict['best_train_precision'] = best_train_precision
            wandb_stats_dict['train_precision'] = train_precision_epoch
        
        if best_train_recall > 0:
            wandb_stats_dict['best_train_recall'] = best_train_recall
            wandb_stats_dict['train_recall'] = train_recall_epoch
            
        if best_train_f1 > 0:
            wandb_stats_dict['best_train_f1'] = best_train_f1
            wandb_stats_dict['train_f1'] = train_f1_epoch
            
        if best_train_accuracy > 0:
            wandb_stats_dict['best_train_accuracy'] = best_train_accuracy
            wandb_stats_dict['train_accuracy'] = train_accuracy_epoch
            
            
        # prec, recall, f1 and acc for validate
        if best_val_precision > 0:
            wandb_stats_dict['best_val_precision'] = best_val_precision
            wandb_stats_dict['val_precision'] = val_precision_epoch
            wandb_stats_dict['best_epoch_for_precision'] = best_epoch_for_precision
        
        if best_val_recall > 0:
            wandb_stats_dict['best_val_recall'] = best_val_recall
            wandb_stats_dict['val_recall'] = val_recall_epoch
            
        if best_val_f1 > 0:
            wandb_stats_dict['best_val_f1'] = best_val_f1
            wandb_stats_dict['val_f1'] = val_f1_epoch
            wandb_stats_dict['best_epoch_for_f1'] = best_epoch_for_f1
            
            
        if best_val_accuracy > 0:
            wandb_stats_dict['best_val_accuracy'] = best_val_accuracy
            wandb_stats_dict['val_accuracy'] = val_accuracy_epoch
            wandb_stats_dict['best_epoch_for_accuracy'] = best_epoch_for_accuracy
            
        if len(cp_threshold_dict) > 0:
            wandb_stats_dict['cp_threshold'] = cp_threshold_dict
            
        # prec, recall, f1 and acc for test
        
        if len(precision_dict) > 0:
            wandb_stats_dict['test_precision'] = precision_dict
            wandb_stats_dict['best_test_precision'] = best_test_precision
            
        if len(recall_dict) > 0:
            wandb_stats_dict['test_recall'] = recall_dict
            wandb_stats_dict['best_test_recall'] = best_test_recall
            
        if len(f1_dict) > 0:
            wandb_stats_dict['test_f1'] = f1_dict
            wandb_stats_dict['best_test_f1'] = best_test_f1
            
        if len(accuracy_dict) > 0:
            wandb_stats_dict['test_accuracy'] = accuracy_dict
            wandb_stats_dict['best_test_accuracy'] = best_test_accuracy
        
        wandb.log(wandb_stats_dict)
            
        if early_break_signal:
            print(f"Early stopping at epoch {epoch}/{nepoch}")
            break
        
    # stop wandb 
    wandb.finish()
    
    return None
