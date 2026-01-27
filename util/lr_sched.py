import math

def adjust_learning_rate(optimizer, epoch, args):

    cur_it = epoch
    max_it = args.epochs
    
    # 预热步数
    wp_it = args.warmup_epochs 
    
    peak_lr = args.lr
    wp0 = getattr(args, 'wp0', 0.005)
    wpe = getattr(args, 'wpe', 0.01)
    sche_type = getattr(args, 'sche', 'lin0')
    
    # Weight Decay 参数
    wd = args.weight_decay
    wd_end = getattr(args, 'weight_decay_end', wd) 

    #计算 Current LR Ratio
    if cur_it < wp_it:
        # Warmup: [wp0, 1.0]
        cur_lr_ratio = wp0 + (1 - wp0) * cur_it / wp_it
    else:
        # Decay Phase
        pasd = (cur_it - wp_it) / (max_it - wp_it) 
        pasd = max(0.0, min(1.0, pasd))
        rest = 1 - pasd # [1, 0]

        if sche_type == 'cos':
            cur_lr_ratio = wpe + (1 - wpe) * (0.5 + 0.5 * math.cos(math.pi * pasd))
            
        elif sche_type == 'lin':
            T = 0.15
            max_rest = 1 - T
            if pasd < T: 
                cur_lr_ratio = 1.0
            else: 
                cur_lr_ratio = wpe + (1 - wpe) * rest / max_rest
                
        elif sche_type == 'lin0':
            T = 0.3
            max_rest = 1 - T
            if pasd < T: 
                cur_lr_ratio = 1.0
            else: 
                cur_lr_ratio = wpe + (1 - wpe) * rest / max_rest
                
        elif sche_type == 'lin00':
            cur_lr_ratio = wpe + (1 - wpe) * rest
            
        elif sche_type.startswith('lin'):
            try:
                T = float(sche_type[3:])
            except:
                T = 0.3 
                
            max_rest = 1 - T
            wpe_mid = wpe + (1 - wpe) * max_rest
            wpe_mid = (1 + wpe_mid) / 2
            
            if pasd < T: 
                cur_lr_ratio = 1 + (wpe_mid - 1) * pasd / T
            else: 
                cur_lr_ratio = wpe + (wpe_mid - wpe) * rest / max_rest
                
        elif sche_type == 'exp':
            T = 0.15
            max_rest = 1 - T
            if pasd < T: 
                cur_lr_ratio = 1.0
            else:
                expo = (pasd - T) / max_rest * math.log(wpe)
                cur_lr_ratio = math.exp(expo)
        else:
            T = 0.3
            max_rest = 1 - T
            if pasd < T: 
                cur_lr_ratio = 1.0
            else: 
                cur_lr_ratio = wpe + (1 - wpe) * rest / max_rest

    # 计算 Current Weight Decay (全程 Cosine)
    wd_progress = cur_it / max_it
    wd_progress = max(0.0, min(1.0, wd_progress))
    cur_wd = wd_end + (wd - wd_end) * (0.5 + 0.5 * math.cos(math.pi * wd_progress))

    
    current_lr_val = cur_lr_ratio * peak_lr
    
    for param_group in optimizer.param_groups:
        lr_sc = param_group.get('lr_sc', 1.0)
        wd_sc = param_group.get('wd_sc', 1.0) # VARSR 源码中使用了 wd_sc
        
        # 更新 LR
        param_group['lr'] = current_lr_val * lr_sc
        
        # 更新 Weight Decay
        if 'wd_sc' in param_group:
             param_group['weight_decay'] = cur_wd * wd_sc
        else:
            if param_group['weight_decay'] > 0:
                param_group['weight_decay'] = cur_wd

    return current_lr_val