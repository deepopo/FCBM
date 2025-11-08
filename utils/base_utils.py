import os
import importlib
import pytorch_lightning.callbacks as plc

def instantiate_from_config(config):
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def examine_dir(dir):
    if os.path.exists(dir):
        pass
    else:
        os.mkdir(dir)
        print(f'make directory: {dir}')

class CustomEarlyStopping(plc.EarlyStopping):
    def __init__(self, start_epoch=0,**kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch < self.start_epoch:
            return
        super().on_validation_end(trainer, pl_module)

class CustomProgressBar(plc.TQDMProgressBar):
    def get_metrics(self, *args, **kwargs):
        metrics = super().get_metrics(*args, **kwargs)
        return {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()}

class DualConditionEarlyStopping(plc.Callback):
    def __init__(self, loss_patience=5, concept_patience=5, act_thred=30):
        super().__init__()
        '''
            1. if val_avg_act_concepts < act_thred, then use conditional_patience and record the minimum val_loss_y
            2. if val_avg_act_concepts >= act_thred, then use global_patience and record the minimum val_avg_act_concepts
        '''
        self.act_thred = act_thred
        self.loss_patience = loss_patience
        self.best_loss = float('inf')
        self.loss_wait = 0
        
        self.concept_patience = concept_patience
        self.best_concept = float('inf')
        self.concept_wait = 0
        
        self.any_qualified = False
        self.final_best = {
            'type': None,  # 'loss' 或 'concept'
            'value': float('inf'),
            'epoch': 0
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        avg_act = trainer.callback_metrics.get('val_avg_act_concepts')
        loss_y = trainer.callback_metrics.get('val_loss_y')
        
        if avg_act is None or loss_y is None:
            return

        if avg_act < self.best_concept:
            self.best_concept = avg_act
            self.concept_wait = 0
            if not self.any_qualified:
                self._update_final_best('concept', avg_act, current_epoch)
        else:
            self.concept_wait += 1

        if avg_act < self.act_thred:
            self.any_qualified = True
            
            if loss_y < self.best_loss:
                self.best_loss = loss_y
                self.loss_wait = 0
                self._update_final_best('loss', loss_y, current_epoch)
            else:
                self.loss_wait += 1

        stop = False
        if self.any_qualified:
            if self.loss_wait >= self.loss_patience:
                stop = True
        else:
            if self.concept_wait >= self.concept_patience:
                stop = True

        if stop:
            trainer.should_stop = True

    def _update_final_best(self, best_type, value, epoch):
        if (best_type == 'loss' and value < self.final_best['value']) or \
           (best_type == 'concept' and value < self.final_best['value']):
            self.final_best.update({
                'type': best_type,
                'value': value,
                'epoch': epoch
            })

class DynamicMonitorCheckpoint(plc.ModelCheckpoint):
    def __init__(self, early_stop_callback, **kwargs):
        super().__init__(**kwargs)
        self.early_stop_callback = early_stop_callback

    def _get_monitor_candidate(self, trainer):
        if self.early_stop_callback.any_qualified:
            return 'val_loss_y'
        return 'val_avg_act_concepts'

    def on_validation_end(self, trainer, pl_module):
        self.monitor = self._get_monitor_candidate(trainer)
        
        super().on_validation_end(trainer, pl_module)

def load_callbacks(monitor='val_loss', patience=100, mode='min'):
    callbacks = [CustomProgressBar()]

    callbacks.append(CustomEarlyStopping(
        monitor=monitor,
        patience=patience,
        mode=mode,
        start_epoch=1
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor=monitor,
        filename='best-{epoch:02d}',
        save_top_k=1,
        mode=mode,
        save_last=True
    ))

    callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks

def load_hyper_callbacks(monitor='val_loss', patience=100, mode='min', act_thred=30):
    callbacks = [CustomProgressBar()]

    early_stop = DualConditionEarlyStopping(
            loss_patience=patience,
            concept_patience=patience, 
            act_thred = act_thred
        )
    callbacks.append(early_stop)

    callbacks.append(DynamicMonitorCheckpoint(
            early_stop_callback=early_stop,
            filename='best-{epoch:02d}',
            mode=mode,
            save_top_k=1, 
            save_last=True
        ))

    callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks