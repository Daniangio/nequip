import wandb

from .trainer_delta import TrainerDelta


class TrainerDeltaWandB(TrainerDelta):
    """Trainer class that adds WandB features"""

    def end_of_epoch_log(self):
        TrainerDelta.end_of_epoch_log(self)
        wandb.log(self.mae_dict)

    def init(self):
        super().init()

        if not self._initialized:
            return

        # upload some new fields to wandb
        wandb.config.update({"num_weights": self.num_weights})

        if self.kwargs.get("wandb_watch", False):
            wandb_watch_kwargs = self.kwargs.get("wandb_watch_kwargs", {})
            wandb.watch(self.model, **wandb_watch_kwargs)
