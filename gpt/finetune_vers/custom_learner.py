from sample_factory.algo.learning.learner import Learner
from sample_factory.utils.utils import log


class FreezeActorLearner(Learner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._actor_freeze = True
        self._unfreeze_steps = 4000

        if self._actor_freeze:
            # self.actor_critic.freeze_actor()
            log.info(f"Freeze actor learner activate. Unfreeze = {self._unfreeze_steps} steps")

    def _train(self, *args, **kwargs):
        if self._actor_freeze and self.train_step > self._unfreeze_steps:
            self.actor_critic.unfreeze_actor()
            self._actor_freeze = False
            log.info(f"Unfreeze actor (train steps = {self.train_step})")  
        return super()._train(*args, **kwargs)


def registry_changed_learner():
    import sample_factory.algo.learning.learner as l
    l.Learner = FreezeActorLearner
    log.info("✅ CustomLearner registered")


registry_changed_learner()
