Things left to implement:

- **[DONE]** Figure out how to report & save stats from runs. Should be saving
  at each epoch, and ideally in similar format to what the rest of RLLib uses.
  Probably use ray.tune.logger.CSVLogger & augment the dicts returned by RLLib.
  Add those things as Sacred artefacts, too.
- [**DONE**] Make sure that I'm recording my ~four key metrics in the logs
  appropriately: discriminator training throughput, TD3 update throughput, and
  TD3 sampling throughput.
- Switch from TD3 to APEX-TD3 during GAIL. Requires new TD3 config, probably
  copied from my git stash on svm.
- Train expert policies for Ant-v2 and Hopper-v2. That should be sufficient for
  the report, once combined with InvertedPendulum-v2 and HalfCheetah-v2.
- Move to distributed SGD for the reward function.
- Move to distributed SGD for the TD3 updates.
- Benchmark all methods with increasing levels of parallelism. Create some plots
  that I can put in slides.


Handy notes:

- I can use TimerStat() to record how long things take. A lot of RLLib is
  already instrumented like that; see `async_replay_optimizer.py` for an
  example.
