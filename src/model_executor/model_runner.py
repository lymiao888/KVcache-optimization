from typing import List, Tuple

import torch
import torch.nn as nn

from src.model_executor.model_metadata import ModelConfig
from sampler.sampling_params import SamplingParams
from engine.master import Master
from scheduler.config import SchedulerConfig, CacheConfig
import time

class LLamaEngine():
    def __init__(
        self, 
        model_config: ModelConfig, 
        scheduler_config: SchedulerConfig = SchedulerConfig(),
        cache_config: CacheConfig = CacheConfig(),
    ) -> nn.Module:
        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_worker()

    def _init_worker(self):
        self.worker = Master(
            model_config=self.model_config,
            scheduler_config=self.scheduler_config,
            cache_config=self.cache_config,
            device=self.device,
        )
        self.worker.start_master()

    
    def generate(self,
                 system_prompt: str,
                 requests: List[Tuple[str, int, int]], 
                 sampling_params: SamplingParams = None):
        self.worker.add_requests(requests, sampling_params)

        start_time = time.time()
        self.worker.run()
        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Total time: {execution_time} seconds, total output tokens: {self.worker.total_out_tokens_len}")
        print(f"Throuputs: {self.worker.total_out_tokens_len/execution_time} tokens/s")
