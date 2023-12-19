import logging
from .diffusion_pipeline_compiler import *

logger = logging.getLogger()

logger.warning(
    '`sfast.compilers.stable_diffusion_pipeline_compiler` is deprecated. Please use `sfast.compilers.diffusion_pipeline_compiler` instead.'
)
