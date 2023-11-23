import pytest

import logging
import os
import tarfile
from urllib.parse import urlparse
from torch.hub import download_url_to_file, get_dir

logger = logging.getLogger()


def get_cache_path_by_url(url):
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, 'checkpoints')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    return cached_file


def download_model(url, model_md5: str = None):
    cached_file = get_cache_path_by_url(url)
    if not os.path.exists(cached_file):
        logger.info("Downloading: '{}' to {}\n".format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)

    return cached_file


def get_model_path(url_or_path):
    if os.path.exists(url_or_path):
        model_path = url_or_path
    else:
        model_path = download_model(url_or_path)

    return model_path


def get_cache_path_for_archiver(file_path):
    parts = urlparse(file_path)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, 'checkpoints')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    filename = os.path.basename(parts.path) + '.unarchived'
    cached_file = os.path.join(model_dir, filename)
    return cached_file


def unarchive_model(file_path):
    cached_file = get_cache_path_for_archiver(file_path)
    if not os.path.exists(cached_file):
        logger.info("Unarchiving: '{}' to {}\n".format(file_path, cached_file))
        with tarfile.open(file_path) as tar:
            tar.extractall(cached_file)

    return cached_file


def path_exists_or(file_path, other):
    if os.path.exists(file_path):
        return file_path
    if isinstance(other, str):
        return other
    return other()


@pytest.fixture
def sd15_model_path():
    return path_exists_or('../stable-diffusion-v1-5',
                          'runwayml/stable-diffusion-v1-5')


@pytest.fixture
def sd_controlnet_canny_model_path():
    return path_exists_or('../sd-controlnet-canny',
                          'lllyasviel/sd-controlnet-canny')


@pytest.fixture
def sd21_model_path():
    return path_exists_or('../stable-diffusion-2-1',
                          'stabilityai/stable-diffusion-2-1')


@pytest.fixture
def sdxl_model_path():
    return path_exists_or('../stable-diffusion-xl-base-1.0',
                          'stabilityai/stable-diffusion-xl-base-1.0')


@pytest.fixture
def sd15_lora_t4_path():
    return path_exists_or('../sd-model-finetuned-lora-t4',
                          'sayakpaul/sd-model-finetuned-lora-t4')


@pytest.fixture
def sd15_lora_dog_path():
    return path_exists_or('../sd-model-finetuned-lora-dog',
                          'sayakpaul/new-lora-check-v15')


@pytest.fixture
def diffusers_dog_example_path():

    def download():
        from huggingface_hub import snapshot_download

        return snapshot_download(
            'diffusers/dog-example',
            repo_type='dataset',
            ignore_patterns=['*.gitignore', '*.gitattributes', '*.DS_Store'],
        )

    return path_exists_or('../dog-example', download)


@pytest.fixture
def llama_2_7b_hf_path():
    return path_exists_or('../Llama-2-7b-hf',
                          'meta-llama/Llama-2-7b-hf')
