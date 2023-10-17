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
        logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
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
        logger.info('Unarchiving: "{}" to {}\n'.format(file_path, cached_file))
        with tarfile.open(file_path) as tar:
            tar.extractall(cached_file)

    return cached_file


@pytest.fixture
def sd15_model_path():
    return 'runwayml/stable-diffusion-v1-5'


@pytest.fixture
def sd_controlnet_canny_model_path():
    return 'lllyasviel/sd-controlnet-canny'


@pytest.fixture
def sd21_model_path():
    return 'runwayml/stable-diffusion-v2-1'
