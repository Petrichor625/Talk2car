# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import json
import shutil
import logging
import tempfile
from io import open
from tqdm import tqdm
from pathlib import Path
from hashlib import sha256
from functools import wraps
from urllib.parse import urlparse

import boto3
import requests
from botocore.exceptions import ClientError

import torch
from torch import nn

# getenv从环境变量中获取PYTOCH_PRETRAINED_BERT_CACHE的值，若不存在这从用户主目录下的pytorch_pretrained_bert获取
PYTORCH_PRETRAINED_BERT_CACHE = Path(
    os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", Path.home() / ".pytorch_pretrained_bert")
)

# 权重名 pytorch_model.bin
WEIGHTS_NAME = "pytorch_model.bin"

TF_WEIGHTS_NAME = 'model.ckpt'

logger = logging.getLogger(__name__)

'''
这段代码定义了一个函数 url_to_filename,它的目的是将给定的 URL网址和 ETag(用于标识文件版本的标签)转换成一个文件名，以便在本地存储文件时使用。
主要用于缓存远程资源，例如从网上下载的文件，以避免重复下载相同的文件。
'''
def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode("utf-8")
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    return filename


'''
定义了一个函数 filename_to_url,用于从缓存目录中的文件名中提取 URL 和 ETag 信息。这在需要检索缓存的远程资源时非常有用，以便确定文件的来源和版本。
'''
def filename_to_url(filename, cache_dir=None):
    """
    Return the url and etag (which may be ``None``) stored for `filename`.
    Raise ``EnvironmentError`` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise EnvironmentError("file {} not found".format(cache_path))

    meta_path = cache_path + ".json"
    if not os.path.exists(meta_path):
        raise EnvironmentError("file {} not found".format(meta_path))

    with open(meta_path, encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
    url = metadata["url"]
    etag = metadata["etag"]

    return url, etag

'''
这段代码定义了一个函数 cached_path,用于根据给定的 URL 或本地文件路径，从缓存中获取文件或下载文件并进行缓存，然后返回缓存的文件路径。
'''
def cached_path(url_or_filename, cache_dir=None):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if sys.version_info[0] == 3 and isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https", "s3"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))


'''split_s3_path:用于将完整的 Amazon S3 路径分割成存储桶名称和路径。'''
def split_s3_path(url):
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad s3 path {}".format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    # Remove '/' at beginning of path.
    if s3_path.startswith("/"):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


def s3_request(func):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                raise EnvironmentError("file {} not found".format(url))
            else:
                raise

    return wrapper



@s3_request
def s3_etag(url):
    """Check ETag on S3 object."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag


@s3_request
def s3_get(url, temp_file):
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)


def http_get(url, temp_file):
    req = requests.get(url, stream=True)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()

'''这段代码定义了一个函数 get_from_cache,用于从缓存中获取文件或下载文件,并将其存储在缓存目录中。'''
def get_from_cache(url, cache_dir=None):
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    这段代码的作用是在给定一个 URL 的情况下，从本地缓存中查找相应的数据集文件。如果数据集文件不在缓存中，它会下载该文件并将其缓存起来，然后返回缓存文件的路径。
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Get eTag to add to filename, if it exists.
    if url.startswith("s3://"):
        etag = s3_etag(url)
    else:
        response = requests.head(url, allow_redirects=True)
        if response.status_code != 200:
            raise IOError("HEAD request failed for url {} with status code {}".format(url, response.status_code))
        etag = response.headers.get("ETag")

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    if not os.path.exists(cache_path):
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info("%s not found in cache, downloading to %s", url, temp_file.name)

            # GET file object
            if url.startswith("s3://"):
                s3_get(url, temp_file)
            else:
                http_get(url, temp_file)

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)

            logger.info("copying %s to cache at %s", temp_file.name, cache_path)
            with open(cache_path, "wb") as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            logger.info("creating metadata file for %s", cache_path)
            meta = {"url": url, "etag": etag}
            meta_path = cache_path + ".json"
            with open(meta_path, "w", encoding="utf-8") as meta_file:
                json.dump(meta, meta_file)

            logger.info("removing temp file %s", temp_file.name)

    return cache_path


def read_set_from_file(filename):
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, "r", encoding="utf-8") as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection


def get_file_extension(path, dot=True, lower=True):
    ext = os.path.splitext(path)[1]
    ext = ext if dot else ext[1:]
    return ext.lower() if lower else ext


class PreTrainedModel(nn.Module):
    r""" Base class for all models.
        :class:`~pytorch_transformers.PreTrainedModel` takes care of storing the configuration of the models and handles methods for loading/downloading/saving models
        as well as a few methods commons to all models to (i) resize the input embeddings and (ii) prune heads in the self-attention heads.
        Class attributes (overridden by derived classes):
            - ``config_class``: a class derived from :class:`~pytorch_transformers.PretrainedConfig` to use as configuration class for this model architecture.
            - ``pretrained_model_archive_map``: a python ``dict`` of with `short-cut-names` (string) as keys and `url` (string) of associated pretrained weights as values.
            - ``load_tf_weights``: a python ``method`` for loading a TensorFlow checkpoint in a PyTorch model, taking as arguments:
                - ``model``: an instance of the relevant subclass of :class:`~pytorch_transformers.PreTrainedModel`,
                - ``config``: an instance of the relevant subclass of :class:`~pytorch_transformers.PretrainedConfig`,
                - ``path``: a path (string) to the TensorFlow checkpoint.
            - ``base_model_prefix``: a string indicating the attribute associated to the base model in derived classes of the same architecture adding modules on top of the base model.
        
        :class:~pytorch_transformers.PreTrainedModel 是一个基类，负责存储模型的配置信息，并处理加载/下载/保存模型的方法。此外，它还包含一些对所有模型通用的方法，用于（i）调整输入嵌入和（ii）修剪自注意力头。
    
        `config_class``:指定模型所使用用的架构类型（预训练模型类型）
        pretrained_model_archive_map:一个字典,将模型的简短名称(字符串)映射到下载相关预训练权重的URL。
        load_tf_weights`:用于加载TSensorFlow的checkpoint加载Pytorch模型中去:
            接受TensotFlow_checkpoint_Path,模型实例(pytorch_transformers.PreTrainedModel),和config(pytorch_transformers.PretrainedConfig)作为参数
        base_model_prefix:
    """

    config_class = None
    pretrained_model_archive_map = {}
    load_tf_weights = lambda model, config, path: None
    base_model_prefix = ""

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        self.config = config

    # 该方法用于从提供的标记嵌入模块构建一个调整大小的嵌入模块，可用于调整嵌入层的大小。
    # 添加EmbedingModule的维度将会在模型的尾部添加新的向量
    # 减少模型的维度将会直接在Embeding尾部删去向量
    
    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None):
        """ Build a resized Embedding Module from a provided token Embedding Module.

            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end

            old_embeddings:导入的是旧的加载embeddings模块
            return返回一个新的torch.nn.Embeddings类型的模块


        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the provided token Embedding Module.

        Return: ``torch.nn.Embeddings``
            Pointer to the resized Embedding Module or the old Embedding Module if new_num_tokens is None
        """
        if new_num_tokens is None:
            return old_embeddings

        # 加载old_tokens和old_embedding的维度
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device)

        # initialize all new embeddings (in particular added tokens)
        self.init_weights(new_embeddings)

        # Copy word embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]

        return new_embeddings

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        # TODO: ignore torch scripts here
        first_module.weight = second_module.weight

    def resize_token_embeddings(self, new_num_tokens=None):
        """ Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
        Take care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.
        Arguments:
            new_num_tokens: (`optional`) int:
                New number of tokens in the embedding matrix. Increasing the size will add newly initialized vectors at the end. Reducing the size will remove vectors from the end. 
                If not provided or None: does nothing and just returns a pointer to the input tokens ``torch.nn.Embeddings`` Module of the model.
        Return: ``torch.nn.Embeddings``
            Pointer to the input tokens Embeddings Module of the model
        """
        base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed
        model_embeds = base_model._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens
        base_model.vocab_size = new_num_tokens

        # Tie weights again if needed
        if hasattr(self, "tie_weights"):
            self.tie_weights()

        return model_embeds

    def prune_heads(self, heads_to_prune):
        """ Prunes heads of the base model.
            Arguments:
                heads_to_prune: dict with keys being selected layer indices (`int`) and associated values being the list of heads to prune in said layer (list of `int`).
        """
        base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed
        base_model._prune_heads(heads_to_prune)

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~pytorch_transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""Instantiate a pretrained pytorch model from a pre-trained model configuration.
        The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with ``model.train()``
        The warning ``Weights from XXX not initialized from pretrained model`` means that the weights of XXX do not come pre-trained with the rest of the model.
        It is up to you to train those weights with a downstream fine-tuning task.
        The warning ``Weights from XXX not used in YYY`` means that the layer XXX is not used by YYY, therefore those weights are discarded.
        Parameters:
            


            pretrained_model_name_or_path: either:
                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                
                预训练模型的快捷名称（如 "bert-base-uncased"），用于从缓存或下载加载模型。
                包含模型权重的目录路径（例如 "./my_model_directory/"），使用 PreTrainedModel.save_pretrained 保存的权重。
                TensorFlow 检查点文件的路径或 URL(例如 ./tf_model/model.ckpt.index)。在这种情况下，应将 from_tf 设置为 True，并提供配置对象作为 config 参数。通过这种方式加载模型速度较慢，因为它涉及将 TensorFlow 检查点转换为 PyTorch 模型。
                
                
            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method
            model_args:(可选)位置参数序列。所有其他位置参数将传递给底层模型的 __init__ 方法。
            
                  
            config: (`optional`) instance of a class derived from :class:`~pytorch_transformers.PretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:
                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.
            config:(可选)来自 PretrainedConfig 派生类的实例。可以用来覆盖自动加载的模型配置。自动加载配置的情况包括：
            模型是库中提供的模型（使用预训练模型的 "shortcut-name" 字符串加载）。
            模型是使用 PreTrainedModel.save_pretrained 保存并通过提供保存目录重新加载的。
            模型是通过提供本地目录作为 pretrained_model_name_or_path 加载的，且该目录中包含名为 config.json 的配置 JSON 文件。
            
            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~pytorch_transformers.PreTrainedModel.save_pretrained` and :func:`~pytorch_transformers.PreTrainedModel.from_pretrained` is not a simpler option.
                (可选）字典，用于模型使用的可选状态字典，而不是从保存的权重文件加载状态字典。
                如果要从预训练配置创建模型但加载自己的权重，则可以使用此选项。不过，您应该检查是否使用 PreTrainedModel.save_pretrained 和
                PreTrainedModel.from_pretrained 是更简单的选项。
                
            
            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.
            cache_dir:(可选)字符串，指定下载的预训练模型配置应该缓存在其中的目录路径，如果不希望使用标准缓存。
                
        
            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.
            output_loading_info:(可选)布尔值，设置为 True 以返回包含缺失键、意外键和错误消息的字典。

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:
                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~pytorch_transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.
        
        Examples:
            model = BertModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = BertModel.from_pretrained('./test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = BertModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
            model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)
        """

        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        from_hf = kwargs.pop("from_hf", False)
        output_loading_info = kwargs.pop("output_loading_info", False)
        default_gpu = kwargs.pop("default_gpu", True)

        # Load config
        assert config is not None
        model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
            archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            if from_tf:
                # Directly load from a TensorFlow checkpoint
                archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
            else:
                archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        else:
            if from_tf:
                # Directly load from a TensorFlow checkpoint
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                logger.error("Couldn't reach server at '{}' to download pretrained weights.".format(archive_file))
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name_or_path, ", ".join(cls.pretrained_model_archive_map.keys()), archive_file)
                )
            return None
        if default_gpu:
            if resolved_archive_file == archive_file:
                logger.info("loading weights file {}".format(archive_file))
            else:
                logger.info("loading weights file {} from cache at {}".format(archive_file, resolved_archive_file))

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location="cpu")
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            return cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # Rename Bert parameters for our framework
        # NB: Assume 1 Bert layer is mapped to 1 layer only (cannot be used to init multiple layers)
        old_keys = []
        new_keys = []
        nums = []
        for key in state_dict.keys():
            new_key = None
            if ".layer." in key and from_hf:
                num = int(key.split(".layer.")[-1].split(".")[0])
                if ".attention." in key:
                    new_key = key.replace(".layer.%d.attention." % num,
                                          ".layer.%d.attention_" % config.bert_layer2attn_sublayer.get(str(num), num))
                elif ".intermediate." in key:
                    new_key = key.replace(".layer.%d.intermediate." % num,
                                          ".layer.%d.intermediate." % config.bert_layer2ff_sublayer.get(str(num), num))
                elif ".output." in key:
                    new_key = key.replace(".layer.%d.output." % num,
                                          ".layer.%d.output." % config.bert_layer2ff_sublayer.get(str(num), num))
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
                nums.append(num)
        for old_key, new_key, _ in sorted(zip(old_keys, new_keys, nums), key=lambda x: x[2], reverse=True):
            state_dict[new_key] = state_dict.pop(old_key)

        # Load from a PyTorch state_dict
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        if not hasattr(model, cls.base_model_prefix) and any(
            s.startswith(cls.base_model_prefix) for s in state_dict.keys()
        ):
            start_prefix = cls.base_model_prefix + "."
        if hasattr(model, cls.base_model_prefix) and not any(
            s.startswith(cls.base_model_prefix) for s in state_dict.keys()
        ):
            model_to_load = getattr(model, cls.base_model_prefix)

        logger.info(start_prefix)
        load(model_to_load, prefix=start_prefix)
        if len(missing_keys) > 0 and default_gpu:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".format(model.__class__.__name__, missing_keys)
            )
        if len(unexpected_keys) > 0 and default_gpu:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(model.__class__.__name__, unexpected_keys)
            )
        if len(error_msgs) > 0 and default_gpu:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(model.__class__.__name__, "\n\t".join(error_msgs))
            )

        if hasattr(model, "tie_weights"):
            model.tie_weights()  # make sure word embedding weights are still tied

        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info

        return model




 
def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Adapted from https://github.com/allenai/allennlp
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


# 这段代码定义了一个名为 masked_softmax 的函数，用于计算在某些元素需要被屏蔽（mask or标记为None）的情况下的 softmax 操作
def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    Adapted from https://github.com/allenai/allennlp

    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """

    '''
        vector：一个包含要进行 softmax 操作的张量。
        mask：一个与 vector 具有相同形状的张量，用于指示哪些元素需要被屏蔽。可以是一个二进制掩码，其中为 1 的元素表示要保留，为 0 的元素表示要屏蔽。
        dim：指定 softmax 操作应该在哪个维度上执行的参数，默认为 -1，表示在最后一个维度上执行 softmax 操作。
        memory_efficient：一个布尔值，用于控制是否采用内存更高效的方式进行 softmax 计算。如果设置为 True，则会将被屏蔽的位置的概率设置为一个极大的负数，以使这些位置的概率接近于 0。如果设置为 False，则会将被屏蔽的位置的概率置为 0。
        mask_fill_value：当 memory_efficient 为 True 时，用于填充被屏蔽位置的值，默认为 -1e32。
    '''
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result
