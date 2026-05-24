# # dataset/aoss_imagenet.py
# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# import boto3
# import botocore
# from botocore.config import Config
# import io
# import sys
# import os
#
#
# class AOSSImageNetDataset(Dataset):
#     def __init__(self, bucket: str, transform=None, filelist_path='./imagenet_train_filelist.txt'):
#         """
#         AOSS ImageNet Dataset
#         """
#         # 清除代理设置
#         os.environ['HTTP_PROXY'] = ''
#         os.environ['HTTPS_PROXY'] = ''
#         os.environ['NO_PROXY'] = '*'
#
#         # 1. 读取文件列表
#         with open(filelist_path) as f:
#             self.filelist = f.read().splitlines()
#
#         self.transform = transform
#         self.bucket = bucket
#
#         # -----------------------------------------------------------
#         # [新增逻辑] 解析类别并生成标签
#         # -----------------------------------------------------------
#         print("正在解析 ImageNet 类别标签...")
#
#         # 假设 filelist 中的路径格式为: "n01440764/n01440764_10026.JPEG"
#         # 我们提取父文件夹名称作为类别标识 (WordNet ID)
#         self.samples = []  # 存储 (path, target_idx)
#
#         # 第一步：提取所有出现的类别名称
#         class_names = set()
#         for path in self.filelist:
#             # os.path.dirname 获取路径目录，split('/')[-1] 确保拿到最后一级目录名
#             # 例如: 'train/n01440764/img.jpg' -> 'n01440764'
#             dirname = os.path.dirname(path)
#             if dirname:
#                 cls_name = dirname.split('/')[-1]
#                 class_names.add(cls_name)
#             else:
#                 # 如果路径没有目录（极其少见），需要根据实际文件名格式处理
#                 # 这里做个简单兼容，假设文件名以类别开头
#                 class_names.add(path.split('_')[0])
#
#         # 第二步：排序类别以保证索引一致性 (0-999)
#         self.classes = sorted(list(class_names))
#         self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
#
#         # 第三步：生成每个样本对应的标签索引 (预处理以加快 __getitem__)
#         self.targets = []
#         for path in self.filelist:
#             dirname = os.path.dirname(path)
#             if dirname:
#                 cls_name = dirname.split('/')[-1]
#             else:
#                 cls_name = path.split('_')[0]
#
#             # 获取 int 类型的标签 ID
#             label_id = self.class_to_idx.get(cls_name, 0)
#             self.targets.append(label_id)
#
#         print(f"解析完成: 包含 {len(self.classes)} 个类别, {len(self.targets)} 张图片。")
#         # -----------------------------------------------------------
#
#         # AOSS配置
#         self.aws_access_key_id = '0198A1B9771F7BAAA9A55AC5B51ACC2F'
#         self.aws_secret_access_key = '0198A1B9771F7B9D998F202B044BE13C'
#         self.endpoint_url = 'http://aoss-internal.cn-sh-01b.sensecoreapi-oss.cn'
#
#         self.s3_client = None
#         self.config = Config(
#             connect_timeout=300,
#             read_timeout=300,
#             retries={'max_attempts': 20, 'mode': 'adaptive'}
#         )
#
#     def _ensure_s3_client(self):
#         """延迟初始化S3客户端"""
#         if self.s3_client is None:
#             try:
#                 self.s3_client = boto3.client(
#                     's3',
#                     endpoint_url=self.endpoint_url,
#                     aws_access_key_id=self.aws_access_key_id,
#                     aws_secret_access_key=self.aws_secret_access_key,
#                     config=self.config
#                 )
#             except Exception as e:
#                 print(f"❌ 连接AOSS失败: {e}")
#                 sys.exit(1)
#
#     def load_image_from_aoss(self, object_key):
#         """从AOSS加载图像"""
#         self._ensure_s3_client()
#
#         max_retries = 3
#         for attempt in range(max_retries):
#             try:
#                 response = self.s3_client.get_object(Bucket=self.bucket, Key=object_key)
#                 img_bytes = response['Body'].read()
#                 img_buffer = io.BytesIO(img_bytes)
#                 img = Image.open(img_buffer).convert('RGB')
#                 return img
#             except botocore.exceptions.ReadTimeoutError as e:
#                 if attempt < max_retries - 1:
#                     continue
#                 print(f"❌ 读取超时 ({object_key}): {e}")
#                 return None
#             except botocore.exceptions.ClientError as e:
#                 if e.response['Error']['Code'] == 'NoSuchKey':
#                     print(f"⚠️ 文件不存在: {object_key}")
#                 else:
#                     print(f"❌ 客户端错误 ({object_key}): {e}")
#                 return None
#             except Exception as e:
#                 print(f"❌ 加载图像出错 ({object_key}): {e}")
#                 return None
#
#         return None
#
#     def __len__(self):
#         return len(self.filelist)
#
#     def __getitem__(self, index: int):
#         img_path = self.filelist[index]
#
#         # 1. 获取对应的标签 (int)
#         target = self.targets[index]
#
#         # 2. 加载图片
#         img = self.load_image_from_aoss(img_path)
#
#         if img is None:
#             # 如果加载失败，返回黑色图像
#             print(f"⚠️ 使用占位图像替代: {img_path}")
#             img = Image.new('RGB', (256, 256), (0, 0, 0))
#
#         if self.transform:
#             img = self.transform(img)
#
#         # 3. 返回 图片 和 真实的标签ID
#         return img, target
#
#
# def build_aoss_imagenet(args, transform):
#     """构建AOSS ImageNet数据集"""
#     return AOSSImageNetDataset(
#         filelist_path=args.data_path,
#         bucket=args.aoss_bucket,
#         transform=transform
#     )



# dataset/aoss_imagenet.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import boto3
import botocore
from botocore.config import Config
import io
import sys
import os
import json
import glob
from pathlib import Path
from datetime import datetime


class AOSSImageNetDataset(Dataset):
    def __init__(self, bucket: str, transform=None, filelist_path='./imagenet_train_filelist.txt'):
        """
        AOSS ImageNet Dataset

        Args:
            filelist_path: 文件列表路径，每行一个对象key
            bucket: AOSS bucket名称
            transform: torchvision transforms
        """
        # 清除代理设置
        os.environ['HTTP_PROXY'] = ''
        os.environ['HTTPS_PROXY'] = ''
        os.environ['NO_PROXY'] = '*'

        # 读取文件列表
        with open(filelist_path) as f:
            self.filelist = f.read().splitlines()

        self.filelist_path = filelist_path
        self.transform = transform
        self.bucket = bucket

        # AOSS配置
        # self.aws_access_key_id = '0198A1B9771F7BAAA9A55AC5B51ACC2F'
        self.aws_access_key_id = '01997084CBF777519D5F10EC029154C6'
        # self.aws_secret_access_key = '0198A1B9771F7B9D998F202B044BE13C'
        self.aws_secret_access_key = '01997084CBF7774488E90D66F4FCB83D'
        # self.endpoint_url = 'http://aoss-internal.cn-sh-01b.sensecoreapi-oss.cn' # 内网地址
        self.endpoint_url = 'http://aoss.cn-sh-01b.sensecoreapi-oss.cn' # 外网地址

        self.s3_client = None
        self.config = Config(
            connect_timeout=300,
            read_timeout=300,
            retries={'max_attempts': 20, 'mode': 'adaptive'}
        )
        self.dump_count = int(os.environ.get('AOSS_DUMP_COUNT', '0'))
        self.dump_dir = os.environ.get('AOSS_DUMP_DIR', '')

    def _ensure_s3_client(self):
        """延迟初始化S3客户端"""
        if self.s3_client is None:
            try:
                self.s3_client = boto3.client(
                    's3',
                    endpoint_url=self.endpoint_url,
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    config=self.config
                )
            except Exception as e:
                print(f"❌ 连接AOSS失败: {e}")
                sys.exit(1)

    def load_image_from_aoss(self, object_key):
        """从AOSS加载图像"""
        self._ensure_s3_client()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.s3_client.get_object(Bucket=self.bucket, Key=object_key)
                img_bytes = response['Body'].read()
                img_buffer = io.BytesIO(img_bytes)
                img = Image.open(img_buffer).convert('RGB')
                return img, None
            except botocore.exceptions.ReadTimeoutError as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ 读取超时，重试 {attempt + 1}/{max_retries}: {object_key}")
                    continue
                print(f"❌ 读取超时 ({object_key}): {e}")
                return None, {
                    'object_key': object_key,
                    'reason': 'ReadTimeoutError',
                    'error': str(e)
                }
            except botocore.exceptions.ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'NoSuchKey':
                    print(f"⚠️ 文件不存在: {object_key}")
                else:
                    print(f"❌ 客户端错误 ({object_key}): {e}")
                return None, {
                    'object_key': object_key,
                    'reason': error_code,
                    'error': str(e)
                }
            except Exception as e:
                print(f"❌ 加载图像出错 ({object_key}): {e}")
                return None, {
                    'object_key': object_key,
                    'reason': type(e).__name__,
                    'error': str(e)
                }

        return None, {
            'object_key': object_key,
            'reason': 'UnknownError',
            'error': 'Failed after retries'
        }

    def _get_failure_log_path(self):
        failure_log_path = os.environ.get('AOSS_FAILURE_LOG_PATH')
        if not failure_log_path:
            return None
        return f"{failure_log_path}.{os.getpid()}"

    def _get_dump_index_path(self):
        if not self.dump_dir:
            return None
        return os.path.join(self.dump_dir, 'dumped_samples_index.jsonl')

    def _append_failure_record(self, failure_record):
        failure_log_path = self._get_failure_log_path()
        if not failure_log_path:
            return
        with open(failure_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(failure_record, ensure_ascii=False) + '\n')

    def _should_dump_sample(self, index):
        return self.dump_count > 0 and index < self.dump_count and bool(self.dump_dir)

    def _dump_sample_image(self, img, index, object_key):
        if not self._should_dump_sample(index):
            return
        Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
        suffix = Path(object_key).suffix or '.png'
        output_name = f'{index:06d}{suffix}'
        output_path = os.path.join(self.dump_dir, output_name)
        img.save(output_path)
        dump_index_path = self._get_dump_index_path()
        if dump_index_path:
            with open(dump_index_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'index': index,
                    'object_key': object_key,
                    'saved_path': os.path.abspath(output_path)
                }, ensure_ascii=False) + '\n')

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index: int):
        img_path = self.filelist[index]
        img, failure_info = self.load_image_from_aoss(img_path)

        if img is None:
            print(f"⚠️ 使用占位图像替代: {img_path}")
            failure_record = {
                'index': index,
                'object_key': img_path,
                'used_placeholder': True,
            }
            if failure_info is not None:
                failure_record.update(failure_info)
            self._append_failure_record(failure_record)
            img = Image.new('RGB', (256, 256), (0, 0, 0))
        else:
            self._dump_sample_image(img, index, img_path)

        if self.transform:
            img = self.transform(img)

        # 返回图像和dummy label（VQ训练不需要label）
        return img, 0


    def write_failure_summary(self, output_path):
        failed_samples = []
        failure_log_path = os.environ.get('AOSS_FAILURE_LOG_PATH')
        if failure_log_path:
            for log_path in glob.glob(f"{failure_log_path}.*"):
                with open(log_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            failed_samples.append(json.loads(line))

        dumped_samples = []
        dump_index_path = self._get_dump_index_path()
        if dump_index_path and os.path.exists(dump_index_path):
            with open(dump_index_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        dumped_samples.append(json.loads(line))

        summary = {
            'bucket': self.bucket,
            'filelist_path': os.path.abspath(self.filelist_path),
            'total_samples': len(self.filelist),
            'placeholder_count': len(failed_samples),
            'has_placeholder': len(failed_samples) > 0,
            'dump_count': self.dump_count,
            'dump_dir': os.path.abspath(self.dump_dir) if self.dump_dir else '',
            'dumped_samples': dumped_samples,
            'message': '没有黑图，占位图像未被使用' if len(failed_samples) == 0 else '存在黑图，占位图像已用于部分样本',
            'failed_samples': failed_samples,
            'generated_at': datetime.now().isoformat()
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


def build_aoss_imagenet(args, transform):
    """构建AOSS ImageNet数据集"""
    return AOSSImageNetDataset(
        filelist_path=args.data_path,  # 这里data_path是filelist路径
        bucket=args.aoss_bucket,
        transform=transform
    )