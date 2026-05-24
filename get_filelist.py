import boto3
import botocore
from botocore.config import Config
from tqdm.auto import tqdm
import os
import time
import sys

# 临时绕过代理（如果环境有HTTP_PROXY，内部端点可能不需要）
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['NO_PROXY'] = '*'  # 绕过所有代理

# 定义图片扩展名
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPEG', '.JPG', '.PNG'}

# 配置参数
bucket = "imagenet"
prefix = "val"  # 前缀（等价于 /train）
endpoint_url = "http://aoss-internal.cn-sh-01b.sensecoreapi-oss.cn"
aws_access_key_id = "0198A1B9771F7BAAA9A55AC5B51ACC2F"
aws_secret_access_key = "0198A1B9771F7B9D998F202B044BE13C"
output_file = "imagenet_val_filelist.txt"  # 输出文件路径

# 创建S3客户端，增加超时和重试配置
config = Config(
    connect_timeout=300,  # 增加到300秒，防慢连接
    read_timeout=300,  # 增加到300秒
    retries={'max_attempts': 20, 'mode': 'adaptive'}  # 自适应重试，最大20次
)

try:
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        config=config
    )
    print("✅ 已创建AOSS客户端。")
except Exception as e:
    print(f"❌ 创建客户端失败: {e}")
    sys.exit(1)

# 先测试连接：列出所有桶（简单请求，验证连通性和凭证）
try:
    print("测试连接：尝试列出桶...")
    s3_client.list_buckets()
    print("✅ 连接测试成功！端点可达，凭证有效。")
except botocore.exceptions.ReadTimeoutError as e:
    print(f"❌ 连接超时: {e}")
    sys.exit(1)
except botocore.exceptions.ClientError as e:
    print(f"❌ 客户端错误（可能权限问题）: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 其他错误: {e}")
    sys.exit(1)

# 使用分页器列出对象
keys = []
paginator = s3_client.get_paginator('list_objects_v2')
page_iter = paginator.paginate(
    Bucket=bucket,
    Prefix=prefix,
    PaginationConfig={'MaxKeys': 1000}  # 每页最多1000，控制负载
)

# 进度条
with tqdm(desc="处理分页", unit="页") as pbar:
    for page_num, page in enumerate(page_iter, start=1):
        start_time = time.time()
        token = page.get('NextContinuationToken', 'None')
        print(f"\n[页 {page_num}] 开始处理... ContinuationToken: {token}")

        try:
            contents = page.get('Contents', [])
            for obj in contents:
                key = obj.get('Key', '')
                if not key or key.endswith('/'):
                    continue
                ext = os.path.splitext(key)[1]
                if ext in IMG_EXTS:
                    keys.append(key)
            duration = time.time() - start_time
            print(f"[页 {page_num}] 处理完成，用时 {duration:.2f} 秒，当前总keys: {len(keys)}")
        except botocore.exceptions.ReadTimeoutError as e:
            print(f"❌ [页 {page_num}] 超时: {e}。尝试重试...")
            # 可以加手动重试逻辑，但botocore已内置
            continue
        except Exception as e:
            print(f"❌ [页 {page_num}] 错误: {e}")
            continue

        pbar.update(1)

# 排序keys
keys.sort()

# 保存到文件
with open(output_file, 'w') as f:
    for key in keys:
        f.write(key + '\n')

print(f"🎉 已收集 {len(keys)} 个图片文件。保存到 {output_file}")
