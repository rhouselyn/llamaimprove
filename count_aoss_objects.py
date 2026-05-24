import os
import sys
import io
import time

import boto3
import botocore
from botocore.config import Config
from PIL import Image


# ============================================================
# 1. 清除代理
# ============================================================
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "*"


# ============================================================
# 2. AOSS 配置
# 这里用你当前 dataset/aoss_imagenet.py 里的外网配置
# ============================================================
bucket = "imagenet"

endpoint_url = "http://aoss.cn-sh-01b.sensecoreapi-oss.cn"

aws_access_key_id = "01997084CBF777519D5F10EC029154C6"
aws_secret_access_key = "01997084CBF7774488E90D66F4FCB83D"


# ============================================================
# 3. filelist 路径
# 你可以按实际路径修改
# ============================================================
filelist_path = "/mnt/afs/zhengmingkai/whl/llamagen/imagenet_train_filelist.txt"

output_image_path = "aoss_test_image.jpg"


def main():
    print("=" * 80)
    print("AOSS 单张图片读取测试")
    print(f"bucket: {bucket}")
    print(f"endpoint_url: {endpoint_url}")
    print(f"filelist_path: {filelist_path}")
    print("=" * 80)

    # ------------------------------------------------------------
    # 1. 检查 filelist 是否存在
    # ------------------------------------------------------------
    if not os.path.exists(filelist_path):
        print(f"❌ filelist 不存在: {filelist_path}")
        sys.exit(1)

    # ------------------------------------------------------------
    # 2. 读取第一条 object key
    # ------------------------------------------------------------
    with open(filelist_path, "r", encoding="utf-8") as f:
        keys = [line.strip() for line in f if line.strip()]

    if len(keys) == 0:
        print("❌ filelist 是空的")
        sys.exit(1)

    object_key = keys[0]

    print(f"filelist 总条数: {len(keys)}")
    print(f"准备测试第一张图片:")
    print(f"object_key: {object_key}")

    # ------------------------------------------------------------
    # 3. 创建 AOSS / S3 客户端
    # ------------------------------------------------------------
    config = Config(
        connect_timeout=300,
        read_timeout=300,
        retries={
            "max_attempts": 20,
            "mode": "adaptive"
        }
    )

    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            config=config,
        )
        print("✅ AOSS client 创建成功")
    except Exception as e:
        print(f"❌ AOSS client 创建失败: {e}")
        sys.exit(1)

    # ------------------------------------------------------------
    # 4. 读取单张图片
    # ------------------------------------------------------------
    try:
        start_time = time.time()

        response = s3_client.get_object(
            Bucket=bucket,
            Key=object_key
        )

        img_bytes = response["Body"].read()
        elapsed = time.time() - start_time

        print("✅ get_object 成功")
        print(f"下载字节数: {len(img_bytes)} bytes")
        print(f"耗时: {elapsed:.2f} 秒")

    except botocore.exceptions.ClientError as e:
        print("❌ get_object 失败，ClientError:")
        print(e)

        error_code = e.response.get("Error", {}).get("Code", "")
        print(f"错误码: {error_code}")

        if error_code == "NoSuchKey":
            print("说明 filelist 里的这个 key 在 bucket 里不存在。")
        elif error_code == "AccessDenied":
            print("说明当前账号没有这个 object 的读取权限。")
        else:
            print("可能是 bucket、key、endpoint 或权限问题。")

        sys.exit(1)

    except botocore.exceptions.ReadTimeoutError as e:
        print("❌ 读取超时:")
        print(e)
        sys.exit(1)

    except Exception as e:
        print("❌ 其他错误:")
        print(repr(e))
        sys.exit(1)

    # ------------------------------------------------------------
    # 5. 尝试解析成图片并保存
    # ------------------------------------------------------------
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        print("✅ 图片解析成功")
        print(f"图片尺寸: {img.size}")
        print(f"图片模式: {img.mode}")

        img.save(output_image_path)

        print(f"✅ 已保存测试图片到: {os.path.abspath(output_image_path)}")

    except Exception as e:
        print("❌ 图片解析失败，说明下载到的内容可能不是正常图片:")
        print(repr(e))
        sys.exit(1)

    print("=" * 80)
    print("测试完成：AOSS 可以正常通过 get_object 读取单张图片")
    print("=" * 80)


if __name__ == "__main__":
    main()