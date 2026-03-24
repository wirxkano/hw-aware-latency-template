import numpy as np

from catalog import build_registry
from registry import DeviceSpec
from core.model import ModelFactory
from core.sample import EncodedSample
from core.dataset import DatasetBuilder
from core.pipeline import HardwareAwarePipeline
from encoders.arch_encoder import ArchEncoder
from encoders.device_encoder import DeviceEncoder
from encoders.cross_encoder import CrossEncoder

pipeline = HardwareAwarePipeline(log_transform_target=True)
pipeline.fit()
print(pipeline)

ARCH = "|nor_conv_3x3~0|+|nor_conv_1x1~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_1x1~2|"
x = pipeline.encode(ARCH, "edgegpu")
print("Feature vector shape:", x.shape)  # (50,)
print("Feature names:", pipeline.feature_names)  # 50 readable names

sample: EncodedSample = pipeline.encode_sample(ARCH, "raspi4")
print("One-hot block   :", sample.one_hot.shape)  # (30,)
print("Op counts block :", sample.op_counts.shape)  # (5,)
print("Derived block   :", sample.derived)  # [param_proxy, skip_ratio, none_ratio]
print("Device cont     :", sample.device_continuous)  # (6,) normalized
print("Device binary   :", sample.device_binary)  # (4,) flags

pipeline.register_device(
    name="jetson_orin",
    features=[275.0, 204.0, 4.0, 15.0, 1792, 1.3, 1, 0, 0, 0],
    description="NVIDIA Jetson AGX Orin — Ampere GPU",
    refit=True,  # scaler is re-fitted automatically
)
print("Devices now:", pipeline.device_names)


from nas_201_api import NASBench201API as API

api = API("NAS-Bench-201-v1_1-096897.pth", verbose=False)
X, y, meta = pipeline.build_dataset(
    api,
    device_names=["edgegpu", "raspi4", "eyeriss"],
    dataset="cifar10",
    max_archs=500,  # quick smoke test
)

model, metrics = pipeline.train(X, y, meta, n_estimators=800)
print(f"Overall MAE : {metrics['mae']:.4f}")
print(f"Overall MAPE: {metrics['mape']:.2f}%")
for dev, mae in metrics["per_device_mae"].items():
    print(f"  {dev:<12} MAE: {mae:.4f}")

registry = build_registry()

registry.register(
    DeviceSpec.from_list(
        name="custom_npu",
        features=[5.0, 50.0, 2.0, 8.0, 512, 1.0, 0, 1, 0, 0],
        description="Custom NPU prototype",
    )
)

arch_enc = ArchEncoder()
device_enc = DeviceEncoder(registry).fit()
cross_enc = CrossEncoder()

builder = DatasetBuilder(
    arch_encoder=arch_enc,
    device_encoder=device_enc,
    cross_encoder=cross_enc,
)

arch_vec = arch_enc.encode(ARCH)  # (38,)
dev_vec = device_enc.encode("custom_npu")  # (10,)
cross_vec = cross_enc.encode(arch_vec, dev_vec)  # (2,)

full_vec = np.concatenate([arch_vec, dev_vec, cross_vec])  # (50,)
print("Manual concat shape:", full_vec.shape)

metrics = ModelFactory.evaluate(model, X_test, y_test, meta, test_idx)
