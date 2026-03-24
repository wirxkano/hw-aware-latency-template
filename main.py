from nas_201_api import NASBench201API as API
from hw_nas_bench_api import HWNASBenchAPI

from core.pipeline import HardwareAwarePipeline

pipeline = HardwareAwarePipeline(log_transform_target=True)
pipeline.fit()
print(pipeline)

nas_path = "/root/quoc-huy/GNN_Latency_Predictor/data/NAS-Bench-201-v1_0-e61699.pth"
hw_nas_path = "/root/quoc-huy/GNN_Latency_Predictor/data/HW-NAS-Bench-v1_0.pickle"
nas_api = API(nas_path, verbose=False)
hw_api = HWNASBenchAPI(hw_nas_path, search_space="nasbench201")
X, y, meta = pipeline.build_dataset(
    nas_api,
    hw_api,
    device_names=["edgegpu", "raspi4", "eyeriss", "pixel3", "fpga"],
    dataset="cifar10",
)

model, metrics = pipeline.train(X, y, meta, n_estimators=600)

print(f"Overall MAE : {metrics['mae']:.4f}")
print(f"Overall MAPE: {metrics['mape']:.2f}%")
for dev, mae in metrics["per_device_mae"].items():
    print(f"  {dev:<12} MAE: {mae:.4f}")