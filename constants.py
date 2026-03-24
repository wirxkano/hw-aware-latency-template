OPS: tuple[str, ...] = (
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
)
OP_TO_IDX: dict[str, int] = {op: i for i, op in enumerate(OPS)}

NUM_EDGES: int = 6
NUM_OPS: int = len(OPS)

OP_MAC_COST: dict[str, float] = {
    "none": 0.0,
    "skip_connect": 0.0,
    "avg_pool_3x3": 1.0,
    "nor_conv_1x1": 1.0,
    "nor_conv_3x3": 9.0,
}

DEVICE_CONTINUOUS_IDX: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
DEVICE_BINARY_IDX: tuple[int, ...] = (6, 7, 8, 9)

DEVICE_FEATURE_NAMES: tuple[str, ...] = (
    "tops",
    "bandwidth",
    "cache_mb",
    "tdp_w",
    "num_cores",
    "clock_ghz",
    "is_gpu",
    "is_asic",
    "is_fpga",
    "is_mobile",
)

DIM_ARCH_ONEHOT: int = NUM_EDGES * NUM_OPS  # 30
DIM_OP_COUNTS: int = NUM_OPS  # 5
DIM_DERIVED: int = 3  # param_proxy, skip_ratio, none_ratio
DIM_DEV_CONT: int = len(DEVICE_CONTINUOUS_IDX)  # 6
DIM_DEV_BINARY: int = len(DEVICE_BINARY_IDX)  # 4
DIM_CROSS: int = 2  # compute_pressure, memory_pressure

DIM_ARCH: int = DIM_ARCH_ONEHOT + DIM_OP_COUNTS + DIM_DERIVED  # 38
DIM_DEVICE: int = DIM_DEV_CONT + DIM_DEV_BINARY  # 10
DIM_TOTAL: int = DIM_ARCH + DIM_DEVICE + DIM_CROSS  # 50

SLICE_ARCH_ONEHOT: slice = slice(0, 30)
SLICE_OP_COUNTS: slice = slice(30, 35)
SLICE_DERIVED: slice = slice(35, 38)
SLICE_DEV_CONT: slice = slice(38, 44)
SLICE_DEV_BINARY: slice = slice(44, 48)
SLICE_CROSS: slice = slice(48, 50)
