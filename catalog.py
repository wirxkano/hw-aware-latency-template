from registry import DeviceRegistry, DeviceSpec

EDGEGPU = DeviceSpec.from_list(
    name="edgegpu",
    features=[
        1.33,  # ~1.33 TOPS FP16
        59.7,  # 59.7 GB/s memory bandwidth
        0.5,  # 0.5 MB L2 cache
        7.5,  # 7.5 W TDP
        256,  # 256 CUDA cores
        0.854,  # 854 MHz GPU clock
        1,  # is_gpu
        0,  # is_asic
        0,  # is_fpga
        0,  # is_mobile
    ],
    description="NVIDIA Jetson Nano — Maxwell GPU, Max-Q mode",
)

RASPI4 = DeviceSpec.from_list(
    name="raspi4",
    features=[
        0.024,  # ~24 GOPS CPU only
        4.8,  # ~4.8 GB/s LPDDR4
        1.0,  # 1 MB shared L2
        7.5,  # ~7.5 W peak TDP
        4,  # 4 × Cortex-A72 cores
        1.5,  # 1.5 GHz
        0,
        0,
        0,
        1,
    ],
    description="Raspberry Pi 4 — Broadcom BCM2711, ARM Cortex-A72",
)

PIXEL3 = DeviceSpec.from_list(
    name="pixel3",
    features=[
        0.68,  # Adreno 630 ~680 GFLOPS FP16
        29.9,  # 29.9 GB/s LPDDR4X
        0.256,  # ~256 KB system L2
        5.0,  # ~5 W SoC TDP
        8,  # 8 cores (4×A75 + 4×A55)
        2.8,  # 2.8 GHz peak
        0,
        0,
        0,
        1,
    ],
    description="Google Pixel 3 — Snapdragon 845 + Pixel Visual Core",
)

FPGA = DeviceSpec.from_list(
    name="fpga",
    features=[
        0.9,  # ~900 GOPS effective
        4.0,  # ~4 GB/s DDR3
        0.437,  # ~437 KB block RAM
        10.0,  # ~10 W typical
        900,  # 900 DSPs
        0.15,  # ~150 MHz HLS clock
        0,
        0,
        1,
        0,
    ],
    description="Xilinx ZC706 — Zynq XC7Z045 SoC FPGA",
)

EYERISS = DeviceSpec.from_list(
    name="eyeriss",
    features=[
        0.168,  # 168 PEs @ ~200 MHz ≈ 0.17 TOPS
        0.0327,  # Very low off-chip DRAM BW (row-stationary)
        0.512,  # 512 KB GLB SRAM
        0.278,  # 278 mW on AlexNet
        168,  # 168 processing elements
        0.2,  # ~200 MHz
        0,
        1,
        0,
        0,
    ],
    description="MIT Eyeriss ASIC — 168 PE row-stationary spatial array",
)

DEFAULT_DEVICES: list[DeviceSpec] = [
    EDGEGPU,
    RASPI4,
    PIXEL3,
    FPGA,
    EYERISS,
]

def build_registry() -> DeviceRegistry:
    return DeviceRegistry(initial_devices=DEFAULT_DEVICES)
