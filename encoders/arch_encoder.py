import numpy as np

from constants import DIM_ARCH, NUM_EDGES, NUM_OPS, OP_MAC_COST, OP_TO_IDX, OPS


class ArchEncoder:
    def encode(self, arch_str: str) -> np.ndarray:
        ops = self._parse(arch_str)
        return np.concatenate(
            [
                self._one_hot(ops),
                self._op_counts(ops),
                self._derived(ops),
            ]
        )

    def encode_batch(self, arch_strings: list[str]) -> np.ndarray:
        return np.stack([self.encode(s) for s in arch_strings])

    def _parse(self, arch_str: str) -> list[str]:
        ops: list[str] = []
        for node in arch_str.split("+"):
            for part in node.strip("|").split("|"):
                op = part.split("~")[0]
                if op not in OP_TO_IDX:
                    raise ValueError(
                        f"Unknown operation '{op}' in arch string '{arch_str}'. "
                        f"Valid ops: {list(OPS)}."
                    )
                ops.append(op)

        if len(ops) != NUM_EDGES:
            raise ValueError(
                f"Expected {NUM_EDGES} edges, parsed {len(ops)} " f"from '{arch_str}'."
            )
        return ops

    def _one_hot(self, ops: list[str]) -> np.ndarray:
        vec = np.zeros(NUM_EDGES * NUM_OPS, dtype=np.float32)
        for edge_idx, op in enumerate(ops):
            vec[edge_idx * NUM_OPS + OP_TO_IDX[op]] = 1.0
        return vec

    def _op_counts(self, ops: list[str]) -> np.ndarray:
        return np.array(
            [ops.count(op) for op in OPS],
            dtype=np.float32,
        )

    def _derived(self, ops: list[str]) -> np.ndarray:
        param_proxy = float(sum(OP_MAC_COST[op] for op in ops))
        skip_ratio = float(ops.count("skip_connect")) / NUM_EDGES
        none_ratio = float(ops.count("none")) / NUM_EDGES
        return np.array([param_proxy, skip_ratio, none_ratio], dtype=np.float32)

    @property
    def output_dim(self) -> int:
        return DIM_ARCH

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        for edge_idx in range(NUM_EDGES):
            for op in OPS:
                names.append(f"edge{edge_idx}_{op}")
        for op in OPS:
            names.append(f"count_{op}")
        names += ["param_proxy", "skip_ratio", "none_ratio"]
        return names

    def __repr__(self) -> str:
        return f"ArchEncoder(output_dim={self.output_dim})"
