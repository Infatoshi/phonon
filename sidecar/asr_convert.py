"""The Mac's speech model from the same published files Windows and Linux run.

Phonon's fine-tuned Parakeet (stock nvidia/parakeet-tdt-0.6b-v2 with the N8avg LoRA merged in) is
published once, as the sherpa-onnx int8 transducer (Infatoshi/phonon-asr-parakeet-tdt-0.6b-v2-onnx).
The Mac's runtimes (parakeet-mlx and the Metal helper pa) read the MLX layout instead: one
model.safetensors with the 697 tensors of mlx-community/parakeet-tdt-0.6b-v2 plus its config and
tokenizer files. `converted_model` builds that folder once from the hash-checked ONNX files and the
hash-checked small files of the MLX repository, and checks the result against the sha256 pinned in
model_pins.json (the conversion is deterministic), so what loads is exactly what was measured.

Conversion: every weight is dequantized as the int8 graph defines it, (q - zero_point) * scale per
tensor, and stored as bfloat16 (round to nearest even), the dtype both Mac runtimes compute in.
The export folded the conformer convolution's batch norm into the depthwise convolution: the
folded bias goes into batch_norm.bias and the norm is the identity (weight 1, mean 0, variance
1 - eps). LSTM gates are reordered from ONNX's (i, o, f, c) to MLX's (i, f, g, o) and the input and
recurrent biases summed. Weight layouts: linear [in, out] -> [out, in]; conv OIHW -> OHWI and
OIK -> OKI (MLX is channels-last).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import sys
from collections.abc import Callable

BATCH_NORM_EPS = 1e-5  # mlx.nn.BatchNorm's default, which parakeet-mlx uses
CONVERTED_DIR_NAME = "converted-models"


def to_bf16(array):
    """float32 -> bfloat16 bits (uint16), round to nearest even. The weights hold no NaN."""
    import numpy as np

    bits = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def reorder_lstm_gates(array, hidden: int):
    """ONNX LSTM gate blocks (i, o, f, c) along axis 0 -> MLX's (i, f, g, o)."""
    import numpy as np

    i, o, f, c = (array[k * hidden : (k + 1) * hidden] for k in range(4))
    return np.concatenate([i, f, c, o])


def write_safetensors(path: str, tensors: dict) -> None:
    """{name: (shape, uint16 bf16 array)} -> a safetensors file, names sorted, 8-byte aligned header."""
    header, offset = {}, 0
    for name in sorted(tensors):
        shape, data = tensors[name]
        size = data.size * 2
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        for name in sorted(tensors):
            handle.write(tensors[name][1].tobytes())


def _attach_external_data(graph, folder: str) -> None:
    """Read each external initializer (encoder.int8.onnx.data) into the tensor. onnx refuses a data file that is
    a symlink, and every file of a Hugging Face snapshot is one; the file was sha256-checked through it."""
    from onnx import TensorProto

    handles: dict = {}
    try:
        for tensor in graph.initializer:
            if tensor.data_location != TensorProto.EXTERNAL:
                continue
            info = {entry.key: entry.value for entry in tensor.external_data}
            location = info["location"]
            if os.path.isabs(location) or ".." in location.replace("\\", "/").split("/"):
                raise ValueError(f"external data outside the model folder: {location}")
            if location not in handles:
                handles[location] = open(os.path.realpath(os.path.join(folder, location)), "rb")  # noqa: SIM115
            handle = handles[location]
            handle.seek(int(info.get("offset", 0)))
            tensor.raw_data = handle.read(int(info["length"]))
            del tensor.external_data[:]
            tensor.data_location = TensorProto.DEFAULT
    finally:
        for handle in handles.values():
            handle.close()


class _Graph:
    def __init__(self, path: str):
        import onnx
        from onnx import numpy_helper

        graph = onnx.load(path, load_external_data=False).graph
        _attach_external_data(graph, os.path.dirname(path))
        self.nodes = list(graph.node)
        self.init = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
        self.consumers: dict = {}
        self.producer: dict = {}
        for node in self.nodes:
            for name in node.input:
                self.consumers.setdefault(name, []).append(node)
            for name in node.output:
                self.producer[name] = node

    def dequantize(self, quantized: str):
        import numpy as np

        base = quantized[: -len("_quantized")]
        q = self.init[quantized].astype(np.float32)
        return (q - self.init[base + "_zero_point"].astype(np.float32)) * self.init[base + "_scale"].astype(
            np.float32
        )

    def bias_after(self, node):
        """The float bias added to a quantized op's output (through its Cast and scale Mul), or None."""
        out = node.output[0]
        for _ in range(4):
            following = self.consumers.get(out, [])
            if len(following) != 1:
                return None
            nxt = following[0]
            if nxt.op_type == "Add":
                for name in nxt.input:
                    if name in self.init:
                        return name
                    source = self.producer.get(name)
                    if source is not None and source.op_type == "Reshape" and source.input[0] in self.init:
                        return source.input[0]
                return None
            if nxt.op_type not in ("Cast", "Mul"):
                return None
            out = nxt.output[0]
        return None


def _encoder(path: str, out: dict) -> None:
    import numpy as np

    graph = _Graph(path)
    used = set()
    for node in graph.nodes:
        if node.op_type not in ("MatMulInteger", "ConvInteger"):
            continue
        match = re.fullmatch(r"/(.+)/(MatMul|Conv)_quant", node.name)
        if not match:
            raise ValueError(f"unexpected quantized node {node.name}")
        module = ".".join(match.group(1).split("/"))
        weight_name = node.input[1]
        if weight_name.endswith(".weight_quantized") and not weight_name.startswith("onnx::"):
            module = weight_name[: -len(".weight_quantized")]
        weight = graph.dequantize(weight_name)
        if node.op_type == "MatMulInteger":
            weight = weight.T
        elif weight.ndim == 4:
            weight = weight.transpose(0, 2, 3, 1)
        else:
            weight = weight.transpose(0, 2, 1)
        key = "encoder." + module
        bias_name = graph.bias_after(node)
        out[key + ".weight"] = weight
        if module.endswith("depthwise_conv"):
            if bias_name is None:
                raise ValueError(f"{node.name}: no folded batch-norm bias")
            bias = graph.init[bias_name].reshape(-1).astype(np.float32)
            norm = key.rsplit(".", 1)[0] + ".batch_norm."
            out[norm + "bias"] = bias
            out[norm + "weight"] = np.ones_like(bias)
            out[norm + "running_mean"] = np.zeros_like(bias)
            out[norm + "running_var"] = np.full_like(bias, 1 - BATCH_NORM_EPS)
        elif bias_name is not None:
            out[key + ".bias"] = graph.init[bias_name].reshape(-1)
        if bias_name is not None:
            used.add(bias_name)
    for name, array in graph.init.items():
        if (
            name in used
            or array.dtype != np.float32
            or name.endswith("_scale")
            or not re.match(r"(layers|pre_encode)\.", name)
        ):
            continue
        out.setdefault("encoder." + name, array)


def _decoder(path: str, out: dict) -> None:
    import numpy as np

    graph = _Graph(path)
    out["decoder.prediction.embed.weight"] = graph.dequantize("prediction.embed.weight_quantized")
    lstms = [node for node in graph.nodes if node.op_type == "DynamicQuantizeLSTM"]
    if len(lstms) != 2:
        raise ValueError(f"expected 2 LSTM layers, found {len(lstms)}")

    def dequantize(q, scale, zero):
        init = graph.init
        return (init[q].astype(np.float32) - init[zero].astype(np.float32)) * init[scale].astype(np.float32)

    for layer, node in enumerate(lstms):
        _, w, r, b = node.input[:4]
        w_scale, w_zero, r_scale, r_zero = node.input[8:12]
        wx = dequantize(w, w_scale, w_zero)[0].T  # [4H, in]
        wh = dequantize(r, r_scale, r_zero)[0].T
        hidden = wx.shape[0] // 4
        bias = graph.init[b][0]
        prefix = f"decoder.prediction.dec_rnn.lstm.{layer}."
        out[prefix + "Wx"] = reorder_lstm_gates(wx, hidden)
        out[prefix + "Wh"] = reorder_lstm_gates(wh, hidden)
        out[prefix + "bias"] = reorder_lstm_gates(bias[: 4 * hidden] + bias[4 * hidden :], hidden)


def _joiner(path: str, out: dict) -> None:
    graph = _Graph(path)
    names = {"pred": "pred", "enc": "enc", "joint_net.joint_net.2": "joint_net.2"}
    for node in graph.nodes:
        if node.op_type != "MatMulInteger":
            continue
        module = names[".".join(re.fullmatch(r"/(.+)/MatMul_quant", node.name).group(1).split("/"))]
        out[f"joint.{module}.weight"] = graph.dequantize(node.input[1]).T
        out[f"joint.{module}.bias"] = graph.init[graph.bias_after(node)].reshape(-1)


def convert(onnx_dir: str, out_path: str, expected_names: set | None = None) -> None:
    """encoder/decoder/joiner .int8.onnx in `onnx_dir` -> MLX-layout bf16 model.safetensors."""
    tensors: dict = {}
    for part, build in (("encoder", _encoder), ("decoder", _decoder), ("joiner", _joiner)):
        arrays: dict = {}
        build(os.path.join(onnx_dir, f"{part}.int8.onnx"), arrays)
        for name, array in arrays.items():
            tensors[name] = (array.shape, to_bf16(array))
    if expected_names is not None and set(tensors) != expected_names:
        missing = sorted(expected_names - set(tensors))[:5]
        extra = sorted(set(tensors) - expected_names)[:5]
        raise ValueError(f"converted tensors differ from the MLX layout: missing {missing}, extra {extra}")
    write_safetensors(out_path, tensors)


def default_output_root() -> str:
    try:
        from cache_gc import phonon_data_dir
    except ImportError:
        from sidecar.cache_gc import phonon_data_dir
    data = phonon_data_dir()
    if data:
        return os.path.join(data, CONVERTED_DIR_NAME)
    from huggingface_hub import constants

    return os.path.join(constants.HF_HUB_CACHE, "phonon-" + CONVERTED_DIR_NAME)


def converted_model(
    repo_id: str,
    revision: str,
    onnx_dir: str,
    layout_dir: str,
    entry: dict,
    *,
    output_root: str | None = None,
    marker_dir: str | None = "default",
    log: Callable[[str], None] | None = None,
    convert_fn: Callable[[str, str], None] | None = None,
) -> str:
    """The MLX-layout folder for `repo_id`@`revision`: built once from `onnx_dir` (the verified ONNX
    snapshot) and `layout_dir` (the verified config and tokenizer files), its model.safetensors checked
    against entry["converted"]; a later launch checks it through the verification marker."""
    try:
        from model_pins import _marker_path, check_snapshot, default_marker_dir
    except ImportError:
        from sidecar.model_pins import _marker_path, check_snapshot, default_marker_dir

    def say(message: str) -> None:
        (log or (lambda m: print(f"speech model: {m}", file=sys.stderr, flush=True)))(message)

    layout_files = entry["mlx_layout"]["files"]
    expected = {"model.safetensors": entry["converted"]["model.safetensors"]}
    for name in layout_files:
        expected[name] = {"size": os.path.getsize(os.path.join(layout_dir, name))}
    root = output_root or default_output_root()
    folder = os.path.join(root, f"{repo_id.replace('/', '--')}@{revision}")
    if marker_dir == "default":
        marker_dir = default_marker_dir()
    marker = _marker_path(marker_dir, repo_id + "@converted") if marker_dir else None
    pinned = {"model.safetensors": expected["model.safetensors"]}

    def complete(path: str) -> bool:
        return all(os.path.isfile(os.path.join(path, name)) for name in expected)

    if complete(folder) and not check_snapshot(folder, revision, pinned, marker, say):
        return folder
    say("converting the published model to the MLX layout (first launch only)")
    os.makedirs(root, exist_ok=True)
    tmp = f"{folder}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    try:
        for name in layout_files:
            shutil.copyfile(os.path.join(layout_dir, name), os.path.join(tmp, name))
        (convert_fn or convert)(onnx_dir, os.path.join(tmp, "model.safetensors"))
        bad = check_snapshot(tmp, revision, pinned, None, say)
        if bad:
            raise ValueError(f"{repo_id}@{revision[:7]}: converted model.safetensors does not match its pinned sha256")
        shutil.rmtree(folder, ignore_errors=True)
        os.replace(tmp, folder)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if check_snapshot(folder, revision, pinned, marker, say):
        raise ValueError(f"{repo_id}@{revision[:7]}: converted model failed verification")
    return folder


def main(argv: list[str]) -> int:
    """asr_convert.py ONNX_DIR OUT.safetensors: convert and print the output's size and sha256."""
    try:
        from model_pins import sha256_of
    except ImportError:
        from sidecar.model_pins import sha256_of
    convert(argv[0], argv[1])
    print(json.dumps({"size": os.path.getsize(argv[1]), "sha256": sha256_of(argv[1])}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
