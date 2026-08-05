#!/usr/bin/env python3
# -* coding: utf-8 -*-
"""
服务器硬件拓扑检测脚本 - 只检测 mlx5_bond_* 设备

可作为模块使用（调用 generate_topology_json），也可作为脚本独立运行。
"""

import argparse
import glob
import json
import os
import re
import subprocess

# 拓扑JSON文件名，默认生成到与本文件相同的目录下
TOPO_JSON_NAME = 'topology.json'


def run_command(cmd: str) -> str:
    """执行shell命令"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout.strip()
    except Exception:  # pylint: disable=broad-except
        return ""


def get_numa_node_count() -> int:
    """获取NUMA节点数量"""
    nodes = glob.glob("/sys/devices/system/node/node[0-9]*")
    return len(nodes) if nodes else 1


def get_pci_numa_node(pci_addr: str) -> int:
    """获取PCI设备的NUMA节点"""
    pci_addr = pci_addr.lower()
    if not pci_addr.startswith("0000:"):
        pci_addr = "0000:" + pci_addr

    numa_file = f"/sys/bus/pci/devices/{pci_addr}/numa_node"
    try:
        with open(numa_file, 'r') as f:
            node = int(f.read().strip())
            return node if node >= 0 else 0
    except Exception:  # pylint: disable=broad-except
        return 0


def discover_gpus() -> dict:
    """发现所有GPU"""
    gpus = {}
    output = run_command(
        "nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader,nounits 2>/dev/null")
    if output:
        for line in output.split('\n'):
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 2:
                    gpu_idx = int(parts[0].strip())
                    pci_id = parts[1].strip().lower()
                    # 标准化PCI地址
                    pci_id = re.sub(r'^0+:', '0000:', pci_id)
                    numa = get_pci_numa_node(pci_id)
                    gpus[gpu_idx] = {'pci': pci_id, 'numa': numa}
    return gpus


def discover_bond_devices() -> dict:
    """只发现 mlx5_bond_* RDMA设备"""
    bonds = {}

    # 通过 /sys/class/infiniband 查找 mlx5_bond_* 设备
    for path in glob.glob("/sys/class/infiniband/mlx5_bond_*"):
        name = os.path.basename(path)

        # 获取NUMA节点
        device_link = os.path.join(path, "device")
        if os.path.islink(device_link):
            real_path = os.path.realpath(device_link)
            pci_addr = os.path.basename(real_path)
            numa = get_pci_numa_node(pci_addr)
            bonds[name] = {'pci': pci_addr, 'numa': numa}

    return bonds


def parse_nvidia_topo_matrix() -> dict:
    """
    解析 nvidia-smi topo -m，获取GPU到mlx5_bond_*的亲和性
    返回: {gpu_idx: {bond_name: affinity_level}}
    """
    # 亲和性级别 (数字越小越近)
    affinity_rank = {'PIX': 0, 'PXB': 1, 'PHB': 2, 'NODE': 3, 'SYS': 4, 'X': 99}

    output = run_command("nvidia-smi topo -m 2>/dev/null")
    if not output:
        return {}

    lines = output.strip().split('\n')
    result = {}

    # 找表头行
    header_idx = -1
    headers = []
    for i, line in enumerate(lines):
        if line.strip().startswith('GPU0') or '\tGPU0' in line:
            headers = line.split()
            header_idx = i
            break

    if header_idx < 0:
        return {}

    # 找出 mlx5_bond_* 列的位置
    bond_columns = {}  # {col_index: bond_name}
    for idx, col in enumerate(headers):
        if col.startswith('mlx5_bond_'):
            bond_columns[idx] = col

    if not bond_columns:
        return {}

    # 解析GPU行
    for line in lines[header_idx + 1:]:
        line = line.strip()
        if not line or line.startswith('Legend') or line.startswith('NV'):
            continue

        parts = line.split()
        if not parts:
            continue

        # 检查是否是GPU行
        gpu_match = re.match(r'GPU(\d+)', parts[0])
        if not gpu_match:
            continue

        gpu_idx = int(gpu_match.group(1))
        result[gpu_idx] = {}

        for col_idx, bond_name in bond_columns.items():
            if col_idx < len(parts):
                affinity = parts[col_idx]
                level = affinity_rank.get(affinity, 99)
                result[gpu_idx][bond_name] = level

    return result


def _pci_bus(pci_addr: str) -> int:
    """从PCI地址提取bus编号（十六进制），失败返回-1"""
    try:
        # 形如 0000:03:00.0 -> 取 '03'
        return int(pci_addr.split(':')[-2], 16)
    except Exception:  # pylint: disable=broad-except
        return -1


def get_gpu_preferred_bond(gpu_idx: int, topo: dict, bonds: dict, gpus: dict) -> str:
    """获取GPU的首选bond设备"""

    # 方法1: 从nvidia-smi topo获取最近的bond
    # 注意: nvidia-smi topo -m 的列名为 NIC0..NICn，不含 mlx5_bond_* 名称，
    # 因此 topo 通常为空，此路径仅在能解析到bond列时生效。
    if gpu_idx in topo and topo[gpu_idx]:
        # 按亲和性级别排序，取最近的
        sorted_bonds = sorted(topo[gpu_idx].items(), key=lambda x: (x[1], x[0]))
        if sorted_bonds:
            return sorted_bonds[0][0]

    # 方法2: 按PCI总线距离选择同NUMA中最近的bond
    if gpu_idx in gpus:
        gpu_numa = gpus[gpu_idx]['numa']
        gpu_bus = _pci_bus(gpus[gpu_idx]['pci'])
        same_numa_bonds = [b for b, info in bonds.items() if info['numa'] == gpu_numa]
        if same_numa_bonds and gpu_bus >= 0:
            # 取与GPU的PCI bus距离最近的bond（同一PCIe switch下bus编号相邻）
            best = min(
                same_numa_bonds,
                key=lambda b: (abs(_pci_bus(bonds[b]['pci']) - gpu_bus), b))
            return best
        # 无PCI信息时回退到同NUMA的bond（按编号分配）
        if same_numa_bonds:
            same_numa_bonds = sorted(same_numa_bonds)
            numa_gpu_list = sorted([i for i, g in gpus.items() if g['numa'] == gpu_numa])
            local_idx = numa_gpu_list.index(gpu_idx) if gpu_idx in numa_gpu_list else 0
            return same_numa_bonds[local_idx % len(same_numa_bonds)]

    # 方法3: 返回第一个bond
    return sorted(bonds.keys())[0] if bonds else ""


def get_numa_bonds_ordered(numa: int, topo: dict, bonds: dict, gpus: dict) -> list:
    """获取某NUMA的bonds，按GPU亲和性顺序排列"""

    # 该NUMA上的GPU（排序）
    numa_gpus = sorted([i for i, g in gpus.items() if g['numa'] == numa])

    # 该NUMA上的bonds
    numa_bonds = [b for b, info in bonds.items() if info['numa'] == numa]

    if not numa_gpus or not numa_bonds:
        return sorted(numa_bonds)

    # 按GPU的首选bond顺序排列
    ordered = []
    for gpu_idx in numa_gpus:
        preferred = get_gpu_preferred_bond(gpu_idx, topo, bonds, gpus)
        if preferred in numa_bonds and preferred not in ordered:
            ordered.append(preferred)

    # 添加剩余的
    for b in sorted(numa_bonds):
        if b not in ordered:
            ordered.append(b)

    return ordered


def generate_topology(bonds: dict, gpus: dict, numa_count: int) -> dict:
    """生成拓扑JSON"""

    # 获取GPU-bond亲和性
    topo = parse_nvidia_topo_matrix()

    # 所有bonds排序
    all_bonds = sorted(bonds.keys())

    topology = {}

    # === CPU拓扑 ===
    for cpu_id in range(numa_count):
        # 本地bonds（按GPU亲和性排序）
        local = get_numa_bonds_ordered(cpu_id, topo, bonds, gpus)

        # 远程bonds
        remote = []
        for other in range(numa_count):
            if other != cpu_id:
                remote.extend(get_numa_bonds_ordered(other, topo, bonds, gpus))

        topology[f"cpu{cpu_id}"] = [local, remote]

    # === GPU拓扑 ===
    for gpu_idx in sorted(gpus.keys()):
        preferred = get_gpu_preferred_bond(gpu_idx, topo, bonds, gpus)
        topology[f"cuda:{gpu_idx}"] = [
            [preferred] if preferred else [],
            all_bonds
        ]

    return topology


def print_info(bonds: dict, gpus: dict, topo: dict):
    """打印检测信息"""
    print("\n" + "=" * 60)
    print("硬件拓扑检测结果")
    print("=" * 60)

    print(f"\nBond设备 ({len(bonds)}):")
    for numa in sorted(set(b['numa'] for b in bonds.values())):
        numa_bonds = [n for n, b in bonds.items() if b['numa'] == numa]
        print(f"   NUMA {numa}: {', '.join(sorted(numa_bonds))}")

    print(f"\nGPU ({len(gpus)}):")
    for idx in sorted(gpus.keys()):
        print(f"   cuda:{idx} -> NUMA {gpus[idx]['numa']}")

    if topo:
        print(f"\nGPU-Bond亲和性 (nvidia-smi topo):")
        bond_names = sorted(set(b for g in topo.values() for b in g.keys()))
        # 表头
        header = "        " + "  ".join(f"{b:>12}" for b in bond_names)
        print(header)
        for gpu_idx in sorted(topo.keys()):
            row = f"GPU{gpu_idx}    "
            for bond in bond_names:
                level = topo[gpu_idx].get(bond, 99)
                level_names = {0: 'PIX', 1: 'PXB', 2: 'PHB', 3: 'NODE', 4: 'SYS', 99: '?'}
                name = level_names.get(level, '?')
                row += f"{name:>12}  "
            print(row)

    print("=" * 60)


def generate_topology_json(output_path: str = None) -> str:
    """
    检测硬件拓扑并生成拓扑JSON文件。
    每次调用都会覆盖生成。默认输出到与本文件相同的目录下。

    :param output_path: 输出文件的完整路径；为空时默认使用本文件所在目录下的 topology.json
    :return: 生成文件的完整路径；若未检测到 mlx5_bond_* 设备则返回空字符串
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TOPO_JSON_NAME)

    numa_count = get_numa_node_count()
    bonds = discover_bond_devices()
    gpus = discover_gpus()

    if not bonds:
        return ""

    topology = generate_topology(bonds, gpus, numa_count)
    with open(output_path, 'w') as f:
        f.write(json.dumps(topology, indent=4))
    return output_path


def main():
    parser = argparse.ArgumentParser(description='生成GPU/NIC拓扑JSON')
    parser.add_argument('-o', '--output', default='topology.json', help='输出文件')
    parser.add_argument('--dry-run', action='store_true', help='只显示不写入')
    parser.add_argument('--pretty', action='store_true', help='格式化JSON')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    args = parser.parse_args()

    print("检测硬件拓扑...")

    numa_count = get_numa_node_count()
    bonds = discover_bond_devices()
    gpus = discover_gpus()
    topo = parse_nvidia_topo_matrix()

    if not bonds:
        print("未找到 mlx5_bond_* 设备")
        return 1

    if args.verbose or args.dry_run:
        print_info(bonds, gpus, topo)

    topology = generate_topology(bonds, gpus, numa_count)

    indent = 4 if args.pretty else None
    json_out = json.dumps(topology, indent=indent)

    if args.dry_run:
        print("\n生成的JSON:")
        print(json_out)
    else:
        with open(args.output, 'w') as f:
            f.write(json_out)
        print(f"已生成: {args.output}")

    return 0


if __name__ == '__main__':
    exit(main())
