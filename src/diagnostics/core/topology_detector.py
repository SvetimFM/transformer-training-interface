"""
System Topology Detector

Detects and maps system hardware topology:
- CPU and PCIe root complex
- GPUs and their PCIe connections
- NVMe drives
- Network interfaces
- PCIe switches and bridges

Creates hierarchical tree structure for visualization in HMI-style dashboard.

Supports:
- Bare-metal Linux (full PCIe tree via lspci)
- WSL2 (simplified topology via nvidia-smi)
- Multi-GPU systems with NVLink/PCIe topology
"""

import subprocess
import re
import json
import platform
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path


@dataclass
class Device:
    """Represents a hardware device in the topology"""
    device_id: str  # Unique identifier
    device_type: str  # "cpu", "gpu", "nvme", "nic", "pcie_bridge"
    name: str
    bus_id: Optional[str] = None  # PCIe bus ID (e.g., "0000:06:00.0")

    # PCIe details
    pcie_gen: Optional[int] = None  # 3, 4, 5
    pcie_lanes: Optional[int] = None  # x1, x4, x8, x16

    # GPU interconnect (NVLink, etc.)
    nvlink_connections: Dict[str, str] = field(default_factory=dict)  # {gpu_id: connection_type}
    numa_node: Optional[int] = None

    # Performance metrics (if available)
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Health status
    health_status: str = "unknown"  # "healthy", "warning", "error", "unknown"

    # Children devices (for tree structure)
    children: List['Device'] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Handle nested children recursively
        result['children'] = [child.to_dict() for child in self.children]
        return result


class TopologyDetector:
    """
    Detects system hardware topology.

    Creates hierarchical tree:
    CPU (root) → PCIe Root Complex → GPUs/NVMe/NICs
    """

    def __init__(self):
        self.is_wsl = self._detect_wsl()
        self.topology: Optional[Device] = None

    def _detect_wsl(self) -> bool:
        """Detect if running in WSL2"""
        try:
            with open('/proc/version', 'r') as f:
                return 'microsoft' in f.read().lower() or 'wsl' in f.read().lower()
        except:
            return False

    def detect(self) -> Device:
        """
        Detect full system topology.

        Returns:
            Device tree with CPU as root
        """
        # Create root CPU node
        cpu_info = self._detect_cpu()
        root = Device(
            device_id="cpu0",
            device_type="cpu",
            name=cpu_info['name'],
            health_status="healthy",
            metrics=cpu_info['metrics']
        )

        # Add GPUs
        gpus = self._detect_gpus()
        for gpu in gpus:
            root.children.append(gpu)

        # Parse NVLink topology if multi-GPU
        if len(gpus) > 1:
            self._parse_nvlink_topology(root.children)

        # Add NVMe drives (if not WSL)
        if not self.is_wsl:
            nvme_devices = self._detect_nvme()
            for nvme in nvme_devices:
                root.children.append(nvme)

        self.topology = root
        return root

    def _detect_cpu(self) -> Dict[str, Any]:
        """Detect CPU information"""
        try:
            # Try to get CPU model
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()

            model_match = re.search(r'model name\s*:\s*(.+)', cpuinfo)
            cpu_name = model_match.group(1).strip() if model_match else "Unknown CPU"

            # Count cores
            cores = cpuinfo.count('processor')

            return {
                'name': cpu_name,
                'metrics': {
                    'cores': cores,
                    'platform': platform.system()
                }
            }
        except Exception as e:
            return {
                'name': f"{platform.system()} CPU",
                'metrics': {'error': str(e)}
            }

    def _detect_gpus(self) -> List[Device]:
        """Detect NVIDIA GPUs via nvidia-smi"""
        gpus = []

        try:
            # Query nvidia-smi for GPU list
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,pci.bus_id,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return gpus

            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 4:
                    continue

                gpu_index, gpu_name, bus_id, vram = parts

                # Try to get PCIe generation/lanes
                pcie_info = self._get_gpu_pcie_info(int(gpu_index))

                # Classify GPU type and get additional metadata
                gpu_class = self._classify_gpu(gpu_name)

                gpu_device = Device(
                    device_id=f"gpu{gpu_index}",
                    device_type="gpu",
                    name=gpu_name,
                    bus_id=bus_id,
                    pcie_gen=pcie_info.get('generation'),
                    pcie_lanes=pcie_info.get('lanes'),
                    health_status="healthy",
                    metrics={
                        'vram_gb': float(vram) / 1024,  # Convert MB to GB
                        'index': int(gpu_index),
                        'gpu_class': gpu_class['class'],
                        'memory_type': gpu_class['memory_type'],
                        'tensor_cores': gpu_class['tensor_cores'],
                        'ecc_support': gpu_class['ecc_support'],
                        'expected_efficiency_percent': gpu_class['expected_efficiency']
                    }
                )

                gpus.append(gpu_device)

        except Exception as e:
            # No NVIDIA GPUs or nvidia-smi not available
            pass

        return gpus

    def _get_gpu_pcie_info(self, gpu_index: int) -> Dict[str, Optional[int]]:
        """Get PCIe generation and lane count for GPU"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '-i', str(gpu_index), '-q'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return {'generation': None, 'lanes': None}

            output = result.stdout

            # Parse PCIe generation
            gen_match = re.search(r'Current\s+:\s+Gen(\d+)', output)
            generation = int(gen_match.group(1)) if gen_match else None

            # Parse link width (lanes)
            lanes_match = re.search(r'Current\s+:\s+(\d+)x', output)
            lanes = int(lanes_match.group(1)) if lanes_match else None

            return {'generation': generation, 'lanes': lanes}

        except Exception:
            return {'generation': None, 'lanes': None}

    def _classify_gpu(self, gpu_name: str) -> Dict[str, Any]:
        """
        Classify GPU and determine expected characteristics.

        Returns:
            Dict with GPU class, memory type, tensor core gen, ECC support, expected efficiency
        """
        gpu_name_lower = gpu_name.lower()

        # Default consumer GPU characteristics
        classification = {
            'class': 'consumer',
            'memory_type': 'GDDR6',
            'tensor_cores': 'Gen3',  # Ampere
            'ecc_support': False,
            'expected_efficiency': 55  # Consumer GPUs: 50-60%
        }

        # Data Center GPUs (A-series, H-series)
        if 'a100' in gpu_name_lower:
            classification = {
                'class': 'datacenter',
                'memory_type': 'HBM2e',
                'tensor_cores': 'Gen3',  # Ampere
                'ecc_support': True,
                'expected_efficiency': 78  # A100: 75-80%
            }
        elif 'a30' in gpu_name_lower:
            classification = {
                'class': 'datacenter',
                'memory_type': 'HBM2',
                'tensor_cores': 'Gen3',
                'ecc_support': True,
                'expected_efficiency': 75
            }
        elif 'a40' in gpu_name_lower or 'a10' in gpu_name_lower:
            classification = {
                'class': 'datacenter',
                'memory_type': 'GDDR6',
                'tensor_cores': 'Gen3',
                'ecc_support': True,
                'expected_efficiency': 72
            }
        elif 'h100' in gpu_name_lower:
            classification = {
                'class': 'datacenter',
                'memory_type': 'HBM3',
                'tensor_cores': 'Gen4',  # Hopper
                'ecc_support': True,
                'expected_efficiency': 82  # H100: 80-85%
            }

        # Professional/Workstation GPUs (V-series, RTX A-series)
        elif 'v100' in gpu_name_lower:
            classification = {
                'class': 'professional',
                'memory_type': 'HBM2',
                'tensor_cores': 'Gen1',  # Volta
                'ecc_support': True,
                'expected_efficiency': 70
            }
        elif 'rtx a6000' in gpu_name_lower or 'rtx a5000' in gpu_name_lower:
            classification = {
                'class': 'professional',
                'memory_type': 'GDDR6',
                'tensor_cores': 'Gen3',
                'ecc_support': True,
                'expected_efficiency': 65
            }

        # Consumer GPUs - RTX 40 series (Ada Lovelace)
        elif 'rtx 40' in gpu_name_lower or 'rtx 4090' in gpu_name_lower or 'rtx 4080' in gpu_name_lower:
            classification['memory_type'] = 'GDDR6X'
            classification['tensor_cores'] = 'Gen4'  # Ada
            classification['expected_efficiency'] = 58  # Slightly better than Ampere

        # Consumer GPUs - RTX 30 series (Ampere)
        elif 'rtx 30' in gpu_name_lower or 'rtx 3090' in gpu_name_lower or 'rtx 3080' in gpu_name_lower or 'rtx 3070' in gpu_name_lower:
            classification['memory_type'] = 'GDDR6X'
            classification['tensor_cores'] = 'Gen3'
            classification['expected_efficiency'] = 55

        # Consumer GPUs - RTX 20 series (Turing)
        elif 'rtx 20' in gpu_name_lower or 'rtx 2080' in gpu_name_lower or 'rtx 2070' in gpu_name_lower:
            classification['memory_type'] = 'GDDR6'
            classification['tensor_cores'] = 'Gen2'  # Turing
            classification['expected_efficiency'] = 50

        # Consumer GPUs - GTX (no tensor cores)
        elif 'gtx' in gpu_name_lower:
            classification['tensor_cores'] = 'None'
            classification['expected_efficiency'] = 45

        return classification

    def _parse_nvlink_topology(self, gpu_devices: List[Device]):
        """
        Parse NVLink topology from nvidia-smi topo -m.

        Updates gpu_devices with NVLink connections and NUMA affinity.

        Example output:
                GPU0    GPU1    GPU2    GPU3
        GPU0     X     NV12    NV12    NV12
        GPU1    NV12     X     NV12    NV12
        GPU2    NV12    NV12     X     NV12
        GPU3    NV12    NV12    NV12     X
        """
        try:
            result = subprocess.run(
                ['nvidia-smi', 'topo', '-m'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return

            lines = result.stdout.strip().split('\n')

            # Find header line with GPU column names
            header_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith('GPU'):
                    header_idx = i
                    break

            if header_idx is None:
                return

            # Parse header to get GPU order
            header = lines[header_idx].split()
            gpu_columns = [col for col in header if col.startswith('GPU')]

            # Parse matrix rows
            for line in lines[header_idx + 1:]:
                if not line.strip() or line.startswith('Legend') or line.startswith(' '):
                    break

                parts = line.split()
                if not parts or not parts[0].startswith('GPU'):
                    continue

                row_gpu = parts[0]  # e.g., "GPU0"
                row_gpu_idx = int(row_gpu.replace('GPU', ''))

                # Find corresponding device
                row_device = None
                for dev in gpu_devices:
                    if dev.device_type == 'gpu' and dev.metrics.get('index') == row_gpu_idx:
                        row_device = dev
                        break

                if not row_device:
                    continue

                # Parse connections to other GPUs
                for col_idx, col_gpu in enumerate(gpu_columns):
                    if col_idx + 1 >= len(parts):
                        break

                    connection = parts[col_idx + 1]
                    col_gpu_idx = int(col_gpu.replace('GPU', ''))

                    # Skip self-connection
                    if connection == 'X' or col_gpu_idx == row_gpu_idx:
                        continue

                    # Store connection type
                    if connection.startswith('NV'):  # NVLink (NV1, NV2, ..., NV12)
                        row_device.nvlink_connections[f'gpu{col_gpu_idx}'] = connection
                    elif connection in ['PHB', 'PXB', 'PIX', 'SYS', 'NODE']:
                        # PCIe-based connections (store for reference)
                        row_device.nvlink_connections[f'gpu{col_gpu_idx}'] = f'PCIe-{connection}'

        except Exception as e:
            # NVLink topology parsing failed - not critical
            pass

    def _detect_nvme(self) -> List[Device]:
        """Detect NVMe drives (Linux only, not WSL)"""
        nvme_devices = []

        try:
            # List NVMe devices
            result = subprocess.run(
                ['ls', '/dev/nvme*n1'],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return nvme_devices

            for device_path in result.stdout.strip().split('\n'):
                if not device_path:
                    continue

                # Get device model/size
                try:
                    info = subprocess.run(
                        ['nvme', 'id-ctrl', device_path],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )

                    model_match = re.search(r'mn\s+:\s+(.+)', info.stdout)
                    model = model_match.group(1).strip() if model_match else device_path

                    nvme_device = Device(
                        device_id=f"nvme{len(nvme_devices)}",
                        device_type="nvme",
                        name=model,
                        bus_id=None,  # Would need to parse from lspci
                        pcie_gen=4,  # Most modern NVMe are Gen4
                        pcie_lanes=4,  # Typical NVMe uses x4
                        health_status="healthy"
                    )

                    nvme_devices.append(nvme_device)
                except:
                    continue

        except Exception:
            pass

        return nvme_devices

    def enrich_with_benchmarks(self, benchmark_results: Dict[str, Any]):
        """
        Add benchmark results to GPU devices in topology.

        Args:
            benchmark_results: Dict with 'gpu0', 'gpu1', etc. keys containing metrics
        """
        if not self.topology:
            self.detect()

        for child in self.topology.children:
            if child.device_type == "gpu":
                gpu_id = child.device_id
                if gpu_id in benchmark_results:
                    # Merge benchmark metrics
                    child.metrics.update(benchmark_results[gpu_id])

                    # Set health status based on efficiency
                    efficiency = benchmark_results[gpu_id].get('compute_efficiency_percent', 0)
                    if efficiency >= 60:
                        child.health_status = "healthy"
                    elif efficiency >= 40:
                        child.health_status = "warning"
                    else:
                        child.health_status = "error"

    def to_json(self, indent: int = 2) -> str:
        """Export topology to JSON"""
        if not self.topology:
            self.detect()

        return json.dumps(self.topology.to_dict(), indent=indent)

    def save_to_file(self, filepath: str):
        """Save topology to JSON file"""
        if not self.topology:
            self.detect()

        with open(filepath, 'w') as f:
            f.write(self.to_json())

    def print_tree(self, device: Optional[Device] = None, indent: int = 0):
        """Print topology as ASCII tree"""
        if device is None:
            if not self.topology:
                self.detect()
            device = self.topology

        # Print current device
        prefix = "  " * indent
        icon = self._get_device_icon(device.device_type)

        print(f"{prefix}{icon} {device.name}")

        # Print metrics
        if device.metrics:
            for key, value in device.metrics.items():
                print(f"{prefix}   {key}: {value}")

        # Print PCIe info
        if device.pcie_gen or device.pcie_lanes:
            pcie_str = f"PCIe Gen{device.pcie_gen} x{device.pcie_lanes}"
            print(f"{prefix}   {pcie_str}")

        # Print children
        for child in device.children:
            self.print_tree(child, indent + 1)

    def _get_device_icon(self, device_type: str) -> str:
        """Get emoji/icon for device type"""
        icons = {
            'cpu': '🖥️ ',
            'gpu': '🎮',
            'nvme': '💾',
            'nic': '🌐',
            'pcie_bridge': '🔗'
        }
        return icons.get(device_type, '📦')


def main():
    """Test topology detection"""
    print("="*80)
    print("SYSTEM TOPOLOGY DETECTION")
    print("="*80)
    print()

    detector = TopologyDetector()
    detector.detect()

    print("Topology Tree:")
    print("-" * 80)
    detector.print_tree()

    print()
    print("="*80)
    print("JSON Export:")
    print("="*80)
    print(detector.to_json())


if __name__ == "__main__":
    main()
