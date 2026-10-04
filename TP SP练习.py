import numpy as np
import threading
import time
from typing import Tuple, List
import matplotlib.pyplot as plt

class TensorParallelSimulator:
    def __init__(self, matrix_size: int = 1024):
        """
        初始化张量并行模拟器

        Args:
            matrix_size: 矩阵大小 (n×n)
        """
        self.matrix_size = matrix_size
        self.numpy_runtime = None
        self.tp_runtime = None

    def generate_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """生成随机矩阵用于乘法"""
        np.random.seed(42)  # 设置随机种子以保证可重复性
        A = np.random.randn(self.matrix_size, self.matrix_size).astype(np.float32)
        B = np.random.randn(self.matrix_size, self.matrix_size).astype(np.float32)
        return A, B

    def numpy_matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """使用numpy的标准矩阵乘法（基准）"""
        start_time = time.time()
        C = np.dot(A, B)
        end_time = time.time()
        self.numpy_runtime = end_time - start_time
        return C

    def tp_matmul_worker(self, A_part: np.ndarray, B_part: np.ndarray,
                         result_part: np.ndarray, worker_id: int):
        """张量并行工作线程：计算部分矩阵乘法"""
        start_time = time.time()
        # 计算部分结果
        partial_result = np.dot(A_part, B_part)

        # 将结果存入共享数组的相应部分
        result_part[:] = partial_result

        end_time = time.time()
        print(f"  工作线程{worker_id}: 计算完成，耗时 {end_time - start_time:.3f}秒")

    def tensor_parallel_matmul(self, A: np.ndarray, B: np.ndarray,
                               num_workers: int = 2) -> np.ndarray:
        """
        张量并行矩阵乘法

        策略：将矩阵B按列分块，每个线程计算A与B的一个列块的乘积
        最后将结果按列拼接
        """
        print(f"\n张量并行矩阵乘法 (使用{num_workers}个工作线程)")
        print(f"矩阵A形状: {A.shape}, 矩阵B形状: {B.shape}")

        # 计算每个工作线程处理的列数
        n_cols = B.shape[1]
        cols_per_worker = n_cols // num_workers

        # 初始化结果矩阵
        C_tp = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)

        threads = []
        start_time = time.time()

        # 创建并启动工作线程
        for i in range(num_workers):
            # 计算当前线程处理的列范围
            start_col = i * cols_per_worker
            # 最后一个线程处理剩余的所有列
            end_col = start_col + cols_per_worker if i < num_workers - 1 else n_cols

            # 获取B的对应列块
            B_part = B[:, start_col:end_col]

            # 获取结果矩阵的对应部分
            C_part = C_tp[:, start_col:end_col]

            # 创建工作线程
            thread = threading.Thread(
                target=self.tp_matmul_worker,
                args=(A, B_part, C_part, i+1)
            )
            threads.append(thread)

            print(f"  分配任务给工作线程{i+1}: 处理B的列{start_col}:{end_col}")

        # 启动所有线程
        print("\n开始并行计算...")
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        end_time = time.time()
        self.tp_runtime = end_time - start_time

        print(f"\n所有工作线程完成，总耗时: {self.tp_runtime:.3f}秒")
        return C_tp

    def validate_result(self, C_numpy: np.ndarray, C_tp: np.ndarray) -> bool:
        """验证两种方法的结果是否一致（在数值误差范围内）"""
        # 计算最大绝对误差和相对误差
        abs_error = np.max(np.abs(C_numpy - C_tp))
        rel_error = np.max(np.abs(C_numpy - C_tp) / (np.abs(C_numpy) + 1e-10))

        print(f"\n结果验证:")
        print(f"  最大绝对误差: {abs_error:.6e}")
        print(f"  最大相对误差: {rel_error:.6e}")

        # 检查是否在可接受的误差范围内
        tolerance = 1e-5
        is_valid = abs_error < tolerance

        if is_valid:
            print("张量并行计算结果与标准numpy计算结果一致！")
        else:
            print("计算结果存在显著差异")

        return is_valid

    def compare_performance(self):
        """比较性能并打印结果"""
        print("\n" + "="*60)
        print("性能对比")
        print("="*60)
        print(f"标准numpy矩阵乘法耗时: {self.numpy_runtime:.3f}秒")
        print(f"张量并行(2线程)矩阵乘法耗时: {self.tp_runtime:.3f}秒")

        if self.numpy_runtime > self.tp_runtime:
            speedup = self.numpy_runtime / self.tp_runtime
            improvement = (self.numpy_runtime - self.tp_runtime) / self.numpy_runtime * 100
            print(f"加速比: {speedup:.2f}倍")
            print(f"性能提升: {improvement:.1f}%")
        else:
            print("注意: 由于Python GIL限制和线程开销，多线程可能不会加速CPU上的矩阵运算")

def plot_comparison():
    """绘制不同矩阵大小下的性能对比"""
    print("\n" + "="*60)
    print("不同矩阵大小下的性能分析")
    print("="*60)

    sizes = [256, 512, 1024, 2048, 4096]
    numpy_times = []
    tp_times = []

    for size in sizes:
        print(f"\n测试矩阵大小: {size}×{size}")
        simulator = TensorParallelSimulator(matrix_size=size)
        A, B = simulator.generate_matrices()

        # 标准numpy
        start = time.time()
        _ = np.dot(A, B)
        numpy_time = time.time() - start
        numpy_times.append(numpy_time)

        # 张量并行（2线程）
        C_tp = simulator.tensor_parallel_matmul(A, B, num_workers=2)
        tp_times.append(simulator.tp_runtime)

        print(f"  标准numpy: {numpy_time:.3f}秒")
        print(f"  张量并行: {simulator.tp_runtime:.3f}秒")

    # 绘制图表
    plt.figure(figsize=(10, 6))
    x = range(len(sizes))

    plt.bar([i - 0.2 for i in x], numpy_times, width=0.4, label='Standard numpy', alpha=0.8, color='blue')
    plt.bar([i + 0.2 for i in x], tp_times, width=0.4, label='Tensor Parallel (2 workers)', alpha=0.8, color='orange')

    plt.xlabel('Matrix Size')
    plt.ylabel('Computation Time (seconds)')
    plt.title('Tensor Parallel vs Standard Matrix Multiplication Performance')
    plt.xticks(x, [f'{size}×{size}' for size in sizes])
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')

    # 在柱状图上添加数值标签
    for i, v in enumerate(numpy_times):
        plt.text(i - 0.2, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(tp_times):
        plt.text(i + 0.2, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    # 计算加速比
    print("\n加速比分析:")
    for i, size in enumerate(sizes):
        speedup = numpy_times[i] / tp_times[i] if tp_times[i] > 0 else 0
        print(f"  {size}×{size}矩阵: 加速比 = {speedup:.2f}倍")

def main():
    """主函数"""
    print("="*60)
    print("张量并行(TP)策略演示")
    print("模拟矩阵乘法的张量并行计算")
    print("="*60)

    # 创建模拟器
    matrix_size = 2048  # 可以调整矩阵大小
    simulator = TensorParallelSimulator(matrix_size=matrix_size)

    # 生成测试矩阵
    print(f"\n生成{matrix_size}×{matrix_size}随机矩阵...")
    A, B = simulator.generate_matrices()

    # 1. 标准numpy矩阵乘法（基准）
    print("\n1. 标准numpy矩阵乘法 (基准):")
    C_numpy = simulator.numpy_matmul(A, B)
    print(f"   计算完成，耗时: {simulator.numpy_runtime:.3f}秒")

    # 2. 张量并行矩阵乘法
    print("\n2. 张量并行矩阵乘法:")
    C_tp = simulator.tensor_parallel_matmul(A, B, num_workers=2)

    # 3. 验证结果正确性
    print("\n3. 验证计算结果:")
    simulator.validate_result(C_numpy, C_tp)

    # 4. 性能对比
    simulator.compare_performance()

    # 5. 不同矩阵大小性能分析
    plot_comparison()


if __name__ == "__main__":
    main()
     