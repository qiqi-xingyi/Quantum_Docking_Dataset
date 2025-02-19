# --*-- conding:utf-8 --*--
# @Time : 11/11/24 1:21 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : utils.py

from qiskit import IBMQ
from qiskit.providers.jobstatus import JobStatus


class QuantumJobResultExtractor:
    def __init__(self, api_token, job_id, provider_name="ibm-q"):
        """
        初始化 QuantumJobResultExtractor 类，提供 API 令牌和 job_id 来提取任务结果和量子线路。

        :param api_token: IBM Quantum 的 API 令牌
        :param job_id: 要查询的量子作业 ID
        :param provider_name: IBM Quantum provider 的名称，默认为 'ibm-q'
        """
        self.api_token = api_token
        self.job_id = job_id
        self.provider_name = provider_name
        self.job = None

    def connect(self):
        """连接到 IBM Quantum 实体机器并获取指定作业"""
        # 加载 IBMQ 账户
        IBMQ.save_account(self.api_token, overwrite=True)
        IBMQ.load_account()

        # 获取 provider 和作业
        provider = IBMQ.get_provider(hub=self.provider_name)
        self.job = provider.retrieve_job(self.job_id)
        print(f"Connected to job ID: {self.job_id}")

    def get_final_qubit_probabilities(self):
        """
        获取每个 qubit 的测量结果（即概率分布）。
        :return: 每个 qubit 在测量中处于 |1⟩ 状态的概率
        """
        if self.job is None:
            raise ValueError("Job not found. Ensure that 'connect' method is called first.")

        # 检查作业状态是否已完成
        job_status = self.job.status()
        if job_status != JobStatus.DONE:
            raise ValueError(f"Job is not completed yet. Current status: {job_status.name}")

        # 获取作业结果并提取测量的概率分布
        result = self.job.result()
        counts = result.get_counts()

        # 计算每个 qubit 的最终状态概率
        num_qubits = self.job.circuits()[0].num_qubits
        qubit_probabilities = [0.0] * num_qubits

        for bitstring, count in counts.items():
            for i, bit in enumerate(reversed(bitstring)):  # 逆序以匹配 qubit 顺序
                qubit_probabilities[i] += int(bit) * count

        # 归一化结果
        total_shots = sum(counts.values())
        qubit_probabilities = [prob / total_shots for prob in qubit_probabilities]

        # 返回每个 qubit 最终处于 |1⟩ 状态的概率
        return qubit_probabilities

    def get_quantum_circuit(self):
        """
        获取提交到量子计算机的量子电路。
        :return: 量子电路对象
        """
        if self.job is None:
            raise ValueError("Job not found. Ensure that 'connect' method is called first.")

        # 返回量子电路
        return self.job.circuits()[0]
