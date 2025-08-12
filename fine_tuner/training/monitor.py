"""
Memory Monitor Module
====================

GPU 메모리 및 성능 모니터링을 담당하는 모듈
"""

import logging
import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """메모리 통계 저장 클래스"""
    gpu_name: str
    max_memory_gb: float
    start_memory_gb: float
    current_memory_gb: float
    used_memory_gb: float
    used_memory_for_training_gb: float
    used_percentage: float
    training_percentage: float


class MemoryMonitor:
    """GPU 메모리 모니터링을 담당하는 클래스"""
    
    def __init__(self):
        """MemoryMonitor 초기화"""
        self.device_id = 0
        self.gpu_stats = None
        self.start_gpu_memory = 0.0
        self.max_memory = 0.0
        
        self._initialize_gpu_stats()
    
    def _initialize_gpu_stats(self) -> None:
        """GPU 통계 초기화"""
        try:
            if torch.cuda.is_available():
                self.gpu_stats = torch.cuda.get_device_properties(self.device_id)
                self.max_memory = round(self.gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
                logger.info(f"GPU 초기화 완료: {self.gpu_stats.name}, 최대 메모리: {self.max_memory} GB")
            else:
                logger.warning("CUDA를 사용할 수 없습니다.")
                
        except Exception as e:
            logger.error(f"GPU 초기화 실패: {e}")
    
    def record_start_memory(self) -> float:
        """
        시작 시점 메모리 기록 (원본 로직)
        
        Returns:
            시작 시점 메모리 사용량 (GB)
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA를 사용할 수 없습니다.")
            return 0.0
        
        try:
            self.start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            
            # 원본 출력 재현
            print(f"GPU = {self.gpu_stats.name}. Max memory = {self.max_memory} GB.")
            print(f"{self.start_gpu_memory} GB of memory reserved.")
            
            logger.info(f"시작 메모리 기록: {self.start_gpu_memory} GB")
            return self.start_gpu_memory
            
        except Exception as e:
            logger.error(f"시작 메모리 기록 실패: {e}")
            return 0.0
    
    def get_current_memory_stats(self) -> MemoryStats:
        """
        현재 메모리 통계 반환
        
        Returns:
            현재 메모리 통계 객체
        """
        if not torch.cuda.is_available():
            return MemoryStats(
                gpu_name="No CUDA",
                max_memory_gb=0.0,
                start_memory_gb=0.0,
                current_memory_gb=0.0,
                used_memory_gb=0.0,
                used_memory_for_training_gb=0.0,
                used_percentage=0.0,
                training_percentage=0.0
            )
        
        try:
            current_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            used_memory_for_training = round(current_memory - self.start_gpu_memory, 3)
            used_percentage = round(current_memory / self.max_memory * 100, 3)
            training_percentage = round(used_memory_for_training / self.max_memory * 100, 3)
            
            return MemoryStats(
                gpu_name=self.gpu_stats.name if self.gpu_stats else "Unknown",
                max_memory_gb=self.max_memory,
                start_memory_gb=self.start_gpu_memory,
                current_memory_gb=current_memory,
                used_memory_gb=current_memory,
                used_memory_for_training_gb=used_memory_for_training,
                used_percentage=used_percentage,
                training_percentage=training_percentage
            )
            
        except Exception as e:
            logger.error(f"메모리 통계 수집 실패: {e}")
            return MemoryStats(
                gpu_name="Error",
                max_memory_gb=0.0,
                start_memory_gb=0.0,
                current_memory_gb=0.0,
                used_memory_gb=0.0,
                used_memory_for_training_gb=0.0,
                used_percentage=0.0,
                training_percentage=0.0
            )
    
    def print_training_stats(self, trainer_stats: Any) -> None:
        """
        훈련 통계 출력 (원본 로직 보존)
        
        Args:
            trainer_stats: 훈련 통계 객체
        """
        try:
            stats = self.get_current_memory_stats()
            
            # 원본 출력 재현
            print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
            print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
            print(f"Peak reserved memory = {stats.used_memory_gb} GB.")
            print(f"Peak reserved memory for training = {stats.used_memory_for_training_gb} GB.")
            print(f"Peak reserved memory % of max memory = {stats.used_percentage} %.")
            print(f"Peak reserved memory for training % of max memory = {stats.training_percentage} %.")
            
            logger.info(f"훈련 완료 - 시간: {trainer_stats.metrics['train_runtime']:.2f}초, "
                       f"메모리 사용: {stats.used_memory_gb} GB ({stats.used_percentage}%)")
            
        except Exception as e:
            logger.error(f"훈련 통계 출력 실패: {e}")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """
        메모리 보고서 반환
        
        Returns:
            메모리 상태 딕셔너리
        """
        stats = self.get_current_memory_stats()
        
        return {
            "gpu_name": stats.gpu_name,
            "max_memory_gb": stats.max_memory_gb,
            "start_memory_gb": stats.start_memory_gb,
            "current_memory_gb": stats.current_memory_gb,
            "used_memory_for_training_gb": stats.used_memory_for_training_gb,
            "used_percentage": stats.used_percentage,
            "training_percentage": stats.training_percentage,
            "available_memory_gb": stats.max_memory_gb - stats.current_memory_gb
        }
    
    def check_memory_usage(self, threshold: float = 90.0) -> bool:
        """
        메모리 사용량 체크
        
        Args:
            threshold: 경고 임계값 (%)
            
        Returns:
            임계값 초과 여부
        """
        stats = self.get_current_memory_stats()
        
        if stats.used_percentage > threshold:
            logger.warning(f"메모리 사용량 높음: {stats.used_percentage}% (임계값: {threshold}%)")
            return True
            
        return False