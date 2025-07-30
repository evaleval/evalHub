import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
from schema.eval_types import EvaluationResult
from typing import Any, List, Union
from pathlib import Path
import json

from integrations.common.error import AdapterError, TransformationError

@dataclass
class AdapterMetadata:
    """Metadata about the adapter"""
    name: str
    version: str
    supported_library_versions: List[str]
    description: str

    
class SupportedLibrary(Enum):
    """Supported evaluation libraries"""
    LM_EVAL = "lm-evaluation-harness"
    INSPECT_AI = "inspect-ai"
    HELM = "helm"
    CUSTOM = "custom"


class BaseEvaluationAdapter(ABC):
    """
    Base class for all evaluation adapters.
    
    Each adapter is responsible for transforming evaluation outputs
    from a specific library into the unified schema format.
    """
    
    def __init__(self, strict_validation: bool = True):
        """
        Initialize the adapter.
        
        Args:
            strict_validation: If True, raise errors on validation failures.
                              If False, log warnings and continue.
        """
        self.strict_validation = strict_validation
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @property
    @abstractmethod
    def metadata(self) -> AdapterMetadata:
        """Return metadata about this adapter"""
        pass
    
    @property
    @abstractmethod
    def supported_library(self) -> SupportedLibrary:
        """Return the library this adapter supports"""
        pass
    
    @abstractmethod
    def _transform_single(self, raw_data: Any) -> EvaluationResult:
        """
        Transform a single evaluation record.
        
        Args:
            raw_data: Single evaluation record in library-specific format
            
        Returns:
            EvaluationResult in unified schema format
            
        Raises:
            TransformationError: If transformation fails
        """
        pass
    
    def transform(self, data: Any) -> Union[EvaluationResult, List[EvaluationResult]]:
        """
        Transform evaluation data to unified schema format.
        
        Args:
            data: Raw evaluation output (single record or list)
            
        Returns:
            Transformed data in unified schema format
        """
        try:
            # Handle both single records and lists
            if isinstance(data, list):
                results = []
                for i, item in enumerate(data):
                    try:
                        result = self._transform_single(item)
                        results.append(result)
                    except Exception as e:
                        self._handle_transformation_error(e, f"item {i}")
                return results
            else:
                return self._transform_single(data)
                
        except Exception as e:
            self._handle_transformation_error(e, "data transformation")
            
    def transform_from_file(self, file_path: Union[str, Path]) -> Union[EvaluationResult, List[EvaluationResult]]:
        """
        Load and transform evaluation data from file.
        
        Args:
            file_path: Path to the evaluation output file
            
        Returns:
            Transformed data in unified schema format
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise AdapterError(f"File not found: {file_path}")
        
        try:
            data = self._load_file(file_path)
            return self.transform(data)
        except Exception as e:
            raise AdapterError(f"Failed to load file {file_path}: {str(e)}")
        
    @abstractmethod
    def transform_from_directory(self, dir_path: Union[str, Path]) -> Union[EvaluationResult, List[EvaluationResult]]:
        """
        Load and transform evaluation data from all files in a directory.
        
        Args:
            dir_path: Path to the directory containing evaluation output files
            
        Returns:
            Transformed data in unified schema format
        """
        dir_path = Path(dir_path)
        
        if not dir_path.is_dir():
            raise AdapterError(f"Path is not a directory: {dir_path}")
        
        # Subclass must implement this part
        # e.g., how to iterate through files and process them
        pass

    def _load_file(self, file_path: Path) -> Any:
        """
        Load data from file. Override for custom file formats.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Loaded data
        """
        # Default implementation for JSON files
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif file_path.suffix.lower() == '.jsonl':
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data
        else:
            raise AdapterError(f"Unsupported file format: {file_path.suffix}")
    
    def _handle_transformation_error(self, error: Exception, context: str):
        """Handle transformation errors based on strict_validation setting"""
        error_msg = f"Transformation error in {context}: {str(error)}"
        
        if self.strict_validation:
            raise TransformationError(error_msg) from error
        else:
            self.logger.warning(error_msg)