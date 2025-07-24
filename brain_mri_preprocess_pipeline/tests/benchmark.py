#!/usr/bin/env python3
"""
Performance benchmarking for the medical image processing pipeline.
Measures execution time and resource usage for different pipeline configurations.
"""

import os
import sys
import time
import json
import argparse
import psutil
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add parent directory to path to allow importing from parent modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brain_mri_preprocess_pipeline.utils.pipeline_orchestrator import PipelineOrchestrator, PipelineStep
from brain_mri_preprocess_pipeline.utils.models import ProcessingStats
from brain_mri_preprocess_pipeline.utils.logging_utils import LogManager
from brain_mri_preprocess_pipeline.utils.config import PipelineConfig

class BenchmarkResult:
    """Class to store benchmark results"""
    
    def __init__(self, name, parallel=False, workers=1):
        """Initialize benchmark result"""
        self.name = name
        self.parallel = parallel
        self.workers = workers
        self.start_time = time.time()
        self.end_time = None
        self.execution_time = None
        self.step_times = {}
        self.memory_usage = []
        self.cpu_usage = []
        
    def stop(self):
        """Stop the benchmark and calculate execution time"""
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
    
    def add_step_time(self, step_name, execution_time):
        """Add execution time for a step"""
        self.step_times[step_name] = execution_time
    
    def add_resource_usage(self, memory_mb, cpu_percent):
        """Add resource usage measurement"""
        self.memory_usage.append(memory_mb)
        self.cpu_usage.append(cpu_percent)
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "parallel": self.parallel,
            "workers": self.workers,
            "execution_time": self.execution_time,
            "step_times": self.step_times,
            "avg_memory_mb": sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
            "max_memory_mb": max(self.memory_usage) if self.memory_usage else 0,
            "avg_cpu_percent": sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
            "max_cpu_percent": max(self.cpu_usage) if self.cpu_usage else 0
        }

class MockProcessor:
    """Mock processor for benchmarking"""
    
    def __init__(self, name, processing_time=0.1, case_count=100):
        """Initialize mock processor"""
        self.name = name
        self.processing_time = processing_time
        self.case_count = case_count
    
    def process_directory(self, parallel=False, max_workers=1):
        """Mock processing method"""
        start_time = time.time()
        
        # Simulate processing time
        if parallel and max_workers > 1:
            # Parallel processing is faster
            time.sleep(self.processing_time * self.case_count / max_workers)
        else:
            # Sequential processing
            time.sleep(self.processing_time * self.case_count)
        
        end_time = time.time()
        
        # Return stats
        return ProcessingStats(
            total_cases=self.case_count,
            successful_cases=self.case_count,
            failed_cases=0,
            processing_time=end_time - start_time
        )

def run_benchmark(config, log_manager, name, parallel=False, workers=1, case_count=100):
    """
    Run a benchmark with the specified configuration
    
    Args:
        config: Pipeline configuration
        log_manager: Log manager
        name: Benchmark name
        parallel: Whether to use parallel processing
        workers: Number of parallel workers
        case_count: Number of cases to process
        
    Returns:
        BenchmarkResult object
    """
    # Create result object
    result = BenchmarkResult(name, parallel, workers)
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(config, log_manager)
    
    # Create steps with different processing times
    steps_data = [
        ("decompression", 0.01),    # Fast step
        ("conversion", 0.05),       # Medium step
        ("registration", 0.1),      # Slow step
        ("skull_stripping", 0.2),   # Very slow step
        ("encryption", 0.01),       # Fast step
        ("organization", 0.02)      # Medium-fast step
    ]
    
    # Add steps to orchestrator
    for name, processing_time in steps_data:
        processor = MockProcessor(name, processing_time, case_count)
        orchestrator.add_step(PipelineStep(
            name=name,
            description=f"Step {name}",
            processor=processor,
            method_name="process_directory"
        ))
    
    # Start resource monitoring in a separate thread
    stop_monitoring = False
    
    def monitor_resources():
        process = psutil.Process(os.getpid())
        while not stop_monitoring:
            # Get memory usage in MB
            memory_mb = process.memory_info().rss / (1024 * 1024)
            # Get CPU usage
            cpu_percent = process.cpu_percent(interval=0.1)
            # Add to result
            result.add_resource_usage(memory_mb, cpu_percent)
            time.sleep(0.5)
    
    # Start monitoring thread
    import threading
    monitor_thread = threading.Thread(target=monitor_resources)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Execute pipeline
    orchestrator.execute_pipeline(parallel=parallel)
    
    # Stop monitoring
    stop_monitoring = True
    monitor_thread.join(timeout=1.0)
    
    # Record step times
    for step in orchestrator.steps:
        result.add_step_time(step.name, step.stats.processing_time)
    
    # Stop benchmark
    result.stop()
    
    return result

def plot_results(results, output_dir):
    """
    Plot benchmark results
    
    Args:
        results: List of BenchmarkResult objects
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot execution time comparison
    plt.figure(figsize=(10, 6))
    names = [result.name for result in results]
    times = [result.execution_time for result in results]
    plt.bar(names, times)
    plt.title('Total Execution Time Comparison')
    plt.xlabel('Benchmark')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'execution_time_comparison.png')
    
    # Plot step times for each benchmark
    for result in results:
        plt.figure(figsize=(10, 6))
        step_names = list(result.step_times.keys())
        step_times = list(result.step_times.values())
        plt.bar(step_names, step_times)
        plt.title(f'Step Times for {result.name}')
        plt.xlabel('Step')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f'step_times_{result.name.replace(" ", "_")}.png')
    
    # Plot resource usage comparison
    plt.figure(figsize=(10, 6))
    names = [result.name for result in results]
    memory = [sum(result.memory_usage) / len(result.memory_usage) if result.memory_usage else 0 for result in results]
    plt.bar(names, memory)
    plt.title('Average Memory Usage Comparison')
    plt.xlabel('Benchmark')
    plt.ylabel('Memory (MB)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_usage_comparison.png')
    
    # Plot CPU usage comparison
    plt.figure(figsize=(10, 6))
    names = [result.name for result in results]
    cpu = [sum(result.cpu_usage) / len(result.cpu_usage) if result.cpu_usage else 0 for result in results]
    plt.bar(names, cpu)
    plt.title('Average CPU Usage Comparison')
    plt.xlabel('Benchmark')
    plt.ylabel('CPU Usage (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'cpu_usage_comparison.png')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Benchmark the medical image processing pipeline")
    parser.add_argument("--output", "-o", type=str, default="benchmark_results",
                        help="Output directory for benchmark results")
    parser.add_argument("--cases", "-c", type=int, default=100,
                        help="Number of cases to process")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Maximum number of parallel workers")
    parser.add_argument("--plot", "-p", action="store_true",
                        help="Generate plots of benchmark results")
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log manager
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_manager = LogManager(log_dir)
    logger = log_manager.setup_step_logger("benchmark")
    
    # Create temporary directories
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    input_dir = temp_dir / "input"
    input_dir.mkdir(exist_ok=True)
    
    # Create configuration
    config = PipelineConfig(
        input_directory=input_dir,
        output_directory=temp_dir / "output",
        log_directory=log_dir
    )
    
    # Define benchmarks to run
    benchmarks = [
        {"name": "Sequential", "parallel": False, "workers": 1},
        {"name": "Parallel (2 workers)", "parallel": True, "workers": 2},
        {"name": "Parallel (4 workers)", "parallel": True, "workers": 4},
        {"name": "Parallel (8 workers)", "parallel": True, "workers": 8}
    ]
    
    # Run benchmarks
    results = []
    for benchmark in benchmarks:
        logger.info(f"Running benchmark: {benchmark['name']}")
        result = run_benchmark(
            config=config,
            log_manager=log_manager,
            name=benchmark["name"],
            parallel=benchmark["parallel"],
            workers=benchmark["workers"],
            case_count=args.cases
        )
        results.append(result)
        logger.info(f"Benchmark {benchmark['name']} completed in {result.execution_time:.2f} seconds")
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"benchmark_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump([result.to_dict() for result in results], f, indent=2)
    
    logger.info(f"Benchmark results saved to {results_file}")
    
    # Generate plots if requested
    if args.plot:
        try:
            logger.info("Generating plots...")
            plot_results(results, output_dir)
            logger.info(f"Plots saved to {output_dir}")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
    
    # Print summary
    logger.info("\nBenchmark Summary:")
    for result in results:
        logger.info(f"{result.name}: {result.execution_time:.2f} seconds")
    
    # Calculate speedup
    if len(results) > 1:
        sequential_time = results[0].execution_time
        for result in results[1:]:
            speedup = sequential_time / result.execution_time
            logger.info(f"Speedup for {result.name}: {speedup:.2f}x")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())