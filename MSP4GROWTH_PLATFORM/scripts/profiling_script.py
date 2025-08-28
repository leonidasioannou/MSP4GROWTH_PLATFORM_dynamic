#!/usr/bin/env python3
"""
Profiling script for the MSP4GROWTH platform.
This script measures CPU usage, memory consumption, and execution time
for the different components of the analysis pipeline.
"""
import time
import os
import psutil
import cProfile
import pstats
import io
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import threading
import resource
import gc
import sys

# Add parent directory to path so we can import the msp4growth package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the refactored modules
from msp4growth.core.data_loader import MSP4GROWTHAnalysis
from msp4growth.core.gurobi_setup import initialize_gurobi
from msp4growth.config import initialize_environment

class ResourceProfiler:
    """A class to handle comprehensive resource profiling of the MSP4GROWTH application."""
    
    def __init__(self, output_dir="profiling_results"):
        """Initialize the profiler with an output directory for results."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.start_time = None
        self.process = psutil.Process(os.getpid())
        self.memory_samples = []
        self.cpu_samples = []
        self.sampling_times = []
        self.tracemalloc_started = False
        self.tracemalloc_snapshots = []

    def start_tracemalloc(self):
        """Start memory allocation tracking."""
        tracemalloc.start()
        self.tracemalloc_started = True
        
    def take_tracemalloc_snapshot(self, label):
        """Take a snapshot of current memory allocations."""
        if self.tracemalloc_started:
            snapshot = tracemalloc.take_snapshot()
            self.tracemalloc_snapshots.append((label, snapshot))
    
    def start(self):
        """Start the profiling session."""
        self.start_time = time.time()
        self.memory_samples = []
        self.cpu_samples = []
        self.sampling_times = []
        gc.collect()  # Force garbage collection before starting
    
    def sample(self):
        """Sample current CPU and memory usage."""
        if not self.start_time:
            raise RuntimeError("Profiler not started. Call start() first.")
            
        current_time = time.time() - self.start_time
        self.sampling_times.append(current_time)
        
        # Memory usage (RSS - Resident Set Size)
        mem_usage = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.memory_samples.append(mem_usage)
        
        # CPU usage
        cpu_percent = self.process.cpu_percent()
        self.cpu_samples.append(cpu_percent)
        
        return current_time, mem_usage, cpu_percent
    
    def stop_and_report(self):
        """Stop profiling and generate reports."""
        if not self.start_time:
            raise RuntimeError("Profiler not started. Call start() first.")
            
        duration = time.time() - self.start_time
        
        # Generate report dictionary
        report = {
            "duration_seconds": duration,
            "peak_memory_mb": max(self.memory_samples) if self.memory_samples else 0,
            "average_memory_mb": np.mean(self.memory_samples) if self.memory_samples else 0,
            "peak_cpu_percent": max(self.cpu_samples) if self.cpu_samples else 0,
            "average_cpu_percent": np.mean(self.cpu_samples) if self.cpu_samples else 0,
            "sampling_count": len(self.sampling_times)
        }
        
        # Save raw data
        raw_data = {
            "times": self.sampling_times,
            "memory_mb": self.memory_samples,
            "cpu_percent": self.cpu_samples
        }
        
        # Generate and save plots
        self._generate_plots()
        
        # Save report and raw data
        with open(self.output_dir / "profiling_summary.json", "w") as f:
            json.dump(report, f, indent=4)
            
        with open(self.output_dir / "profiling_raw_data.json", "w") as f:
            # Convert numpy values to native Python types for JSON serialization
            serializable_data = {
                "times": [float(x) for x in self.sampling_times],
                "memory_mb": [float(x) for x in self.memory_samples],
                "cpu_percent": [float(x) for x in self.cpu_samples]
            }
            json.dump(serializable_data, f, indent=4)
            
        # Stop tracemalloc if started
        if self.tracemalloc_started:
            self._analyze_memory_traces()
            tracemalloc.stop()
            self.tracemalloc_started = False
            
        return report
    
    def _generate_plots(self):
        """Generate plots for CPU and memory usage."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.sampling_times, self.memory_samples)
        plt.title("Memory Usage Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Memory Usage (MB)")
        plt.grid(True)
        plt.savefig(self.output_dir / "memory_usage.png")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.sampling_times, self.cpu_samples)
        plt.title("CPU Usage Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("CPU Usage (%)")
        plt.grid(True)
        plt.savefig(self.output_dir / "cpu_usage.png")
    
    def _analyze_memory_traces(self):
        """Analyze tracemalloc snapshots and generate reports."""
        if not self.tracemalloc_snapshots:
            return
            
        for i, (label, snapshot) in enumerate(self.tracemalloc_snapshots):
            top_stats = snapshot.statistics('lineno')
            
            # Write top memory users to a file
            with open(self.output_dir / f"memory_trace_{label}.txt", "w") as f:
                f.write(f"Top memory allocations for {label}:\n")
                for stat in top_stats[:20]:  # Top 20 allocations
                    f.write(f"{stat}\n")
            
            # Compare with previous snapshot if available
            if i > 0:
                prev_label, prev_snapshot = self.tracemalloc_snapshots[i-1]
                comparison = snapshot.compare_to(prev_snapshot, 'lineno')
                
                with open(self.output_dir / f"memory_diff_{prev_label}_to_{label}.txt", "w") as f:
                    f.write(f"Memory changes from {prev_label} to {label}:\n")
                    for stat in comparison[:20]:  # Top 20 differences
                        f.write(f"{stat}\n")

def load_config(config_path):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary with configuration parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {str(e)}")
        raise

def run_profiled_analysis(config_path):
    """Run the MSP4GROWTH analysis with comprehensive profiling."""
    # Set project root directory for data paths before initialization
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root = script_dir.parent
    
    # Override DATA_DIR in msp4growth.config
    # This is a monkey patch to fix the path issue without modifying the original code
    import sys
    if 'msp4growth.config' in sys.modules:
        import msp4growth.config
        msp4growth.config.DATA_DIR = project_root / "data"
        msp4growth.config.RESULTS_DIR = project_root / "results"
        print(f"Overriding data directory to: {msp4growth.config.DATA_DIR}")
        print(f"Overriding results directory to: {msp4growth.config.RESULTS_DIR}")
    
    # Create necessary directories
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Create sample habitat data file if it doesn't exist
    habitat_file = data_dir / "Habitat_map_clean.geojson"
    if not habitat_file.exists():
        print(f"Creating sample habitat data file at {habitat_file}")
        sample_habitat = create_sample_habitat_data()
        with open(habitat_file, 'w') as f:
            json.dump(sample_habitat, f)
    
    # Create other required files
    for filename in ["Bathymetry.geojson", "Coastline.geojson", "Ports.geojson", "WindSpeed.geojson"]:
        file_path = data_dir / filename
        
        if not file_path.exists():
            
            print(f"Creating sample {filename} file")
            sample_data = create_sample_geo_data(filename)
            with open(file_path, 'w') as f:
                json.dump(sample_data, f)
    
    # Initialize environment
    initialize_environment()
    
    # Initialize Gurobi environment
    gurobi_env = initialize_gurobi()
    
    # Initialize profilers
    resource_profiler = ResourceProfiler()
    resource_profiler.start_tracemalloc()
    resource_profiler.start()
    
    # For cProfile
    pr = cProfile.Profile()
    pr.enable()
    
    try:
        # Load configuration
        config = load_config(config_path)
        print(f"Loaded configuration: {config}")
        # Create analysis instance
        analyzer = MSP4GROWTHAnalysis(
            request_id=config["request_id"],
            input_polygon= config["input_polygon"],
            use_case=config["use_case"],
            model=config["model"],
            C_number=config["C_number"],
            N_size=config["N_size"],
            datasets=config.get("datasets"),
            thresholds=config["thresholds"],
            calc_type=config.get("calc_type")
        )
        
        # Take memory snapshot after initialization
        resource_profiler.take_tracemalloc_snapshot("initialization")
        
        # Start sampling thread
        def sampling_loop():
            while True:
                resource_profiler.sample()
                time.sleep(1)  # Sample every second
                
        sampling_thread = threading.Thread(target=sampling_loop)
        sampling_thread.daemon = True  # Thread will exit when main thread exits
        sampling_thread.start()
        
        # Run analysis
        print(f"Starting analysis at {time.strftime('%H:%M:%S')}")
        start_time = time.time()
        result_paths = analyzer.run_analysis()

        # Access the PDF report
        pdf_path = result_paths.get("pdf_report")
        print(f"PDF Report: {pdf_path}")
        end_time = time.time()
        print(f"Analysis completed at {time.strftime('%H:%M:%S')}")
        
        # Take memory snapshot after analysis
        resource_profiler.take_tracemalloc_snapshot("after_analysis")
        
        # Get resource usage
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        
        # Generate report
        analysis_report = {
            "execution_time": end_time - start_time,
            "max_rss_kb": rusage.ru_maxrss,
            "user_cpu_time": rusage.ru_utime,
            "system_cpu_time": rusage.ru_stime,
            "page_faults": rusage.ru_majflt,
            "voluntary_context_switches": rusage.ru_nvcsw,
            "involuntary_context_switches": rusage.ru_nivcsw,
            "output_files": [str(path) for path in result_paths.values()]
        }
        
        with open(resource_profiler.output_dir / "analysis_report.json", "w") as f:
            json.dump(analysis_report, f, indent=4)
            
        return result_paths, analysis_report
        
    finally:
        # Stop cProfile
        pr.disable()
        
        # Save cProfile results
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(50)  # Top 50 functions by time
        
        with open(resource_profiler.output_dir / "cprofile_results.txt", "w") as f:
            f.write(s.getvalue())
            
        # Create callers/callees profiles
        with open(resource_profiler.output_dir / "cprofile_callers.txt", "w") as f:
            ps = pstats.Stats(pr, stream=f).sort_stats('cumulative')
            ps.print_callers(50)
            
        with open(resource_profiler.output_dir / "cprofile_callees.txt", "w") as f:
            ps = pstats.Stats(pr, stream=f).sort_stats('cumulative')
            ps.print_callees(50)
        
        # Stop resource profiler and get report
        resource_report = resource_profiler.stop_and_report()
        
        print("\n=== Profiling Summary ===")
        print(f"Total run time: {resource_report['duration_seconds']:.2f} seconds")
        print(f"Peak memory usage: {resource_report['peak_memory_mb']:.2f} MB")
        print(f"Average memory usage: {resource_report['average_memory_mb']:.2f} MB")
        print(f"Peak CPU usage: {resource_report['peak_cpu_percent']:.2f}%")
        print(f"Average CPU usage: {resource_report['average_cpu_percent']:.2f}%")
        print(f"Detailed results saved to: {resource_profiler.output_dir}")

def main():
    """Main entry point for the profiling script."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # Default to test_input.json in the test_input directory
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        project_root = script_dir.parent
        config_path = project_root / "test_input" / "test_input.json"
    
    print(f"Running profiled analysis with config: {config_path}")
    result_paths, report = run_profiled_analysis(config_path)
    
    # Print summary of results
    print(f"Analysis complete in {report['execution_time']:.2f} seconds")
    print("Results saved to:")
    for model_name, path in result_paths.items():
        print(f"  - {model_name}: {path}")

if __name__ == "__main__":
    main()