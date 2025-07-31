#!/usr/bin/env python
"""
Comprehensive MIMII Dataset Analysis for Novel Architecture Development
Analyzes patterns, characteristics, and insights unique to the MIMII dataset
to inform the design of a tailored neural network architecture.
"""
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import welch, spectrogram
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

class MIMIIDatasetAnalyzer:
    """Comprehensive analyzer for MIMII dataset patterns and characteristics"""
    
    def __init__(self, dataset_path: str, output_path: str):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.results = {}
        self.analysis_log = []
        
        # Audio processing parameters
        self.sr = 16000  # Standard sample rate for MIMII
        self.n_fft = 1024
        self.hop_length = 512
        self.n_mels = 64
        
    def log_finding(self, finding: str):
        """Log important findings for architecture design"""
        self.analysis_log.append(finding)
        logging.info(f"FINDING: {finding}")
    
    def analyze_dataset_structure(self):
        """Analyze the overall structure and distribution of the dataset"""
        self.log_finding("=== DATASET STRUCTURE ANALYSIS ===")
        
        structure_stats = {}
        
        # Find all machine directories
        machine_dirs = []
        for db_dir in self.dataset_path.glob("*"):
            if db_dir.is_dir() and not db_dir.name.startswith('.'):
                for machine_type_dir in db_dir.glob("*"):
                    if machine_type_dir.is_dir():
                        for machine_id_dir in machine_type_dir.glob("*"):
                            if machine_id_dir.is_dir() and (machine_id_dir / "normal").exists():
                                machine_dirs.append(machine_id_dir)
        
        self.log_finding(f"Total machine configurations found: {len(machine_dirs)}")
        
        # Analyze file distributions
        total_normal = 0
        total_abnormal = 0
        machine_stats = []
        
        for machine_dir in machine_dirs:
            normal_files = list((machine_dir / "normal").glob("*.wav"))
            abnormal_files = list((machine_dir / "abnormal").glob("*.wav"))
            
            total_normal += len(normal_files)
            total_abnormal += len(abnormal_files)
            
            machine_stats.append({
                'path': str(machine_dir),
                'db_level': machine_dir.parents[1].name,
                'machine_type': machine_dir.parents[0].name,
                'machine_id': machine_dir.name,
                'normal_count': len(normal_files),
                'abnormal_count': len(abnormal_files),
                'imbalance_ratio': len(normal_files) / max(len(abnormal_files), 1)
            })
        
        self.results['dataset_structure'] = {
            'total_machines': len(machine_dirs),
            'total_normal': total_normal,
            'total_abnormal': total_abnormal,
            'overall_imbalance': total_normal / max(total_abnormal, 1),
            'machine_stats': machine_stats
        }
        
        self.log_finding(f"Dataset imbalance ratio (normal/abnormal): {total_normal/max(total_abnormal, 1):.2f}")
        self.log_finding(f"Average normal files per machine: {total_normal/len(machine_dirs):.1f}")
        self.log_finding(f"Average abnormal files per machine: {total_abnormal/len(machine_dirs):.1f}")
        
        return machine_dirs
    
    def analyze_audio_characteristics(self, machine_dirs: List[Path], max_files_per_type: int = 50):
        """Analyze fundamental audio characteristics across the dataset"""
        self.log_finding("=== AUDIO CHARACTERISTICS ANALYSIS ===")
        
        audio_stats = {
            'durations': [],
            'sample_rates': [],
            'rms_energies': [],
            'spectral_centroids': [],
            'spectral_bandwidths': [],
            'zero_crossing_rates': [],
            'mfcc_means': [],
            'labels': [],
            'machine_types': [],
            'db_levels': []
        }
        
        for machine_dir in tqdm(machine_dirs[:10], desc="Analyzing audio characteristics"):  # Limit for efficiency
            db_level = machine_dir.parents[1].name
            machine_type = machine_dir.parents[0].name
            
            # Analyze normal files
            normal_files = list((machine_dir / "normal").glob("*.wav"))[:max_files_per_type]
            for file_path in normal_files:
                try:
                    y, sr = librosa.load(file_path, sr=None)
                    
                    # Basic characteristics
                    audio_stats['durations'].append(len(y) / sr)
                    audio_stats['sample_rates'].append(sr)
                    audio_stats['rms_energies'].append(np.sqrt(np.mean(y**2)))
                    audio_stats['zero_crossing_rates'].append(np.mean(librosa.feature.zero_crossing_rate(y)))
                    
                    # Spectral features
                    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
                    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    
                    audio_stats['spectral_centroids'].append(np.mean(spectral_centroids))
                    audio_stats['spectral_bandwidths'].append(np.mean(spectral_bandwidth))
                    audio_stats['mfcc_means'].append(np.mean(mfccs, axis=1))
                    
                    audio_stats['labels'].append('normal')
                    audio_stats['machine_types'].append(machine_type)
                    audio_stats['db_levels'].append(db_level)
                    
                except Exception as e:
                    logging.warning(f"Error processing {file_path}: {e}")
            
            # Analyze abnormal files
            abnormal_files = list((machine_dir / "abnormal").glob("*.wav"))[:max_files_per_type]
            for file_path in abnormal_files:
                try:
                    y, sr = librosa.load(file_path, sr=None)
                    
                    # Basic characteristics
                    audio_stats['durations'].append(len(y) / sr)
                    audio_stats['sample_rates'].append(sr)
                    audio_stats['rms_energies'].append(np.sqrt(np.mean(y**2)))
                    audio_stats['zero_crossing_rates'].append(np.mean(librosa.feature.zero_crossing_rate(y)))
                    
                    # Spectral features
                    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
                    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    
                    audio_stats['spectral_centroids'].append(np.mean(spectral_centroids))
                    audio_stats['spectral_bandwidths'].append(np.mean(spectral_bandwidth))
                    audio_stats['mfcc_means'].append(np.mean(mfccs, axis=1))
                    
                    audio_stats['labels'].append('abnormal')
                    audio_stats['machine_types'].append(machine_type)
                    audio_stats['db_levels'].append(db_level)
                    
                except Exception as e:
                    logging.warning(f"Error processing {file_path}: {e}")
        
        # Statistical analysis
        normal_mask = np.array(audio_stats['labels']) == 'normal'
        abnormal_mask = np.array(audio_stats['labels']) == 'abnormal'
        
        # Compare normal vs abnormal characteristics
        for feature in ['rms_energies', 'spectral_centroids', 'spectral_bandwidths', 'zero_crossing_rates']:
            normal_vals = np.array(audio_stats[feature])[normal_mask]
            abnormal_vals = np.array(audio_stats[feature])[abnormal_mask]
            
            if len(normal_vals) > 0 and len(abnormal_vals) > 0:
                t_stat, p_val = stats.ttest_ind(normal_vals, abnormal_vals)
                effect_size = (np.mean(abnormal_vals) - np.mean(normal_vals)) / np.sqrt((np.var(normal_vals) + np.var(abnormal_vals)) / 2)
                
                self.log_finding(f"{feature}: Normal={np.mean(normal_vals):.4f}±{np.std(normal_vals):.4f}, "
                               f"Abnormal={np.mean(abnormal_vals):.4f}±{np.std(abnormal_vals):.4f}, "
                               f"Effect size={effect_size:.3f}, p={p_val:.4f}")
        
        self.results['audio_characteristics'] = audio_stats
        
        # Key insights for architecture design
        duration_std = np.std(audio_stats['durations'])
        self.log_finding(f"Audio duration variability (std): {duration_std:.3f}s - {'HIGH' if duration_std > 1.0 else 'LOW'} variability")
        
        return audio_stats
    
    def analyze_spectral_patterns(self, machine_dirs: List[Path], max_files: int = 100):
        """Analyze spectral patterns and frequency domain characteristics"""
        self.log_finding("=== SPECTRAL PATTERNS ANALYSIS ===")
        
        spectral_data = {
            'mel_spectrograms': [],
            'power_spectra': [],
            'labels': [],
            'machine_types': [],
            'dominant_frequencies': [],
            'spectral_rolloffs': [],
            'spectral_contrasts': []
        }
        
        for machine_dir in tqdm(machine_dirs[:5], desc="Analyzing spectral patterns"):  # Limit for efficiency
            machine_type = machine_dir.parents[0].name
            
            # Process both normal and abnormal files
            for label in ['normal', 'abnormal']:
                files = list((machine_dir / label).glob("*.wav"))[:max_files//2]
                
                for file_path in files:
                    try:
                        y, sr = librosa.load(file_path, sr=self.sr)
                        
                        # Mel spectrogram
                        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, 
                                                                n_fft=self.n_fft, hop_length=self.hop_length)
                        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
                        
                        # Power spectrum
                        freqs, psd = welch(y, sr, nperseg=1024)
                        
                        # Spectral features
                        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                        
                        # Find dominant frequency
                        dominant_freq_idx = np.argmax(psd)
                        dominant_freq = freqs[dominant_freq_idx]
                        
                        spectral_data['mel_spectrograms'].append(log_mel)
                        spectral_data['power_spectra'].append(psd)
                        spectral_data['labels'].append(label)
                        spectral_data['machine_types'].append(machine_type)
                        spectral_data['dominant_frequencies'].append(dominant_freq)
                        spectral_data['spectral_rolloffs'].append(np.mean(spectral_rolloff))
                        spectral_data['spectral_contrasts'].append(np.mean(spectral_contrast))
                        
                    except Exception as e:
                        logging.warning(f"Error processing {file_path}: {e}")
        
        # Analyze spectral differences
        normal_mask = np.array(spectral_data['labels']) == 'normal'
        abnormal_mask = np.array(spectral_data['labels']) == 'abnormal'
        
        # Dominant frequency analysis
        normal_freqs = np.array(spectral_data['dominant_frequencies'])[normal_mask]
        abnormal_freqs = np.array(spectral_data['dominant_frequencies'])[abnormal_mask]
        
        if len(normal_freqs) > 0 and len(abnormal_freqs) > 0:
            freq_diff = np.mean(abnormal_freqs) - np.mean(normal_freqs)
            self.log_finding(f"Dominant frequency shift (abnormal - normal): {freq_diff:.1f} Hz")
            
            # Frequency band analysis
            low_freq_normal = np.sum(normal_freqs < 1000) / len(normal_freqs)
            low_freq_abnormal = np.sum(abnormal_freqs < 1000) / len(abnormal_freqs)
            self.log_finding(f"Low frequency dominance (<1kHz): Normal={low_freq_normal:.2f}, Abnormal={low_freq_abnormal:.2f}")
        
        # Spectral rolloff analysis
        normal_rolloff = np.array(spectral_data['spectral_rolloffs'])[normal_mask]
        abnormal_rolloff = np.array(spectral_data['spectral_rolloffs'])[abnormal_mask]
        
        if len(normal_rolloff) > 0 and len(abnormal_rolloff) > 0:
            rolloff_diff = np.mean(abnormal_rolloff) - np.mean(normal_rolloff)
            self.log_finding(f"Spectral rolloff difference: {rolloff_diff:.1f} Hz")
        
        self.results['spectral_patterns'] = spectral_data
        
        return spectral_data
    
    def analyze_temporal_patterns(self, machine_dirs: List[Path], max_files: int = 50):
        """Analyze temporal patterns and time-domain characteristics"""
        self.log_finding("=== TEMPORAL PATTERNS ANALYSIS ===")
        
        temporal_data = {
            'onset_patterns': [],
            'tempo_estimates': [],
            'rhythm_patterns': [],
            'envelope_shapes': [],
            'labels': [],
            'autocorrelations': [],
            'periodicity_strengths': []
        }
        
        for machine_dir in tqdm(machine_dirs[:5], desc="Analyzing temporal patterns"):
            for label in ['normal', 'abnormal']:
                files = list((machine_dir / label).glob("*.wav"))[:max_files//2]
                
                for file_path in files:
                    try:
                        y, sr = librosa.load(file_path, sr=self.sr)
                        
                        # Onset detection
                        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
                        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                        
                        # Tempo estimation
                        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                        
                        # Envelope analysis
                        envelope = np.abs(librosa.stft(y))
                        envelope_shape = np.mean(envelope, axis=0)
                        
                        # Autocorrelation for periodicity
                        autocorr = np.correlate(y, y, mode='full')
                        autocorr = autocorr[autocorr.size // 2:]
                        
                        # Periodicity strength (ratio of max autocorr to initial value)
                        if len(autocorr) > 1:
                            periodicity = np.max(autocorr[1:]) / autocorr[0] if autocorr[0] != 0 else 0
                        else:
                            periodicity = 0
                        
                        temporal_data['onset_patterns'].append(len(onset_times))
                        temporal_data['tempo_estimates'].append(tempo)
                        temporal_data['envelope_shapes'].append(envelope_shape[:100])  # Truncate for consistency
                        temporal_data['labels'].append(label)
                        temporal_data['autocorrelations'].append(autocorr[:1000])  # Truncate
                        temporal_data['periodicity_strengths'].append(periodicity)
                        
                    except Exception as e:
                        logging.warning(f"Error processing {file_path}: {e}")
        
        # Analyze temporal differences
        normal_mask = np.array(temporal_data['labels']) == 'normal'
        abnormal_mask = np.array(temporal_data['labels']) == 'abnormal'
        
        # Onset pattern analysis
        normal_onsets = np.array(temporal_data['onset_patterns'])[normal_mask]
        abnormal_onsets = np.array(temporal_data['onset_patterns'])[abnormal_mask]
        
        if len(normal_onsets) > 0 and len(abnormal_onsets) > 0:
            onset_diff = np.mean(abnormal_onsets) - np.mean(normal_onsets)
            self.log_finding(f"Onset density difference: {onset_diff:.1f} onsets per file")
        
        # Periodicity analysis
        normal_periodicity = np.array(temporal_data['periodicity_strengths'])[normal_mask]
        abnormal_periodicity = np.array(temporal_data['periodicity_strengths'])[abnormal_mask]
        
        if len(normal_periodicity) > 0 and len(abnormal_periodicity) > 0:
            periodicity_diff = np.mean(abnormal_periodicity) - np.mean(normal_periodicity)
            self.log_finding(f"Periodicity strength difference: {periodicity_diff:.3f}")
        
        self.results['temporal_patterns'] = temporal_data
        
        return temporal_data
    
    def analyze_feature_correlations(self, audio_stats: Dict, spectral_data: Dict):
        """Analyze correlations between different feature types"""
        self.log_finding("=== FEATURE CORRELATION ANALYSIS ===")
        
        # Create feature matrix
        features = {}
        min_samples = min(len(audio_stats['rms_energies']), len(spectral_data['dominant_frequencies']))
        
        features['rms_energy'] = np.array(audio_stats['rms_energies'])[:min_samples]
        features['spectral_centroid'] = np.array(audio_stats['spectral_centroids'])[:min_samples]
        features['spectral_bandwidth'] = np.array(audio_stats['spectral_bandwidths'])[:min_samples]
        features['zero_crossing_rate'] = np.array(audio_stats['zero_crossing_rates'])[:min_samples]
        features['dominant_frequency'] = np.array(spectral_data['dominant_frequencies'])[:min_samples]
        features['spectral_rolloff'] = np.array(spectral_data['spectral_rolloffs'])[:min_samples]
        
        # Calculate correlation matrix
        feature_names = list(features.keys())
        feature_matrix = np.column_stack([features[name] for name in feature_names])
        
        corr_matrix = np.corrcoef(feature_matrix.T)
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                corr_val = corr_matrix[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append((feature_names[i], feature_names[j], corr_val))
        
        self.log_finding(f"Highly correlated feature pairs (|r| > 0.7):")
        for feat1, feat2, corr in high_corr_pairs:
            self.log_finding(f"  {feat1} <-> {feat2}: r={corr:.3f}")
        
        # PCA analysis for dimensionality insights
        pca = PCA()
        pca.fit(feature_matrix)
        
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Find number of components for 95% variance
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        self.log_finding(f"PCA Analysis: {n_components_95}/{len(feature_names)} components explain 95% variance")
        self.log_finding(f"Top 3 components explain {cumulative_variance[2]:.1%} of variance")
        
        self.results['feature_correlations'] = {
            'correlation_matrix': corr_matrix,
            'feature_names': feature_names,
            'high_correlations': high_corr_pairs,
            'pca_variance': explained_variance,
            'optimal_dimensions': n_components_95
        }
        
        return corr_matrix, feature_names
    
    def analyze_machine_type_differences(self, machine_dirs: List[Path]):
        """Analyze differences between machine types for architecture insights"""
        self.log_finding("=== MACHINE TYPE ANALYSIS ===")
        
        machine_features = {}
        
        for machine_dir in tqdm(machine_dirs[:10], desc="Analyzing machine types"):
            machine_type = machine_dir.parents[0].name
            
            if machine_type not in machine_features:
                machine_features[machine_type] = {
                    'spectral_centroids': [],
                    'rms_energies': [],
                    'dominant_freqs': []
                }
            
            # Sample a few files from each machine
            normal_files = list((machine_dir / "normal").glob("*.wav"))[:10]
            
            for file_path in normal_files:
                try:
                    y, sr = librosa.load(file_path, sr=self.sr)
                    
                    # Extract key features
                    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                    rms_energy = np.sqrt(np.mean(y**2))
                    
                    freqs, psd = welch(y, sr, nperseg=1024)
                    dominant_freq = freqs[np.argmax(psd)]
                    
                    machine_features[machine_type]['spectral_centroids'].append(spectral_centroid)
                    machine_features[machine_type]['rms_energies'].append(rms_energy)
                    machine_features[machine_type]['dominant_freqs'].append(dominant_freq)
                    
                except Exception as e:
                    continue
        
        # Compare machine types
        machine_types = list(machine_features.keys())
        self.log_finding(f"Machine types found: {machine_types}")
        
        for feature in ['spectral_centroids', 'rms_energies', 'dominant_freqs']:
            self.log_finding(f"\n{feature.upper()} by machine type:")
            for machine_type in machine_types:
                values = machine_features[machine_type][feature]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    self.log_finding(f"  {machine_type}: {mean_val:.2f} ± {std_val:.2f}")
        
        # Calculate inter-machine variability
        all_centroids = []
        machine_labels = []
        for machine_type, features in machine_features.items():
            all_centroids.extend(features['spectral_centroids'])
            machine_labels.extend([machine_type] * len(features['spectral_centroids']))
        
        if len(set(machine_labels)) > 1:
            # ANOVA to test for significant differences
            groups = [machine_features[mt]['spectral_centroids'] for mt in machine_types if machine_features[mt]['spectral_centroids']]
            if len(groups) > 1 and all(len(g) > 0 for g in groups):
                f_stat, p_val = stats.f_oneway(*groups)
                self.log_finding(f"Machine type spectral differences: F={f_stat:.2f}, p={p_val:.4f}")
        
        self.results['machine_type_analysis'] = machine_features
        
        return machine_features
    
    def generate_architecture_insights(self):
        """Generate specific insights for novel architecture design"""
        self.log_finding("=== ARCHITECTURE DESIGN INSIGHTS ===")
        
        insights = []
        
        # 1. Input processing insights
        if 'audio_characteristics' in self.results:
            duration_var = np.std(self.results['audio_characteristics']['durations'])
            if duration_var > 1.0:
                insights.append("HIGH_DURATION_VARIABILITY: Use adaptive/variable-length input processing")
            else:
                insights.append("LOW_DURATION_VARIABILITY: Fixed-length windows are suitable")
        
        # 2. Feature dimensionality insights
        if 'feature_correlations' in self.results:
            optimal_dims = self.results['feature_correlations']['optimal_dimensions']
            total_features = len(self.results['feature_correlations']['feature_names'])
            compression_ratio = optimal_dims / total_features
            
            if compression_ratio < 0.5:
                insights.append(f"HIGH_REDUNDANCY: Feature compression beneficial (reduce to {optimal_dims}/{total_features})")
            else:
                insights.append("LOW_REDUNDANCY: Preserve most feature dimensions")
        
        # 3. Spectral processing insights
        if 'spectral_patterns' in self.results:
            normal_mask = np.array(self.results['spectral_patterns']['labels']) == 'normal'
            abnormal_mask = np.array(self.results['spectral_patterns']['labels']) == 'abnormal'
            
            if len(self.results['spectral_patterns']['dominant_frequencies']) > 0:
                normal_freqs = np.array(self.results['spectral_patterns']['dominant_frequencies'])[normal_mask]
                abnormal_freqs = np.array(self.results['spectral_patterns']['dominant_frequencies'])[abnormal_mask]
                
                if len(normal_freqs) > 0 and len(abnormal_freqs) > 0:
                    freq_separation = abs(np.mean(abnormal_freqs) - np.mean(normal_freqs))
                    if freq_separation > 500:  # Hz
                        insights.append("FREQUENCY_SEPARATION: Multi-scale frequency analysis beneficial")
                    else:
                        insights.append("FREQUENCY_OVERLAP: Fine-grained spectral analysis needed")
        
        # 4. Temporal processing insights
        if 'temporal_patterns' in self.results:
            normal_mask = np.array(self.results['temporal_patterns']['labels']) == 'normal'
            abnormal_mask = np.array(self.results['temporal_patterns']['labels']) == 'abnormal'
            
            if len(self.results['temporal_patterns']['periodicity_strengths']) > 0:
                normal_periodicity = np.array(self.results['temporal_patterns']['periodicity_strengths'])[normal_mask]
                abnormal_periodicity = np.array(self.results['temporal_patterns']['periodicity_strengths'])[abnormal_mask]
                
                if len(normal_periodicity) > 0 and len(abnormal_periodicity) > 0:
                    periodicity_diff = abs(np.mean(abnormal_periodicity) - np.mean(normal_periodicity))
                    if periodicity_diff > 0.1:
                        insights.append("PERIODICITY_DIFFERENCE: Temporal pattern modeling crucial")
                    else:
                        insights.append("PERIODICITY_SIMILAR: Focus on spectral rather than temporal features")
        
        # 5. Machine type insights
        if 'machine_type_analysis' in self.results:
            machine_types = list(self.results['machine_type_analysis'].keys())
            if len(machine_types) > 1:
                insights.append("MULTI_MACHINE: Machine-type-aware architecture or domain adaptation needed")
            else:
                insights.append("SINGLE_MACHINE: Specialized architecture for specific machine type")
        
        # 6. Dataset balance insights
        if 'dataset_structure' in self.results:
            imbalance_ratio = self.results['dataset_structure']['overall_imbalance']
            if imbalance_ratio > 3:
                insights.append("HIGH_IMBALANCE: Implement class balancing or focal loss")
            else:
                insights.append("MODERATE_IMBALANCE: Standard training approaches suitable")
        
        self.log_finding("ARCHITECTURE RECOMMENDATIONS:")
        for insight in insights:
            self.log_finding(f"  {insight}")
        
        self.results['architecture_insights'] = insights
        
        return insights
    
    def save_results(self):
        """Save analysis results to text file for LLM consumption"""
        output_file = self.output_path / "dataset_analysis_results.txt"
        
        with open(output_file, 'w') as f:
            f.write("MIMII DATASET ANALYSIS FOR NOVEL ARCHITECTURE DESIGN\n")
            f.write("=" * 60 + "\n\n")
            
            # Write all findings
            for finding in self.analysis_log:
                f.write(finding + "\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("=" * 60 + "\n")
            
            # Dataset structure summary
            if 'dataset_structure' in self.results:
                ds = self.results['dataset_structure']
                f.write(f"Total machines: {ds['total_machines']}\n")
                f.write(f"Total normal samples: {ds['total_normal']}\n")
                f.write(f"Total abnormal samples: {ds['total_abnormal']}\n")
                f.write(f"Imbalance ratio: {ds['overall_imbalance']:.2f}\n\n")
            
            # Feature correlation summary
            if 'feature_correlations' in self.results:
                fc = self.results['feature_correlations']
                f.write(f"Optimal feature dimensions: {fc['optimal_dimensions']}/{len(fc['feature_names'])}\n")
                f.write(f"High correlation pairs: {len(fc['high_correlations'])}\n\n")
            
            # Architecture insights
            if 'architecture_insights' in self.results:
                f.write("ARCHITECTURE DESIGN RECOMMENDATIONS:\n")
                for insight in self.results['architecture_insights']:
                    f.write(f"  {insight}\n")
        
        logging.info(f"Analysis results saved to {output_file}")

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MIMII Dataset Analysis for Novel Architecture Design")
    parser.add_argument("--dataset_path", type=str, default="./dataset", 
                       help="Path to MIMII dataset")
    parser.add_argument("--output_path", type=str, default="./logs", 
                       help="Path to save analysis results")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = MIMIIDatasetAnalyzer(args.dataset_path, args.output_path)
    
    try:
        logging.info("Starting comprehensive MIMII dataset analysis...")
        start_time = time.time()
        
        # 1. Analyze dataset structure
        machine_dirs = analyzer.analyze_dataset_structure()
        
        if not machine_dirs:
            logging.error("No machine directories found. Check dataset path.")
            return
        
        # 2. Analyze audio characteristics
        audio_stats = analyzer.analyze_audio_characteristics(machine_dirs)
        
        # 3. Analyze spectral patterns
        spectral_data = analyzer.analyze_spectral_patterns(machine_dirs)
        
        # 4. Analyze temporal patterns
        temporal_data = analyzer.analyze_temporal_patterns(machine_dirs)
        
        # 5. Analyze feature correlations
        analyzer.analyze_feature_correlations(audio_stats, spectral_data)
        
        # 6. Analyze machine type differences
        analyzer.analyze_machine_type_differences(machine_dirs)
        
        # 7. Generate architecture insights
        analyzer.generate_architecture_insights()
        
        # 8. Save results
        analyzer.save_results()
        
        end_time = time.time()
        logging.info(f"Analysis completed in {end_time - start_time:.1f} seconds")
        
        # Print key insights
        print("\n" + "="*60)
        print("KEY INSIGHTS FOR NOVEL ARCHITECTURE DESIGN")
        print("="*60)
        
        if 'architecture_insights' in analyzer.results:
            for insight in analyzer.results['architecture_insights']:
                print(f"• {insight}")
        
        print(f"\nDetailed analysis saved to: {analyzer.output_path}/dataset_analysis_results.txt")
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
