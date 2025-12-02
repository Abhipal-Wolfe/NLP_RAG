"""Check document length distribution in corpus"""
import json
from pathlib import Path
from collections import Counter, defaultdict

def analyze_corpus(corpus_path):
    """Analyze document lengths and show distribution"""
    
    print(f"Analyzing: {corpus_path}\n")
    
    if not Path(corpus_path).exists():
        print(f"File not found: {corpus_path}")
        return
    
    lengths = []
    source_counts = Counter()
    lengths_by_source = defaultdict(list)
    
    print("Reading corpus...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            text = doc.get('text', '')
            source = doc.get('source', 'unknown')
            
            doc_length = len(text)
            lengths.append(doc_length)
            source_counts[source] += 1
            lengths_by_source[source].append(doc_length)
    
    if not lengths:
        print("No documents found")
        return
    
    # Basic stats
    print("\n=== OVERALL STATISTICS ===")
    print(f"Total documents: {len(lengths):,}")
    print(f"Max length: {max(lengths):,} chars")
    print(f"Min length: {min(lengths):,} chars")
    print(f"Average length: {sum(lengths)/len(lengths):,.2f} chars")
    
    # Source distribution
    print("\n=== DOCUMENTS BY SOURCE ===")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(lengths)
        avg_len = sum(lengths_by_source[source]) / len(lengths_by_source[source])
        print(f"{source}: {count:,} ({pct:.1f}%) - avg length: {avg_len:,.0f} chars")
    
    # Statistics per source
    print("\n=== STATISTICS BY SOURCE ===")
    for source in sorted(lengths_by_source.keys()):
        src_lengths = lengths_by_source[source]
        print(f"\n{source.upper()}:")
        print(f"  Count: {len(src_lengths):,}")
        print(f"  Max: {max(src_lengths):,} chars")
        print(f"  Min: {min(src_lengths):,} chars")
        print(f"  Average: {sum(src_lengths)/len(src_lengths):,.2f} chars")
        print(f"  Median: {sorted(src_lengths)[len(src_lengths)//2]:,} chars")
    
    # Percentiles (overall)
    print("\n=== OVERALL LENGTH DISTRIBUTION (Percentiles) ===")
    sorted_lengths = sorted(lengths)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        idx = int(len(sorted_lengths) * p / 100)
        print(f"{p:>3}th: {sorted_lengths[idx]:>8,} chars")
    
    # Length buckets (overall)
    print("\n=== OVERALL LENGTH BUCKETS ===")
    buckets = [
        ("Very short (< 50)", lambda x: x < 50),
        ("Short (50-200)", lambda x: 50 <= x < 200),
        ("Medium (200-1000)", lambda x: 200 <= x < 1000),
        ("Long (1000-5000)", lambda x: 1000 <= x < 5000),
        ("Very long (5000-10000)", lambda x: 5000 <= x < 10000),
        ("Extremely long (> 10000)", lambda x: x >= 10000),
    ]
    
    for label, condition in buckets:
        count = sum(1 for l in lengths if condition(l))
        pct = 100 * count / len(lengths)
        bar = "█" * int(pct / 2)
        print(f"{label:30} {count:>8,} ({pct:>5.2f}%) {bar}")
    
    # Try to create visualizations (optional, requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        print("\n=== GENERATING PLOTS ===")
        
        # Get the 3 main sources
        sources = sorted(lengths_by_source.keys())
        n_sources = len(sources)
        
        # Create figure with subplots for each source + overall (4 rows × 4 cols)
        fig = plt.figure(figsize=(20, 16))
        
        # Calculate 99th percentile to set reasonable x-axis limits
        p99_overall = np.percentile(lengths, 99)
        
        # Overall distributions (top row)
        # 1. Overall Histogram
        ax1 = plt.subplot(4, 4, 1)
        ax1.hist(lengths, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.set_xlabel('Document Length (characters)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Overall Distribution (99th percentile view)')
        ax1.set_xlim(0, p99_overall)
        ax1.grid(True, alpha=0.3)
        
        # 2. Overall Box Plot
        ax2 = plt.subplot(4, 4, 2)
        ax2.boxplot(lengths, vert=True)
        ax2.set_ylabel('Document Length (characters)')
        ax2.set_title('Overall Box Plot (99th %ile view)')
        ax2.set_ylim(0, p99_overall)
        ax2.grid(True, alpha=0.3)
        
        # 3. Overall CDF
        ax3 = plt.subplot(4, 4, 3)
        sorted_all = sorted(lengths)
        cdf_all = np.arange(1, len(sorted_all) + 1) / len(sorted_all) * 100
        ax3.plot(sorted_all, cdf_all, color='steelblue', linewidth=2)
        ax3.set_xlabel('Document Length (characters)')
        ax3.set_ylabel('Cumulative %')
        ax3.set_title('Overall CDF (99th percentile view)')
        ax3.set_xlim(0, p99_overall)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Median')
        ax3.axhline(y=95, color='orange', linestyle='--', alpha=0.5, label='95th')
        
        # 4. Comparison: All sources on one plot
        ax4 = plt.subplot(4, 4, 4)
        colors = ['steelblue', 'coral', 'mediumseagreen']
        for idx, source in enumerate(sources):
            src_lengths = lengths_by_source[source]
            ax4.hist(src_lengths, bins=50, alpha=0.5, label=source, color=colors[idx % len(colors)])
        ax4.set_xlabel('Document Length (characters)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('All Sources Comparison (99th percentile view)')
        ax4.set_xlim(0, p99_overall)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Individual source distributions (rows 2-4)
        colors_main = ['steelblue', 'coral', 'mediumseagreen']
        
        for idx, source in enumerate(sources):
            src_lengths = lengths_by_source[source]
            color = colors_main[idx % len(colors_main)]
            row = idx + 1  # Row 1, 2, 3 for the 3 sources
            
            # Calculate 99th percentile for this source
            p99_source = np.percentile(src_lengths, 99)
            
            # Histogram
            ax_hist = plt.subplot(4, 4, row * 4 + 1)
            ax_hist.hist(src_lengths, bins=50, edgecolor='black', alpha=0.7, color=color)
            ax_hist.set_xlabel('Length (chars)')
            ax_hist.set_ylabel('Frequency')
            ax_hist.set_title(f'{source.upper()} - Distribution (99th %ile)')
            ax_hist.set_xlim(0, p99_source)
            ax_hist.grid(True, alpha=0.3)
            
            # Box Plot
            ax_box = plt.subplot(4, 4, row * 4 + 2)
            ax_box.boxplot(src_lengths, vert=True)
            ax_box.set_ylabel('Length (chars)')
            ax_box.set_title(f'{source.upper()} - Box Plot (99th %ile)')
            ax_box.set_ylim(0, p99_source)
            ax_box.grid(True, alpha=0.3)
            
            # CDF
            ax_cdf = plt.subplot(4, 4, row * 4 + 3)
            sorted_src = sorted(src_lengths)
            cdf_src = np.arange(1, len(sorted_src) + 1) / len(sorted_src) * 100
            ax_cdf.plot(sorted_src, cdf_src, color=color, linewidth=2)
            ax_cdf.set_xlabel('Length (chars)')
            ax_cdf.set_ylabel('Cumulative %')
            ax_cdf.set_title(f'{source.upper()} - CDF (99th %ile)')
            ax_cdf.set_xlim(0, p99_source)
            ax_cdf.grid(True, alpha=0.3)
            ax_cdf.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Median')
            ax_cdf.axhline(y=95, color='orange', linestyle='--', alpha=0.5, label='95th')
            
            # Stats text
            ax_stats = plt.subplot(4, 4, row * 4 + 4)
            ax_stats.axis('off')
            stats_text = (
                f"{source.upper()}\n\n"
                f"Count: {len(src_lengths):,}\n"
                f"Max: {max(src_lengths):,}\n"
                f"Min: {min(src_lengths):,}\n"
                f"Mean: {np.mean(src_lengths):,.0f}\n"
                f"Median: {np.median(src_lengths):,.0f}\n"
                f"Std: {np.std(src_lengths):,.0f}\n\n"
                f"Percentiles:\n"
                f"25th: {np.percentile(src_lengths, 25):,.0f}\n"
                f"50th: {np.percentile(src_lengths, 50):,.0f}\n"
                f"75th: {np.percentile(src_lengths, 75):,.0f}\n"
                f"95th: {np.percentile(src_lengths, 95):,.0f}"
            )
            ax_stats.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle(f'Document Length Distribution - {Path(corpus_path).name}', fontsize=16, y=0.995)
        plt.tight_layout()
        
        # Save plot
        output_path = corpus_path.replace('.jsonl', '_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        
        # Show plot
        plt.show()
        print("Done!")
        
    except ImportError:
        print("\n[Note] matplotlib not installed - skipping plots")
        print("Install with: pip install matplotlib")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        corpus_path = sys.argv[1]
    else:
        # Default to main corpus
        corpus_path = "data/corpus/med_pure_corpus.jsonl"
    
    analyze_corpus(corpus_path)
